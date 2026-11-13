"""A1（V0.7）：PatchCore-lite 无监督异常检测（自增优化，用户指定优先）。

对照 VIMO"无监督"：只用 OK/良品图建特征记忆库，输出异常热力图/检出——
命中"缺陷样本稀缺、正常样本充足"的工业主场景。纯 torch 实现（骨干
torchvision resnet18 离线权重，不新增 pip 依赖）。

管线（PatchCore-lite 简化版）：
1. 特征：resnet18 layer2(128ch, stride 8) + layer3(256ch, stride 16)，
   各做 3×3 平均池化聚合（邻域感知），layer3 上采样到 layer2 分辨率后
   拼接 → 每位置 384 维 patch 向量；
2. 记忆库：全部 OK 图 patch 随机子采样（上限 5 万）入库存 float16；
3. 标定：留出 10% OK 图不入库，其 patch 距离分布给出 mu/sigma——预测时
   距离标准化为 z 分数，阈值语义 = 距离超出 OK 分布多少个标准差；
4. 推理：逐 patch 最近邻距离 → 标准化 → z 阈值（conf 参数映射
   z = 1/conf - 1，conf 0.25 ⇒ z=3）连通域成框 + 掩码 + 图像级分数。

指标：图像级 AUROC / AP（NG 图 + 留出 OK 图），OK 误报率。
TRT 导出：暂不支持（记忆库非标准算子图；.pt 交付，文档注明）。
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from ..dataset.base import TaskType
from ..dataset.reader import is_negative_dir_name
from ..io_utils import imread_unicode, save_json
from .base import BaseModel, Prediction, TrainConfig, TrainReporter

log = logging.getLogger(__name__)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_MEMORY_CAP = 50_000       # 记忆库 patch 上限（随机子采样）
_CALIB_RATIO = 0.10        # 留出标定的 OK patch 比例
_TOPK_RATIO = 0.01         # 图像分数 = 距离最大的 1% patch 均值


def _z_from_conf(conf: float) -> float:
    """conf∈(0,1] → z 阈值：conf 0.25 ⇒ z=3.0，0.5 ⇒ 1.0（单调、可解释）。"""
    conf = min(max(float(conf), 1e-3), 1.0)
    return 1.0 / conf - 1.0


def image_auroc(scores: list, labels: list) -> float:
    """图像级 AUROC（秩和法，纯 numpy；标签 1=异常/0=正常）。"""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    if len(s) < 2 or len(np.unique(y)) < 2:
        return 0.0
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # 并列分数取平均秩
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def image_ap(scores: list, labels: list) -> float:
    """图像级 AP（按分数降序的精度-召回积分，纯 numpy）。"""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    n1 = int(y.sum())
    if n1 == 0 or len(s) == 0:
        return 0.0
    order = np.argsort(-s)
    hits = y[order] == 1
    prec = np.cumsum(hits) / (np.arange(len(hits)) + 1)
    return float((prec * hits).sum() / n1)


class _FeatureTap:
    """resnet18 layer2+layer3 前向钩子（输出池化聚合后的 patch 特征）。"""

    def __init__(self, backbone: torchvision.models.ResNet):
        self._backbone = backbone
        self._feats: list = []
        self._hooks = [
            backbone.layer2.register_forward_hook(self._make(0)),
            backbone.layer3.register_forward_hook(self._make(1)),
        ]

    def _make(self, idx: int):
        def hook(_module, _inp, out):
            self._feats[idx] = out
        return hook

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """输入归一化张量 (N,3,H,W) → (N, 384, H/8, W/8) patch 特征。"""
        self._feats = [None, None]
        with torch.no_grad():
            # 只跑到 layer3（layer4 跳过，省算力）
            x = self._stem(x)
        f2, f3 = self._feats
        if f2 is None or f3 is None:
            raise RuntimeError("特征钩子未触发（骨干结构异常）")
        f2 = F.avg_pool2d(f2, 3, stride=1, padding=1)
        f3 = F.avg_pool2d(f3, 3, stride=1, padding=1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear",
                           align_corners=False)
        return torch.cat([f2, f3], dim=1)

    def _stem(self, x: torch.Tensor) -> torch.Tensor:
        b = self._backbone
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.relu(x)
        x = b.maxpool(x)
        x = b.layer1(x)
        x = b.layer2(x)
        x = b.layer3(x)
        return x

    def close(self):
        for h in self._hooks:
            h.remove()


class AnomalyModel(BaseModel):
    """PatchCore-lite：OK 图记忆库 + 最近邻 z 分数。"""

    name = "patchcore"
    family = "folder"
    supported_tasks = (TaskType.ANOMALY,)
    default_image_size = 256

    def __init__(self, num_classes: int = 1, device: str = "auto",
                 use_pretrained: bool = True):
        super().__init__(num_classes, device)
        self.image_size = self.default_image_size
        self.backbone = self._build_backbone(use_pretrained)
        self.backbone.to(self.device).eval()
        self.tap = _FeatureTap(self.backbone)
        self.memory: torch.Tensor | None = None   # (M, 384) float16 cpu
        self.dist_mu = 0.0
        self.dist_sigma = 1.0
        self.score_mu = 0.0
        self.score_sigma = 1.0
        self.classes = ["异常"]

    # ---- 骨干 ----
    def _build_backbone(self, use_pretrained: bool):
        weights = None
        if use_pretrained:
            try:
                weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
                return torchvision.models.resnet18(weights=weights)
            except Exception:  # 旧 torchvision 或离线无缓存：退化为随机骨干
                log.warning("ResNet18 预训练权重不可用（离线/旧版本），异常检测"
                            "改用随机初始化骨干；建议预置 torchvision 权重缓存",
                            exc_info=True)
                return torchvision.models.resnet18(weights=None)
        return torchvision.models.resnet18(weights=weights)

    # ---- 预处理 ----
    def _prepare(self, img_bgr: np.ndarray) -> torch.Tensor:
        size = self.image_size
        rgb = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
        x = rgb[:, :, ::-1].astype(np.float32) / 255.0
        x = (x - np.array(_IMAGENET_MEAN)) / np.array(_IMAGENET_STD)
        t = torch.from_numpy(x.astype(np.float32)).permute(2, 0, 1)
        return t.unsqueeze(0).to(self.device)

    def _patch_dists(self, img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(异常 z 图（img_size/8 网格）, 图像级原始距离分数)。"""
        feats = self.tap.extract(self._prepare(img_bgr))[0]   # (C, h, w)
        c, h, w = feats.shape
        flat = feats.reshape(c, -1).t()                       # (h*w, C) fp32
        mem = self.memory.to(flat.device)                     # (M, C) fp16
        # 分块最近邻（防大图 × 大库一次性 cdist 爆显存）
        chunk = max(1, 4096)
        mins = []
        for i in range(0, flat.shape[0], chunk):
            d = torch.cdist(flat[i:i + chunk].float(), mem.float())
            mins.append(d.min(dim=1).values)
        dist = torch.cat(mins).reshape(h, w)
        k = max(1, int(round(h * w * _TOPK_RATIO)))
        img_score = float(dist.flatten().topk(k).values.mean())
        return dist.cpu().numpy(), img_score

    # ---- 训练 ----
    def fit(self, data, cfg: TrainConfig, reporter: TrainReporter | None = None) -> dict:
        reporter = reporter or TrainReporter()
        self.image_size = cfg.image_size or self.default_image_size
        ok_labels = {c for c in (data.classes or [])
                     if is_negative_dir_name(c)}
        pool = ([s for s in data.train_samples if s.label in ok_labels]
                if ok_labels else list(data.train_samples))
        if not pool:  # 数据全为 NG：按 label 不在 ok_labels 兜底取全部
            pool = list(data.train_samples)
        if not pool:
            raise ValueError("异常检测需要 OK/良品图（NEGATIVE_DIR_NAMES 目录）"
                             "或至少一组正常样本图")
        val_ok = [s for s in data.val_samples
                  if (ok_labels and s.label in ok_labels) or not ok_labels]
        val_ng = [s for s in data.val_samples
                  if ok_labels and s.label not in ok_labels]
        started = datetime.now()
        rng = np.random.default_rng(cfg.seed)

        # 1) 抽取 OK patch（随机留出 _CALIB_RATIO 做标定，不入库）
        patches: list = []
        calib: list = []
        n_img = len(pool)
        for i, s in enumerate(pool, start=1):
            img = imread_unicode(s.image_path)
            if img is None:
                continue
            feats = self.tap.extract(self._prepare(img))[0]     # (C,h,w)
            vec = feats.reshape(feats.shape[0], -1).t()         # (h*w, C)
            n = vec.shape[0]
            keep = max(1, int(n * (1.0 - _CALIB_RATIO)))
            idx = rng.permutation(n)
            patches.append(vec[idx[:keep]])
            calib.append(vec[idx[keep:]])
            if reporter is not None:
                reporter.on_train_progress(1, i, n_img)
            if i % 50 == 0:
                log.info("异常检测特征抽取 %d/%d", i, n_img)
        if not patches:
            raise ValueError("OK 图特征抽取失败（图片全部无法读取）")
        all_p = torch.cat(patches)
        if all_p.shape[0] > _MEMORY_CAP:
            sel = rng.choice(all_p.shape[0], _MEMORY_CAP, replace=False)
            all_p = all_p[sel]
        self.memory = all_p.cpu().to(torch.float16)
        log.info("记忆库构建完成: %d patch（%s 维，留出标定 %.0f%%）",
                 self.memory.shape[0], self.memory.shape[1],
                 _CALIB_RATIO * 100)

        # 2) 留出 OK patch 距离标定（mu/sigma）
        calib_d = self._calib_dists(torch.cat(calib)) if calib else None
        if calib_d is not None and len(calib_d) > 100:
            self.dist_mu = float(np.mean(calib_d))
            self.dist_sigma = max(float(np.std(calib_d)), 1e-6)
        else:
            self.dist_mu, self.dist_sigma = 0.0, 1.0
        # 图像级分数标定：留出 OK 图的 topk 均值分布
        ok_scores = []
        for s in pool[:min(len(pool), 200)]:
            img = imread_unicode(s.image_path)
            if img is None:
                continue
            _, sc = self._patch_dists(img)
            ok_scores.append(sc)
        if len(ok_scores) >= 5:
            self.score_mu = float(np.mean(ok_scores))
            self.score_sigma = max(float(np.std(ok_scores)), 1e-6)

        # 3) 验证集指标（NG + 留出 OK；全 OK 数据集时无 AUROC）
        scores, labels = [], []
        for s in val_ng:
            img = imread_unicode(s.image_path)
            if img is not None:
                scores.append(self._patch_dists(img)[1])
                labels.append(1)
        for s in val_ok:
            img = imread_unicode(s.image_path)
            if img is not None:
                scores.append(self._patch_dists(img)[1])
                labels.append(0)
        auroc = image_auroc(scores, labels)
        ap = image_ap(scores, labels)
        z25 = _z_from_conf(0.25)
        n_fp = sum(1 for s, l in zip(scores, labels)
                   if l == 0 and (s - self.score_mu) / self.score_sigma >= z25)
        n_ok_v = sum(1 for l in labels if l == 0)
        # 权重落盘：平台自有明文容器 best.pt（社区版不加密）
        best_pt = Path(cfg.output_dir) / "best.pt"
        self.save_weights(best_pt)
        metrics = {
            "final": {"auroc": round(auroc, 4), "ap": round(ap, 4),
                      "ok_fp_rate": (round(n_fp / n_ok_v, 4) if n_ok_v else None)},
            "classes": self.classes,
            "memory_patches": int(self.memory.shape[0]),
            "train_ok_images": len(pool),
            "val_ng_images": len(val_ng),
            "val_ok_images": len(val_ok),
            "image_size": self.image_size,
            "history": [{"epoch": 1, "train_loss": 0.0, "auroc": round(auroc, 4)}],
        }
        metrics["start_time"] = started.strftime("%Y-%m-%d %H:%M:%S")
        metrics["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metrics["status"] = "completed"
        reporter.on_epoch(1, {"auroc": auroc})
        reporter.on_finish(True)
        return metrics

    def _calib_dists(self, calib: torch.Tensor, chunk: int = 8192) -> np.ndarray:
        mem = self.memory.to(calib.device)
        out = []
        for i in range(0, calib.shape[0], chunk):
            d = torch.cdist(calib[i:i + chunk].float(), mem.float())
            out.append(d.min(dim=1).values.cpu().numpy())
        return np.concatenate(out) if out else np.array([])

    # ---- 推理 ----
    def predict(self, image_bgr: np.ndarray, conf: float = 0.25,
                iou: float = 0.45) -> Prediction:
        if self.memory is None or self.memory.shape[0] == 0:
            raise RuntimeError("异常检测模型未训练（记忆库为空）")
        dist, img_score = self._patch_dists(image_bgr)
        z_map = (dist - self.dist_mu) / self.dist_sigma
        img_z = (img_score - self.score_mu) / self.score_sigma
        z_thr = _z_from_conf(conf)
        h_img, w_img = image_bgr.shape[:2]
        mask = (z_map > z_thr).astype(np.uint8)
        mask_full = cv2.resize(mask, (w_img, h_img),
                               interpolation=cv2.INTER_NEAREST)
        boxes, scores, masks = [], [], []
        contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        gh, gw = z_map.shape                     # patch 网格（img_size/8）
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < 2 or bh < 2:
                continue
            boxes.append([float(x), float(y), float(x + bw), float(y + bh)])
            # 组件分数 = 网格坐标下该框覆盖区域的最大 z（网格对齐换算）
            gx1 = max(0, min(gw - 1, int(x * gw / w_img)))
            gy1 = max(0, min(gh - 1, int(y * gh / h_img)))
            gx2 = max(gx1 + 1, min(gw, int((x + bw) * gw / w_img) + 1))
            gy2 = max(gy1 + 1, min(gh, int((y + bh) * gh / h_img) + 1))
            scores.append(round(float(z_map[gy1:gy2, gx1:gx2].max()), 4))
            # drawContours 不收 bool 图（P162 同族）：uint8 绘制后转 bool
            m = np.zeros((h_img, w_img), dtype=np.uint8)
            cv2.drawContours(m, [cnt], -1, 1, -1)
            masks.append(m.astype(bool))
        return Prediction(labels=[0] * len(boxes), scores=scores,
                          boxes=boxes, masks=masks or None,
                          probs=[round(float(img_z), 4)])

    # ---- 权重 ----
    def save_weights(self, path: Path) -> None:
        """xccv-anomaly 容器（平台自有明文 .pt 权重）。"""
        ckpt = {
            "format": "xccv-anomaly",
            "classes": self.classes,
            "image_size": self.image_size,
            "memory": self.memory,
            "dist_mu": self.dist_mu, "dist_sigma": self.dist_sigma,
            "score_mu": self.score_mu, "score_sigma": self.score_sigma,
            "backbone_state": {k: v.cpu() for k, v in
                               self.backbone.state_dict().items()},
        }
        torch.save(ckpt, path)

    def load_weights(self, source) -> None:
        """source：明文路径 / 解密后的 ckpt dict（registry 统一入口语义）。"""
        if isinstance(source, dict):
            ckpt = source
        else:
            ckpt = torch.load(source, map_location="cpu", weights_only=False)
        if ckpt.get("format") != "xccv-anomaly":
            raise ValueError("不是 xccv-anomaly 权重文件")
        self.classes = list(ckpt.get("classes") or ["异常"])
        self.image_size = int(ckpt.get("image_size") or self.default_image_size)
        self.memory = ckpt["memory"].cpu().to(torch.float16)
        self.dist_mu = float(ckpt.get("dist_mu", 0.0))
        self.dist_sigma = max(float(ckpt.get("dist_sigma", 1.0)), 1e-6)
        self.score_mu = float(ckpt.get("score_mu", 0.0))
        self.score_sigma = max(float(ckpt.get("score_sigma", 1.0)), 1e-6)
        state = ckpt.get("backbone_state")
        if state:
            try:
                self.backbone.load_state_dict(state)
                self.backbone.to(self.device).eval()
            except Exception:
                log.warning("异常检测骨干权重恢复失败（用当前骨干继续）",
                            exc_info=True)
