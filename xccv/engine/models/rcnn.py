"""R-CNN 家族封装：Faster R-CNN（检测）/ Mask R-CNN（实例分割），数据走 COCO 格式。"""

import logging
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
import torchvision
from torchvision.ops import sigmoid_focal_loss

from ..dataset.base import TaskType
from ..dataset.converters import _clipped_bbox, _seg_points
from ..dataset.transforms import maybe_apply_ops
from ..io_utils import imread_unicode
from ..metrics.detection import detection_confusion_matrix
from .modules import CBAM, replace_convs_with_dcn
from .base import BaseModel, Prediction, TrainConfig, TrainReporter

log = logging.getLogger(__name__)


class RcnnDataset(Dataset):
    """内存数据集：直接读取原图，训练侧在线增强，训练目标由 shapes 实时构造。

    取代旧的 COCO json + 图片导出目录方案（训练数据零落盘）。
    空标注图（OK 图，shapes 为空或全部无效）在训练侧保留为纯背景负样本
    （torchvision 原生支持空目标，兼容 OK 空标注图）；
    验证侧全部保留以保持评估口径完整。
    """

    def __init__(self, samples, classes: list[str], task: TaskType,
                 augment_ops: dict | None = None, train: bool = True,
                 augment_probs: dict | None = None,
                 tiling: dict | None = None):
        self.classes = list(classes)
        self.with_masks = task is TaskType.SEGMENTATION
        self.ops = dict(augment_ops) if (train and augment_ops) else None
        self.op_probs = dict(augment_probs or {})
        self.samples = list(samples)
        # T9-3 切图训练（opt-in）：预生成 (样本, x, y) 索引现场裁剪、不落盘；
        # 验证集不传 tiling（始终整图评估）
        tiling = dict(tiling) if (train and tiling) else None
        self._tile = int((tiling or {}).get("tile", 640))
        self._min_vis = float((tiling or {}).get("min_visibility", 0.3))
        self._entries = None
        if tiling:
            from ..dataset.tiling import plan_tile_entries

            self._entries = plan_tile_entries(
                self.samples, tile=self._tile,
                overlap=float(tiling.get("overlap", 0.2)),
                bg_tile_keep=float(tiling.get("bg_tile_keep", 0.15)),
                mix_full_view=float(tiling.get("mix_full_view", 0.1)),
                min_visibility=self._min_vis)

    def __len__(self) -> int:
        return (len(self._entries) if self._entries is not None
                else len(self.samples))

    def __getitem__(self, index: int):
        if self._entries is not None:
            return self._getitem_tiled(index)
        s = self.samples[index]
        img = imread_unicode(s.image_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {s.image_path}")
        shapes = list(s.shapes)
        if self.ops:  # 按每算子启用概率随机增强（图像与标注点同步变换）
            img, shapes = maybe_apply_ops(img, shapes, self.ops, probs=self.op_probs)
        h, w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        boxes, labels, areas, masks = [], [], [], []
        for sh in shapes:
            bbox = _clipped_bbox(sh, w, h)
            ci = self.classes.index(sh.label) if sh.label in self.classes else -1
            if bbox is None or ci < 0:
                continue
            x1, y1, x2, y2 = bbox
            boxes.append([x1, y1, x2, y2])
            labels.append(ci + 1)  # torchvision 0 号是背景
            areas.append((x2 - x1) * (y2 - y1))
            if self.with_masks:
                pts = _seg_points(sh, w, h)
                if pts is None:
                    masks.append(torch.zeros((h, w), dtype=torch.uint8))
                else:
                    rles = maskUtils.frPyObjects([[c for p in pts for c in p]], h, w)
                    rle = maskUtils.merge(rles)
                    masks.append(torch.from_numpy(maskUtils.decode(rle)))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([index + 1]),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.zeros(len(labels), dtype=torch.int64),
        }
        if self.with_masks:  # 空目标图也要有空的 masks 张量（maskrcnn 训练协议）
            target["masks"] = (torch.stack(masks).to(torch.uint8) if masks
                               else torch.zeros((0, h, w), dtype=torch.uint8))
        return img, target

    def _getitem_tiled(self, index: int):
        """切图条目（T9-3）：加载整图→在线增强（画布口径）→裁 tile→标签
        按中心归属/可见面积过滤（重叠区去重已在 crop_target_boxes 执行——
        框只归中心所在的第一个窗口，跨 tile 不重复）。x/y 为 None=整图混入。"""
        e = self._entries[index]
        s = self.samples[e.sample_idx]
        img = imread_unicode(s.image_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {s.image_path}")
        shapes = list(s.shapes)
        H, W = img.shape[:2]
        if e.x is None:  # 整图混入条目
            e_x, e_y, tile = 0, 0, W if W >= H else H
            crop = img
        else:
            e_x, e_y, tile = e.x, e.y, self._tile
            crop = img[e_y:e_y + tile, e_x:e_x + tile]
            if crop.size == 0:
                raise ValueError(
                    f"切图窗口越界: tile=({e_x},{e_y},{tile}) 图={W}x{H}")
        # 先裁剪后增强（修复：resize 增强可能把图缩到比规划窗口小，
        # 原顺序裁出空数组 → cvtColor 断言崩）；几何增强在 tile 内进行，
        # 标签点同步变换后无需再平移（同窗内）
        if self.ops:
            crop, shapes = maybe_apply_ops(crop, shapes, self.ops,
                                           probs=self.op_probs)
        th2, tw2 = crop.shape[:2]
        if (th2, tw2) != (tile, tile) and e.x is not None:
            # 几何增强改变了 tile 尺寸（resize 类）：标签平移量不变，
            # 裁回固定 tile 尺寸保证 batch 一致
            crop = crop[:tile, :tile]
            th2, tw2 = crop.shape[:2]
        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        th, tw = img.shape[1], img.shape[2]
        th, tw = int(th), int(tw)
        boxes, labels, areas, masks = [], [], [], []
        # 归属判定必须用【原始图坐标】（e.box_ids 由规划期基于原始坐标生成，
        # 增强会改变尺寸/翻转，索引语义只在原坐标系成立）
        raws, metas = [], []
        for sh in shapes:
            ci = self.classes.index(sh.label) if sh.label in self.classes else -1
            raw = _clipped_bbox(sh, W, H)
            if ci < 0 or raw is None:
                continue
            raws.append(raw)
            metas.append((ci, sh))
        from ..dataset.tiling import crop_target_boxes

        if e.x is None:  # 整图混入条目：全部框原样（box_ids=全量）
            keep_ids = [i for i in range(len(raws))]
        else:  # 按规划期唯一归属（box_ids）裁剪：跨 tile 不重复
            kept = crop_target_boxes(raws, e_x, e_y, tile, W, H,
                                     self._min_vis, only_ids=e.box_ids)
            keep_ids = [raw_i for raw_i, _b in kept]
        ch, cw = crop.shape[:2]   # 增强后的 tile 尺寸
        for raw_i in keep_ids:
            ci, sh = metas[raw_i]
            # 标签几何在【增强后 crop 坐标】内重算（增强已同步变换
            # shapes 点；不再减 e_x/e_y——原实现混用新旧坐标系产生
            # 负宽框 / 掩膜错位）
            bbox = _clipped_bbox(sh, cw, ch)
            if bbox is None:
                continue
            boxes.append(bbox)
            labels.append(ci + 1)
            areas.append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            if self.with_masks:
                pts = _seg_points(sh, cw, ch)
                if pts is None:
                    masks.append(torch.zeros((th, tw), dtype=torch.uint8))
                else:
                    # _seg_points 返回 [[x,y], ...] 点对列表（原按
                    # "多边形列表"双层解包抛 cannot unpack float，
                    # 切图+分割组合从未跑通）
                    flat = [c for p in pts for c in p]
                    rles = maskUtils.frPyObjects([flat], th, tw)
                    rle = maskUtils.merge(rles)
                    masks.append(torch.from_numpy(maskUtils.decode(rle)))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([index + 1]),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.zeros(len(labels), dtype=torch.int64),
        }
        if self.with_masks:
            target["masks"] = (torch.stack(masks).to(torch.uint8) if masks
                               else torch.zeros((0, th, tw),
                                                dtype=torch.uint8))
        return img, target


def memory_coco(samples, classes: list[str], task: str) -> COCO:
    """在内存中构建 COCO 评估对象（不写 json 文件；file_name 为原图绝对路径）。"""
    images, annotations = [], []
    ann_id = 1
    for img_id, s in enumerate(samples, start=1):
        images.append({"id": img_id, "file_name": str(s.image_path),
                       "width": s.width, "height": s.height})
        for sh in s.shapes:
            ci = classes.index(sh.label) if sh.label in classes else -1
            bbox = _clipped_bbox(sh, s.width, s.height)
            if ci < 0 or bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            ann = {"id": ann_id, "image_id": img_id, "category_id": ci + 1,
                   "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1),
                   "iscrowd": 0}
            if task == "segmentation":
                pts = _seg_points(sh, s.width, s.height)
                if pts is None:
                    continue  # 与旧导出逻辑一致：无效轮廓不参与评估
                ann["segmentation"] = [[c for p in pts for c in p]]
            annotations.append(ann)
            ann_id += 1
    coco = COCO.__new__(COCO)  # 绕过文件加载，直接内存建索引
    coco.annotation_file = None
    coco.dataset = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(classes)],
    }
    coco.createIndex()
    return coco


def _collate(batch):
    return tuple(zip(*batch))


class _BodyWithCBAM(nn.Module):
    """在 IntermediateLayerGetter 各 stage 输出后插入 CBAM，保持 FPN 接口不变。"""

    def __init__(self, inner, cbam_channels: dict):
        super().__init__()
        self.inner = inner
        self.return_layers = dict(inner.return_layers)
        self.cbams = nn.ModuleDict({k: CBAM(v) for k, v in cbam_channels.items()})
        self.grad_checkpoint = False  # 激活检查点开关（fit 按显存预算开启）

    def forward(self, x):
        from torch.utils.checkpoint import checkpoint

        use_ckpt = (self.grad_checkpoint and self.training
                    and torch.is_grad_enabled() and x.device.type == "cuda")
        out = {}
        t = x
        for name, module in self.inner.named_children():
            t = checkpoint(module, t, use_reentrant=False) if use_ckpt else module(t)
            if name in self.cbams:
                t = self.cbams[name](t)
            if name in self.return_layers:
                out[self.return_layers[name]] = t
        return out


def _deepest_stages(body: nn.Module, n: int = 2) -> dict[str, int]:
    """实测 body 最深 n 个 stage 的输出通道（键为 stage 名）。

    通道表不能硬编码：ResNet18 与 ResNet50 的 layer3/layer4 输出不同
    （256/512 与 1024/2048），按 R50 写死会在 maskrcnn18 上产生 CBAM
    通道失配（mat1/mat2 形状错误）。用假输入顺序流过各 stage 实测，
    遍历方式与 _BodyWithCBAM.forward 一致，适配任意 torchvision 骨干。
    """
    was_training = body.training
    body.eval()
    channels: dict[str, int] = {}
    with torch.no_grad():
        t = torch.zeros(1, 3, 256, 256)
        for name, module in body.named_children():
            try:
                t = module(t)
            except Exception:
                break  # 非特征子模块（如裸 ResNet 的分类头），特征 stage 已记录完
            if t.dim() != 4 or t.shape[-1] < 2 or t.shape[-2] < 2:
                break  # avgpool/flatten 等终止特征流的子模块不记录
            channels[name] = int(t.shape[1])
    if was_training:
        body.train()
    names = list(channels)[-n:]
    return {k: channels[k] for k in names}


# ---- 工业优化损失补丁（Focal 分类 + Dice+BCE 掩码），类级替换并支持恢复 ----
_RCNN_LOSS_PATCH = {"focal": False, "dice": False}
_RCNN_LOSS_ORIG: dict = {}


def _fastrcnn_loss_focal(class_logits, box_regression, labels, regression_targets):
    """分类损失 CE → Focal Loss（α=0.25, γ=2）；回归损失保持 SmoothL1。"""
    labels_t = torch.cat(labels, dim=0)
    regression_targets_t = torch.cat(regression_targets, dim=0)
    targets_one_hot = F.one_hot(labels_t, num_classes=class_logits.shape[1]).to(class_logits.dtype)
    classification_loss = sigmoid_focal_loss(
        class_logits, targets_one_hot, alpha=0.25, gamma=2.0, reduction="sum"
    ) / max(labels_t.numel(), 1)
    sampled_pos = torch.where(labels_t > 0)[0]
    labels_pos = labels_t[sampled_pos]
    n = class_logits.shape[0]
    box_regression = box_regression.reshape(n, box_regression.size(-1) // 4, 4)
    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos, labels_pos],
        regression_targets_t[sampled_pos],
        beta=1 / 9, reduction="sum",
    ) / labels_t.numel()
    return classification_loss, box_loss


def _apply_rcnn_loss_patches(with_dice: bool):
    from torchvision.models.detection import roi_heads

    if not _RCNN_LOSS_PATCH["focal"]:
        _RCNN_LOSS_ORIG["fastrcnn_loss"] = roi_heads.fastrcnn_loss
        roi_heads.fastrcnn_loss = _fastrcnn_loss_focal
        _RCNN_LOSS_PATCH["focal"] = True
    if with_dice and not _RCNN_LOSS_PATCH["dice"]:
        _RCNN_LOSS_ORIG["maskrcnn_loss"] = roi_heads.maskrcnn_loss

        def maskrcnn_loss_dice(mask_logits, mask_proposals, gt_masks, gt_labels, gt_matched_idxs):
            loss_bce = _RCNN_LOSS_ORIG["maskrcnn_loss"](
                mask_logits, mask_proposals, gt_masks, gt_labels, gt_matched_idxs)
            # 逐正样本投影 gt 掩码到 28×28，并保持与 mask_logits 行序一致
            targets, labels_t = [], []
            for masks_img, labels_img, props_img, idxs_img in zip(
                    gt_masks, gt_labels, mask_proposals, gt_matched_idxs):
                for pos_i, gt_idx in enumerate(idxs_img.tolist()):
                    if not (0 <= gt_idx < masks_img.shape[0]):
                        continue
                    box = props_img[pos_i]
                    mask = masks_img[gt_idx]
                    rois = torch.tensor([0.0] + box.tolist(), dtype=box.dtype,
                                        device=box.device).unsqueeze(0)
                    proj = torchvision.ops.roi_align(
                        mask[None, None].to(dtype=torch.float32), rois,
                        (mask_logits.shape[-1],) * 2, 1.0, 0, True)[0, 0]
                    targets.append(proj)
                    labels_t.append(labels_img[gt_idx])
            if not targets:
                return loss_bce
            targets_t = torch.stack(targets)
            labels_t = torch.stack(labels_t)
            rows = torch.arange(targets_t.shape[0], device=targets_t.device)
            probs = torch.sigmoid(mask_logits[rows, labels_t])
            inter = (probs * targets_t).sum(dim=(1, 2))
            card = probs.sum(dim=(1, 2)) + targets_t.sum(dim=(1, 2))
            dice = (1 - ((2 * inter + 1) / (card + 1))).mean()
            return 0.5 * loss_bce + 0.5 * dice

        roi_heads.maskrcnn_loss = maskrcnn_loss_dice
        _RCNN_LOSS_PATCH["dice"] = True


def _restore_rcnn_loss_patches():
    from torchvision.models.detection import roi_heads

    if _RCNN_LOSS_PATCH["focal"] and "fastrcnn_loss" in _RCNN_LOSS_ORIG:
        roi_heads.fastrcnn_loss = _RCNN_LOSS_ORIG["fastrcnn_loss"]
        _RCNN_LOSS_PATCH["focal"] = False
    if _RCNN_LOSS_PATCH["dice"] and "maskrcnn_loss" in _RCNN_LOSS_ORIG:
        roi_heads.maskrcnn_loss = _RCNN_LOSS_ORIG["maskrcnn_loss"]
        _RCNN_LOSS_PATCH["dice"] = False


def project_masks_on_boxes(gt_masks, proposals, matched_idxs, discretization_size):
    """把 gt 掩码按 proposal 框投影到 M×M 网格（与 torchvision 内部逻辑一致）。"""
    from torchvision.ops import roi_align

    if gt_masks.numel() == 0:
        return torch.zeros((0, discretization_size, discretization_size),
                           dtype=torch.float32)
    matched_boxes = torch.index_select(proposals, 0, matched_idxs)
    out = []
    for mask, box in zip(gt_masks, matched_boxes):
        rois = torch.tensor([0.0] + box.tolist(), dtype=box.dtype,
                            device=box.device).unsqueeze(0)
        out.append(roi_align(mask[None, None].to(dtype=torch.float32), rois,
                             (discretization_size, discretization_size),
                             1.0, 0, True)[0, 0])
    return torch.stack(out)


def app_paths_fallback() -> str:
    """安装态兜底目录（app_root()/weights）；开发态引擎目录不存在时用。"""
    try:
        from ...app_paths import app_root

        return str(Path(app_root()) / "weights")
    except Exception:
        return ""


def _load_slim_pretrained(net, num_classes: int) -> bool:
    """maskrcnn18+slim 头的平台级预训练注入（weights/maskrcnn18_slim_*.pth）。

    明文 .pth 预训练权重。骨干+FPN+RPN 严格加载；类别头按形状匹配
    （COCO 类数与用户类别数不一致的头层跳过，不静默丢弃）。
    缺文件/损坏只告警返回 False，不阻断训练。
    """
    from pathlib import Path

    from ..offline import WEIGHTS_DIR   # 约定：算法引擎权重目录（开发态=
    # xccv/engine/weights/，安装态=运行根 weights/，见 engine/offline.py）
    from .modules import load_state_dict_compat

    roots = [WEIGHTS_DIR, Path(app_paths_fallback())]
    cands = []
    for root in roots:
        cands += [root / "maskrcnn18_slim_init.pth",
                  root / "maskrcnn18_slim_coco.pth"]
    for cand in cands:
        if not cand.is_file():
            continue
        try:
            import torch

            ckpt = torch.load(cand, map_location="cpu", weights_only=False)
            sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            model_sd = net.state_dict()
            strict, skipped = {}, []
            for k, v in sd.items():
                if k in model_sd and model_sd[k].shape == v.shape:
                    strict[k] = v
                else:
                    skipped.append(k)
            # 头层（cls/mask logits 等）跳过属预期；骨干缺键才告警
            missing = [k for k in model_sd
                       if k not in strict and not any(
                           k.endswith(s) for s in
                           ("cls_score.weight", "cls_score.bias",
                            "bbox_pred.weight", "bbox_pred.bias",
                            "mask_fcn_logits.weight", "mask_fcn_logits.bias"))]
            if missing:
                log.warning("slim 预训练缺骨干键（仅补齐可加载部分）: %s",
                            missing[:6])
            if skipped:
                log.info("slim 预训练头层按类别数不匹配跳过 %d 个（新头随机初始化）",
                         len(skipped))
            net.load_state_dict(strict, strict=False)
            log.info("已注入 slim 预训练权重：%s", cand.name)
            return True
        except Exception:  # noqa: BLE001  预训练失败不阻断训练
            log.warning("slim 预训练权重加载失败（忽略）：%s", cand.name,
                        exc_info=True)
    return False


def _apply_train_input(net, train_input: int, canvas_short: int,
                       arch: str) -> int:
    """T9-2 两段式缩放（opt-in）：画布增强保持画布口径，入网前由
    GeneralizedRCNNTransform 按 keep_ratio 缩到显式 train_input（points/
    boxes/masks 由 transform 同步缩放）。

    仅 R-CNN 家族生效；train_input 须为 32 的倍数且小于画布短边，否则不
    启用（返回 0=跟随画布=默认现状）。返回实际生效的网络输入短边。
    """
    ti = int(train_input or 0)
    if arch not in ("fasterrcnn", "maskrcnn", "maskrcnn18"):
        return 0
    if not ti:
        return 0
    if ti % 32 or ti >= canvas_short:
        log.warning("两段式缩放未启用：train_input=%s 需为 32 的倍数且小于"
                    "画布短边 %s（已回退为跟随画布）", ti, canvas_short)
        return 0
    t = getattr(net, "transform", None)
    if t is None:  # 非 torchvision 检测结构（如剪枝后替换网络）
        return 0
    t.min_size = (ti,)
    t.max_size = (round(ti * 4 / 3),)
    log.info("两段式缩放启用: 画布短边 %s → 网络输入 %s（max %s）",
             canvas_short, ti, round(ti * 4 / 3))
    return ti


def _rcnn_train_step(net, imgs, tgts, opt, scaler, use_amp: bool,
                     clip_norm: float = 10.0, accum: int = 1,
                     accum_idx: int = 0):
    """单步训练：返回 (有限损失 or None, 分量字典)。

    损失非有限（NaN/Inf）时跳过本步（不更新参数、不计入轮平均）并返回
    None——调用方累计跳过数，持续出现则中止训练。这样 NaN 不再污染整条
    损失曲线，分量值写入日志后可直接定位来源。
    梯度裁剪（unscale 后 L2 范数 clip_norm）：检测模型 + AMP 的标准防发散手段。
    梯度累积（accum>1）：accum_idx 为批内序号（1 起），组首清零梯度、组末
    裁剪+更新——物理小批次凑等效大批次；sched 由调用方按同序号推进。
    """
    if accum <= 1 or (accum_idx - 1) % accum == 0:
        opt.zero_grad(set_to_none=True)  # T9-6：梯度清零不驻留旧张量
    with torch.amp.autocast("cuda", enabled=use_amp):
        loss_dict = net(imgs, tgts)
        loss = sum(loss_dict.values())
    if not torch.isfinite(loss):
        return None, {k: float(v) for k, v in loss_dict.items()}
    scaler.scale(loss).backward()
    if accum <= 1 or accum_idx % accum == 0:
        scaler.unscale_(opt)  # AMP 关闭时为 no-op，裁剪仍作用于真实梯度
        torch.nn.utils.clip_grad_norm_(
            [p for p in net.parameters() if p.requires_grad], clip_norm)
        scaler.step(opt)
        scaler.update()
    return loss, loss_dict


def _nan_abort(nan_steps: int, n_batches: int) -> bool:
    """非有限步占比过高（≥30% 且 ≥5 步）时应中止训练。"""
    return nan_steps >= max(5, 0.3 * n_batches)


def _anchor_base(min_size: int) -> float:
    """小目标锚框金字塔底径：入网短边/80（夹 8~64）。1280→16，1920→24。"""
    return min(max(float(min_size) / 80.0, 8.0), 64.0)


def _vram_arch_factor(arch: str) -> float:
    """显存经验值按 R50 校准，轻量骨干（R18-FPN64）按实测缩水。"""
    return {"maskrcnn18": 0.45}.get(arch, 1.0)


def _vram_batch_cap(requested: int, min_size: int, use_amp: bool,
                    grad_ckpt: bool = False, arch: str = "") -> int:
    """按当前空闲显存估算训练 batch 上限（R-CNN 大图训练的 OOM 防护）。

    经验值：800px 单图 fp32 ≈ 2.2GB / AMP ≈ 1.4GB，随像素数线性放大；
    梯度检查点启用后激活显存约省 60%；预算取空闲显存的 85%。
    仅在估算上限低于用户设置时生效（下调并记录日志）。
    """
    try:
        free_b, _total = torch.cuda.mem_get_info()
    except Exception:
        return requested
    per_img_gb = (max(min_size, 320) / 800.0) ** 2 * (1.4 if use_amp else 2.2)
    per_img_gb *= _vram_arch_factor(arch)
    if grad_ckpt:
        per_img_gb *= 0.4
    budget_gb = (free_b / 2**30) * 0.85
    cap = max(1, int(budget_gb / max(per_img_gb, 0.1)))
    return max(1, min(requested, cap))


def _apply_grad_checkpoint(net) -> list:
    """给 backbone body / FPN 挂激活检查点（显存换算力的 OOM 防护）。

    实例级 forward 补丁：不改模块树、不污染 state_dict（保存/加载/剪枝
    均不受影响）；训练前向按 stage 重算激活，评估与推理自动走原路径。
    返回补丁清单，用后须 _restore_grad_checkpoint（保存 net_module 前必恢复）。
    """
    from torch.utils.checkpoint import checkpoint as ckpt

    patched = []
    body = net.backbone.body
    if isinstance(body, _BodyWithCBAM):
        body.grad_checkpoint = True
        patched.append(("cbam-flag", body))
    else:
        orig_body = body.forward

        def body_forward(x, _f=orig_body, _b=body):
            if not (_b.training and torch.is_grad_enabled()
                    and x.device.type == "cuda"):
                return _f(x)
            result, t = {}, x
            for name, module in _b.named_children():
                t = ckpt(module, t, use_reentrant=False)
                if name in _b.return_layers:
                    result[_b.return_layers[name]] = t
            return result

        body.forward = body_forward
        patched.append(("patched", body))

    fpn = net.backbone.fpn
    orig_fpn = fpn.forward

    def fpn_forward(x, _f=orig_fpn, _m=fpn):
        if _m.training and torch.is_grad_enabled():
            return ckpt(_f, x, use_reentrant=False)
        return _f(x)

    fpn.forward = fpn_forward
    patched.append(("patched", fpn))
    return patched


def _restore_grad_checkpoint(patched: list) -> None:
    for kind, module in patched:
        if kind == "cbam-flag":
            module.grad_checkpoint = False
        else:
            module.__dict__.pop("forward", None)


def _install_cpu_mask_postprocess(net) -> None:
    """推理/验证期把 mask 后处理挪到 CPU 分块执行（用后需 _restore）。

    torchvision 的 postprocess 会把每条检出的 mask 贴回原图尺寸
    （一次性申请 N×oH×oW×4B 显存）：工业大图（4000px+）× 100 条检出
    可达数 GB，是 16GB 显卡在验证阶段 OOM 的直接原因。改为 CPU 上
    每 8 条一块贴回，结果完全一致，GPU 峰值显存近乎为零。
    """
    from torchvision.models.detection.roi_heads import paste_masks_in_image
    from torchvision.models.detection.transform import resize_boxes

    def postprocess(result, image_shapes, original_image_sizes):
        for pred, im_s, o_im_s in zip(result, image_shapes, original_image_sizes):
            boxes = resize_boxes(pred["boxes"], im_s, o_im_s)
            pred["boxes"] = boxes
            if "masks" in pred:
                masks = pred.pop("masks").cpu()  # (N,1,H,W) 模型输入尺度
                pasted = [paste_masks_in_image(masks[i:i + 8],
                                               boxes[i:i + 8].cpu(), o_im_s)
                          for i in range(0, masks.shape[0], 8)]
                pred["masks"] = torch.cat(pasted) if pasted else \
                    masks.new_zeros((0, 1, o_im_s[0], o_im_s[1]))
        return result

    net.transform.postprocess = postprocess


def _restore_mask_postprocess(net) -> None:
    """卸载实例级 postprocess 补丁，恢复类上的原实现。"""
    net.transform.__dict__.pop("postprocess", None)


class RCNNModel(BaseModel):
    family = "coco"
    default_image_size = 640

    def __init__(self, arch: str, task: TaskType, num_classes: int,
                 device: str = "auto", use_pretrained: bool = True,
                 industrial: bool = False, mask_head_light: bool = False):
        super().__init__(num_classes, device)
        self.arch = arch
        self.task = TaskType(task)
        self.with_masks = self.task is TaskType.SEGMENTATION
        self.industrial = industrial
        self.mask_head_light = mask_head_light  # 随 ckpt/引擎容器透传（重建头结构必需）
        self._min_size = self.default_image_size
        self.net = self._build(arch, num_classes, self._min_size, use_pretrained,
                               industrial, mask_head_light=mask_head_light).to(self.device)

    def _build(self, arch: str, num_classes: int, min_size: int, use_pretrained: bool,
               industrial: bool = False, freeze_backbone: bool = False,
               rpn_reduced: bool = False, mask_head_light: bool = False,
               max_size: int | None = None, slim_head: bool = False,
               anchor_base: float | None = None):
        bb = "DEFAULT" if use_pretrained else None
        max_size = max_size or int(min_size * 1333 / 800)
        num_classes = num_classes + 1  # torchvision 检测模型 0 号是背景
        # 冻结骨干：torchvision 官方机制，0=整个 body 冻结（FPN/RPN/头仍可训练）；
        # DCN 替换与 CBAM 包装发生在构造之后，新建模块天然可训练（先冻结后插入）
        trainable_layers = 0 if (freeze_backbone and use_pretrained) else None
        extra = dict(rpn_post_nms_top_n_train=256, rpn_post_nms_top_n_test=100) \
            if rpn_reduced else {}
        if arch == "fasterrcnn":
            net = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=bb,
                                          num_classes=num_classes, min_size=min_size,
                                          max_size=max_size,
                                          trainable_backbone_layers=trainable_layers, **extra)
        elif arch == "maskrcnn":
            net = maskrcnn_resnet50_fpn(weights=None, weights_backbone=bb,
                                        num_classes=num_classes, min_size=min_size,
                                        max_size=max_size,
                                        trainable_backbone_layers=trainable_layers, **extra)
        elif arch == "maskrcnn18":
            # 轻量实例分割：ResNet18 骨干 + FPN + 标准 Mask R-CNN 头。
            # 参数量/显存约为 R50 版的一半，小数据集工业缺陷任务上精度接近；
            # ImageNet 骨干权重走随包 resnet18-f37072fd.pth（离线可用）
            from torchvision.models.detection import MaskRCNN
            from torchvision.models.resnet import ResNet18_Weights
            from torchvision.models.detection.backbone_utils import (
                BackboneWithFPN, resnet_fpn_backbone)

            weights18 = ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None
            if slim_head:
                # T9-5 slim（FPN64，约 12.5M）：FPN 输出 64 通道；
                # 官方 _resnet_fpn_extractor 硬编码 256，须直构 BackboneWithFPN。
                # 冻结口径与官方一致（从 layer4 起数 trainable_layers）。
                from torchvision.models import resnet18
                from torchvision.ops.misc import FrozenBatchNorm2d

                body = resnet18(weights=weights18, norm_layer=FrozenBatchNorm2d)
                n_train = 3 if trainable_layers is None else trainable_layers
                keep = ["layer4", "layer3", "layer2", "layer1",
                        "conv1"][:n_train]
                for name, param in body.named_parameters():
                    if all(not name.startswith(k) for k in keep):
                        param.requires_grad_(False)
                backbone = BackboneWithFPN(
                    body,
                    return_layers={"layer1": "0", "layer2": "1",
                                   "layer3": "2", "layer4": "3"},
                    in_channels_list=[64, 128, 256, 512], out_channels=64)
            else:
                backbone = resnet_fpn_backbone(
                   backbone_name="resnet18", weights=weights18,
                   # T9-1 freeze bug：不能写 `trainable_layers or 3`——冻结标志 0
                   # 是假值会被覆盖成 3（骨干冻结在 maskrcnn18 上从未生效）
                   trainable_layers=3 if trainable_layers is None
                   else trainable_layers)
            net = MaskRCNN(backbone, num_classes=num_classes, min_size=min_size,
                           max_size=max_size, **extra)
        else:
            raise ValueError(f"未知 R-CNN 模型: {arch}")
        if arch in ("maskrcnn", "maskrcnn18") and mask_head_light \
                and not (arch == "maskrcnn18" and slim_head):
            # 轻量掩码头（构造后替换，避免构造器 num_classes 与自定义预测器的互斥约束）：
            # 4 层 256 通道 → 2 层 128 通道；输出仍是 28×28（转置卷积上采样保留），
            # 与工业 Dice 损失按 mask_logits 尺寸投影的逻辑天然兼容
            net.roi_heads.mask_head = MaskRCNNHeads(256, [128, 128], 1)
            net.roi_heads.mask_predictor = MaskRCNNPredictor(128, 128, num_classes)
        if arch == "maskrcnn18" and slim_head:
            # T9-5 slim 头组：FPN64 → 两层 256 FC + 4×64 掩码头（light 时 2×64）。
            # 键名按 torchvision 0.22 实测（TwoMLPHead 两参签名、mask 头 ModuleList
            # 索引直挂、predictor 键 conv5_mask/mask_fcn_logits）——转换器依赖此结构。
            from torchvision.models.detection.faster_rcnn import (
                FastRCNNPredictor, TwoMLPHead)

            oc = net.backbone.out_channels
            net.roi_heads.box_head = TwoMLPHead(oc * 7 * 7, 256)
            net.roi_heads.box_predictor = FastRCNNPredictor(256, num_classes)
            mask_layers = [64, 64] if mask_head_light else [64, 64, 64, 64]
            net.roi_heads.mask_head = MaskRCNNHeads(oc, mask_layers, 1)
            net.roi_heads.mask_predictor = MaskRCNNPredictor(oc, 64, num_classes)
            if use_pretrained:
                _load_slim_pretrained(net, num_classes)
        if anchor_base:
            # 小目标锚框自适应：torchvision 默认金字塔 (32,64,128,256,512)
            # 针对 800 输入/大目标设计——工业小缺陷与最小锚 IoU<0.1，RPN
            # 正样本只剩"每 GT 最优锚兜底"。底径=入网短边/80（夹 8~64，
            # 见 _anchor_base），最小锚与最小目标同量级；锚框无权重，不
            # 影响预训练注入。旧实现引用未定义的 size 静默 NameError，
            # 该块此前从未生效（历史 ckpt 不带 anchor_base 键→保持默认）。
            from torchvision.models.detection.rpn import AnchorGenerator

            net.rpn.anchor_generator = AnchorGenerator(
                sizes=tuple(anchor_base * (2 ** i) for i in range(5)),
                aspect_ratios=((0.5, 1.0, 2.0),) * 5)
            log.info("锚框自适应: 底径=%g 金字塔=%g..%g",
                     anchor_base, anchor_base, anchor_base * 16)
        if industrial:
            # 工业优化：最深两个 stage 的 3×3 conv → DCNv2；stage 输出后插 CBAM（code.md）。
            # stage 名与通道数按骨干实测（R18=256/512，R50=1024/2048），不硬编码
            stages = _deepest_stages(net.backbone.body, 2)
            for name in stages:
                replace_convs_with_dcn(getattr(net.backbone.body, name))
            net.backbone.body = _BodyWithCBAM(net.backbone.body, stages)
        return net

    # ---- 训练 ----
    def fit(self, data, cfg: TrainConfig, reporter: TrainReporter) -> dict:
        if cfg.image_width > 0 and cfg.image_height > 0:
            # 与增强 resize 统一：图像已是宽×高，短边/长边按宽高映射（保持长宽比不拉伸）
            size = min(cfg.image_width, cfg.image_height)
            max_size = max(cfg.image_width, cfg.image_height)
        else:
            size = cfg.image_size or self.default_image_size
            max_size = None
        self._min_size = size
        resume = cfg.resume_from_pruned and getattr(self, "resume_weights", False)
        anchor_used = None
        if getattr(cfg, "anchor_small", False):
            # 小目标锚框自适应（训练平台 UI 可选，默认关）：切图训练按 tile
            # 边长取底径，否则按入网短边；ckpt 记录 anchor_base 供加载重建
            eff = int(cfg.tile_size) if (getattr(cfg, "tiling", False)
                                         and int(cfg.tile_size) > 0) else size
            anchor_used = _anchor_base(eff)
        if not resume:
            if cfg.industrial_opt != self.industrial:
                self.industrial = cfg.industrial_opt
            # 锚框自适应默认关闭（B2 实测 1280/1920 底径 16/24 均未获益，
            # 见 runs/_trainres R2/R3）；勾选「小目标锚框」经 cfg.anchor_small 启用
            self.net = self._build(self.arch, self.num_classes, size, cfg.use_pretrained,
                                   cfg.industrial_opt, cfg.freeze_backbone, cfg.rpn_reduced,
                                   cfg.mask_head_light, max_size=max_size,
                                   slim_head=cfg.slim_head,
                                   anchor_base=anchor_used).to(self.device)
        elif cfg.freeze_backbone:  # 剪枝预训练模型：冻结骨干 body，FPN/RPN/头保持训练
            for p in self.net.backbone.body.parameters():
                p.requires_grad_(False)
        # T9-2 两段式缩放（opt-in）：在网络建好/恢复后改写 transform 入网尺寸
        applied_input = _apply_train_input(self.net, cfg.train_input, size,
                                           self.arch)

        if cfg.industrial_opt:
            _apply_rcnn_loss_patches(with_dice=self.with_masks)

        try:
            tiling_cfg = None
            if getattr(cfg, "tiling", False):
                tiling_cfg = {"tile": int(cfg.tile_size),
                              "overlap": float(cfg.tile_overlap),
                              "bg_tile_keep": float(cfg.bg_tile_keep),
                              "min_visibility": float(cfg.min_visibility),
                              "mix_full_view": float(cfg.mix_full_view)}
            train_set = RcnnDataset(data.train_samples, data.classes, self.task,
                                    cfg.augment_ops, train=True,
                                    augment_probs=cfg.augment_probs,
                                    tiling=tiling_cfg)
            # 切图训练：tile 按原尺度入网（网络变换尺寸=tile），验证/保存仍按画布
            # 尺寸——原先 tile 被网络变换放大到画布短边（640 tile ×3 放大），训练与
            # 整图评估尺度错位（E4 切图实验 0.156 的根因）
            self._tile_tsize = None
            if tiling_cfg and getattr(self.net, "transform", None) is not None:
                _t = self.net.transform
                self._tile_tsize = ((int(cfg.tile_size),), int(cfg.tile_size),
                                    tuple(_t.min_size), _t.max_size)
            if tiling_cfg:
                log.info("切图训练启用: tile=%s overlap=%s bg_keep=%s vis=%s "
                         "mix_full=%s → 训练条目 %d（验证集仍整图评估；推理侧"
                         "请配合分块推理）", cfg.tile_size, cfg.tile_overlap,
                         cfg.bg_tile_keep, cfg.min_visibility,
                         cfg.mix_full_view, len(train_set))
            val_set = RcnnDataset(data.val_samples, data.classes, self.task, train=False)
            val_set.coco = memory_coco(data.val_samples, data.classes, self.task.value)
            # 混合精度：GPU 上启用（缩短训练时间），CPU 自动关闭（界面可控）
            use_amp = cfg.use_amp and self.device == "cuda"
            # 显存预算：预估超出空闲显存时先启用梯度检查点（激活省约 60%），
            # 仍不够再下调 batch——大图训练常用更小输入，这里
            # 保证"大分辨率 + 用户批次"能在 16GB 级显卡跑通
            need_ckpt = False
            est_side = int(cfg.tile_size) if tiling_cfg else size
            if self.device == "cuda":
                per_img_gb = (max(est_side, 320) / 800.0) ** 2 * (1.4 if use_amp else 2.2)
                per_img_gb *= _vram_arch_factor(self.arch)
                try:
                    free_gb = torch.cuda.mem_get_info()[0] / 2**30
                except Exception:
                    free_gb = None
                need_ckpt = bool(free_gb and cfg.batch_size * per_img_gb > free_gb * 0.85)
            ckpt_patches = _apply_grad_checkpoint(self.net) if need_ckpt else []
            bs = cfg.batch_size
            if self.device == "cuda":
                bs = _vram_batch_cap(bs, est_side, use_amp, grad_ckpt=need_ckpt,
                                     arch=self.arch)
            # 物理批次被显存预算下调时，梯度累积凑回有效批次（优化稳定性）
            accum = max(1, round(cfg.batch_size / max(bs, 1)))
            log.info("R-CNN 训练: arch=%s 输入 min=%s max=%s 训练样本=%d 验证样本=%d "
                     "工业优化=%s AMP=%s 批次=%d 梯度累积=%d%s 梯度检查点=%s 数据落盘=无",
                     self.arch, size, max_size, len(train_set), len(val_set),
                     cfg.industrial_opt, use_amp, bs, accum,
                     f"（物理批次按显存预算由 {cfg.batch_size} 下调）" if bs < cfg.batch_size else "",
                     "开" if need_ckpt else "关")
            pin = self.device == "cuda"
            loader = DataLoader(train_set, batch_size=bs, shuffle=True,
                                num_workers=cfg.workers, collate_fn=_collate,
                                pin_memory=pin,
                                persistent_workers=(cfg.workers > 0))
            # 验证固定 batch=1：postprocess 会按原图尺寸给每条检出贴 mask，
            # 大图 × 批内多条检出同时驻留显存极易 OOM（详见 training.log 09-09 案例）
            vloader = DataLoader(val_set, batch_size=1,
                                 num_workers=cfg.workers, collate_fn=_collate,
                                 pin_memory=pin,
                                 persistent_workers=(cfg.workers > 0))

            params = [p for p in self.net.parameters() if p.requires_grad]
            # R-CNN 默认 AdamW lr=1e-4——旧默认 SGD lr=1e-3 在冻结 BN 的
            # 小数据 R-CNN 上易平台期（用户实测第 2 轮后损失几乎不动）。
            # 默认 AdamW(1e-4, wd 0.05)，rcnn_optimizer="sgd" 可回退旧行为
            if cfg.rcnn_optimizer == "adamw":
                base_lr = cfg.rcnn_learning_rate or 1e-4   # 0=自动
                opt = torch.optim.AdamW(params, lr=base_lr,
                                        weight_decay=cfg.rcnn_weight_decay)
            else:
                base_lr = cfg.learning_rate
                opt = torch.optim.SGD(params, lr=base_lr, momentum=0.9,
                                      weight_decay=cfg.weight_decay)
            log.info("R-CNN 优化器: %s（base_lr=%g，wd=%g）",
                     type(opt).__name__, base_lr,
                     opt.defaults.get("weight_decay", 0.0))
            # 默认"预热 1 轮 + 余弦退火至 5%"；XCCV_RCNN_LR_SCHEDULE=step 切换为
            # mmdet 式三段（预热 500 步 → 恒定 → 8/12、11/12 进度 ×0.1）——小数据
            # 实测持平（1600 入网）或因更新步少而欠训练（2560 入网），故不作默认
            steps_per_epoch = max(len(loader), 1)
            total_steps = max(cfg.epochs * steps_per_epoch, 1)
            if accum > 1:  # 累积时按有效更新步计（预热/里程碑同步缩放）
                steps_per_epoch = max(1, math.ceil(len(loader) / accum))
                total_steps = max(cfg.epochs * steps_per_epoch, 1)
            if cfg.rcnn_optimizer == "adamw" and \
                    os.environ.get("XCCV_RCNN_LR_SCHEDULE", "") == "step":
                warm_steps = max(1, min(500, total_steps // 10))
                m1 = round(cfg.epochs * 8 / 12) * steps_per_epoch
                m2 = round(cfg.epochs * 11 / 12) * steps_per_epoch

                def _lr_lambda(step: int) -> float:
                    if step < warm_steps:
                        return 0.001 + 0.999 * step / warm_steps
                    if step >= m2:
                        return 0.01
                    if step >= m1:
                        return 0.1
                    return 1.0
            else:
                warmup_steps = steps_per_epoch
                total_steps = max(cfg.epochs * steps_per_epoch, warmup_steps + 1)

                def _lr_lambda(step: int) -> float:
                    if step < warmup_steps:
                        return (step + 1) / warmup_steps
                    p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * p))

            sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

            best, best_state, history = -1.0, None, []
            stale, early_stopped = 0, False
            total_nan_steps = 0
            n_batches = max(len(loader), 1)
            for epoch in range(1, cfg.epochs + 1):
                if reporter.stop_requested:
                    break  # 界面请求停止：保存已有最优权重
                self.net.train()
                if self._tile_tsize:
                    self.net.transform.min_size = self._tile_tsize[0]
                    self.net.transform.max_size = self._tile_tsize[1]
                total, count = 0.0, 0
                nan_steps = 0
                comp_totals: dict = {}
                for bi, (imgs, targets) in enumerate(loader, start=1):
                    reporter.on_train_progress(epoch, bi, n_batches)  # 轮内进度上报
                    imgs = [im.to(self.device) for im in imgs]
                    tgts = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                    loss, loss_dict = _rcnn_train_step(self.net, imgs, tgts, opt,
                                                       scaler, use_amp,
                                                       accum=accum, accum_idx=bi)
                    if loss is None:
                        # 数值防护：非有限损失（NaN/Inf）跳过本步，不更新参数，
                        # 也不计入轮平均——否则整个曲线被污染且难以定位
                        nan_steps += 1
                        if nan_steps <= 3:
                            log.warning(
                                "第 %d 轮第 %d/%d 步损失非有限，本步已跳过。分量: %s",
                                epoch, bi, n_batches,
                                {k: round(v, 4) if math.isfinite(v) else v
                                 for k, v in loss_dict.items()})
                        continue
                    total += loss.item()
                    count += 1
                    if accum <= 1 or bi % accum == 0:
                        sched.step()  # 预热/退火按有效更新步推进（NaN 步不计）
                    for k, v in loss_dict.items():
                        comp_totals[k] = comp_totals.get(k, 0.0) + float(v)
                total_nan_steps += nan_steps
                if nan_steps:
                    log.warning("第 %d 轮共 %d/%d 步损失非有限（已跳过）",
                                epoch, nan_steps, n_batches)
                if _nan_abort(nan_steps, n_batches):
                    raise RuntimeError(
                        f"第 {epoch} 轮有 {nan_steps}/{n_batches} 步损失为 NaN/Inf，已中止训练"
                        "（具体分量值见日志）。请把 platform_data/logs/training.log 发回分析，"
                        "也可先尝试关闭「AMP 混合精度」再训练")
                # 组件损失逐项上报（分类/回归/RPN/掩码），供曲线可视化
                n_b = max(count, 1)
                logs = {"train_loss": round(total / n_b, 4)}
                for k, v in comp_totals.items():
                    logs[k.replace("loss_", "")] = round(v / n_b, 4)

                if epoch % cfg.eval_interval == 0 or epoch == cfg.epochs:
                    ev = self._eval_coco(vloader)
                    logs.update(ev)
                    # 选优口径：掩码模型按 mask mAP（内部参数）；对外展示恒为 bbox
                    sel = ev["mask_map50"] if self.with_masks else ev["map50"]
                    if sel >= best:
                        best = sel
                        best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                        stale = 0
                    else:
                        stale += 1
                    if cfg.patience and stale >= cfg.patience:
                        early_stopped = True
                history.append({"epoch": epoch, **logs})
                reporter.on_epoch(epoch, logs)
                if getattr(self, "epoch_store", None) is not None:
                    # 逐轮存档（内存序列化 → .pt）
                    self.epoch_store.save(epoch, {
                        "format": "xccv-rcnn", "arch": self.arch, "task": self.task.value,
                        "classes": data.classes, "min_size": size,
                        "train_input": applied_input,
                        "state_dict": {k: v.detach().cpu()
                                       for k, v in self.net.state_dict().items()},
                        "industrial_opt": cfg.industrial_opt,
                        "mask_head_light": cfg.mask_head_light,
                        "slim_head": cfg.slim_head, "epoch": epoch,
                        "anchor_base": anchor_used,
                    })
                if early_stopped:
                    break

            if best_state is not None:
                self.net.load_state_dict(best_state)
            # 保存最优权重 best.pt
            best_pt = cfg.output_dir / "best.pt"
            best_ckpt = {
                "format": "xccv-rcnn", "arch": self.arch, "task": self.task.value,
                "classes": data.classes, "min_size": size, "state_dict": self.net.state_dict(),
                "industrial_opt": cfg.industrial_opt, "mask_head_light": cfg.mask_head_light,
                "slim_head": cfg.slim_head,
                "anchor_base": anchor_used,
                # T10-3 联动：部署侧读 meta 决定推理口径（0=跟随画布）
                "train_input": applied_input,
            }
            if cfg.resume_from_pruned:
                best_ckpt["net_module"] = self.net  # 剪后网络结构随文件保存
            torch.save(best_ckpt, best_pt)

            final = self._eval_coco(vloader)
            if ckpt_patches:
                _restore_grad_checkpoint(ckpt_patches)  # 保存 net_module 前摘除闭包补丁
            if cfg.industrial_opt:
                _restore_rcnn_loss_patches()  # 恢复原生损失，避免影响后续基线训练
            log.info("R-CNN 训练完成: arch=%s 选优口径=%s 最优=%.4f 最终 bbox mAP@50=%.4f 掩码 mAP@50=%s "
                     "非有限步=%d", self.arch,
                     "mask" if self.with_masks else "bbox", best if best >= 0 else -1.0,
                     final["map50"], final.get("mask_map50", "-"), total_nan_steps)
            return {
                "best_val_map50": best if best >= 0 else final["map50"],
                "final": final,
                "industrial_opt": cfg.industrial_opt,
                "rcnn_optimizer": str(getattr(cfg, "rcnn_optimizer", "adamw")),
                "rcnn_learning_rate": float(
                    getattr(cfg, "rcnn_learning_rate", 0.0) or 0.0),
                "rcnn_weight_decay": float(
                    getattr(cfg, "rcnn_weight_decay", 0.05) or 0.0),
                "train_input": applied_input,
                "freeze_backbone": bool(cfg.freeze_backbone and cfg.use_pretrained),
                "mask_head_light": bool(cfg.mask_head_light
                                        and self.arch in ("maskrcnn",
                                                          "maskrcnn18")),
                "trainable_params": sum(p.numel() for p in self.net.parameters() if p.requires_grad),
                "early_stopped": early_stopped,
                "history": history,
                "confusion_matrix": self._confusion_matrix(data),
                "classes": data.classes,
            }
        finally:
            # P92 兜底：NaN 中止/任何异常路径也必须恢复进程全局损失补丁，
            # 否则残留会污染后续关闭工业优化的训练（正常路径已在收尾处恢复，幂等）
            if cfg.industrial_opt:
                _restore_rcnn_loss_patches()

    # ---- 评估 ----
    @torch.inference_mode()  # T9-6：评估期免版本图（更省内存）
    def _eval_coco(self, vloader) -> dict:
        """COCO mAP 评估（低置信度阈值，保证指标客观）。

        对外口径统一为 bbox：map50/map50_95 恒为边界框 mAP（进曲线与摘要）。
        掩码模型额外返回 mask_map50/mask_map50_95（mask 口径），仅作内部
        选优参数，不对外可视化。
        另产出宽松口径：map_loose（IoU {0.10,0.25,0.50} 三档平均）、
        map10/map25/mar_loose，工业缺陷框匹配更宽容。
        """
        self.net.eval()
        _ts = getattr(self, "_tile_tsize", None)
        if _ts:   # 切图训练：整图验证恢复画布尺寸
            self.net.transform.min_size, self.net.transform.max_size = _ts[2], _ts[3]
        self.net.roi_heads.score_thresh = 0.05
        # 评估口径对齐参考工具：低 NMS（默认 0.5 压掉密集小目标检出）
        self.net.roi_heads.box_nms_thresh = 0.2
        if self.device == "cuda":
            # 训练期缓存块先释放：验证期 postprocess 按原图尺寸贴 mask，
            # 需要大块连续显存（此前大图训练在验证阶段 OOM 的直接原因）
            torch.cuda.empty_cache()
        _install_cpu_mask_postprocess(self.net)
        try:
            dets = self._collect_detections(vloader)
        finally:
            _restore_mask_postprocess(self.net)
        if not dets:
            out = {"map50": 0.0, "map50_95": 0.0,
                   "map_loose": 0.0, "map10": 0.0, "map25": 0.0, "mar_loose": 0.0}
            if self.with_masks:
                out.update(mask_map50=0.0, mask_map50_95=0.0)
            return out
        coco_gt = vloader.dataset.coco
        coco_dt = coco_gt.loadRes(dets)
        # pycocotools 对"无有效评估组合"返回哨兵值 -1，
        # 平台展示口径统一归零，保证指标始终在 [0,1]
        def _sane(x: float) -> float:
            return round(max(0.0, float(x)), 4)

        iou_types = ["bbox"] + (["segm"] if self.with_masks else [])
        out: dict = {}
        for iou_type in iou_types:
            ev = COCOeval(coco_gt, coco_dt, iouType=iou_type)
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
            prefix = "" if iou_type == "bbox" else "mask_"
            out[f"{prefix}map50"] = _sane(ev.stats[1])
            out[f"{prefix}map50_95"] = _sane(ev.stats[0])
        # 宽松口径：IoU 阈值 {0.10, 0.25, 0.50} 三档 AP 平均（比 COCO 的
        # [0.50:0.95] 十档平均宽松得多，工业缺陷框匹配更宽容）；
        # COCO 标准口径照常并存。
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.params.iouThrs = np.array([0.10, 0.25, 0.50])
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        # stats 槽位固定对应 IoU .5/.75 摘要，自定义 iouThrs 下语义失效
        # （曾把 map10 读成 -1→0、mar_loose 读成 AP@0.5）——逐档直接取
        # precision/recall 数组（area=all、maxDets=100）
        if ev.eval is None:  # 无有效评估组合（如验证集无 GT）
            out.update(map_loose=0.0, map10=0.0, map25=0.0, mar_loose=0.0)
            return out
        prec = ev.eval["precision"]  # [T,R,K,A,M]
        rec = ev.eval["recall"]      # [T,K,A,M]

        def _avg(arr: np.ndarray) -> float:
            v = arr[arr > -1]
            return float(v.mean()) if v.size else 0.0

        aps = [_avg(prec[t, :, :, 0, -1]) for t in range(3)]
        ars = [_avg(rec[t, :, 0, -1]) for t in range(3)]
        out["map10"] = _sane(aps[0])
        out["map25"] = _sane(aps[1])
        out["map_loose"] = _sane(sum(aps) / 3.0)
        out["mar_loose"] = _sane(sum(ars) / 3.0)
        return out

    def _collect_detections(self, vloader) -> list:
        """跑一遍验证集，收集 COCO 格式检出（mask 已按原图尺寸贴好）。"""
        dets = []
        for imgs, targets in vloader:
            outs = self.net([im.to(self.device) for im in imgs])
            for t, out in zip(targets, outs):
                img_id = int(t["image_id"])
                for i, (box, label, score) in enumerate(zip(
                        out["boxes"].cpu().numpy(),
                        out["labels"].cpu().numpy(),
                        out["scores"].cpu().numpy())):
                    x1, y1, x2, y2 = box.tolist()
                    det = {"image_id": img_id, "category_id": int(label),
                           "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score)}
                    if self.with_masks and "masks" in out:
                        m = (out["masks"][i, 0] > 0.5).numpy().astype(np.uint8)
                        rle = maskUtils.encode(np.asfortranarray(m))
                        det["segmentation"] = {"size": list(rle["size"]),
                                               "counts": rle["counts"].decode("utf-8")}
                    dets.append(det)
        return dets

    def _confusion_matrix(self, data, conf: float = 0.25, iou: float = 0.5) -> list:
        """用最终权重在内存验证集上做 IoU 匹配统计。"""
        per_gt, per_pred = [], []
        for s in data.val_samples:
            img = imread_unicode(s.image_path)
            if img is None:
                continue
            gt_boxes, gt_labels = [], []
            for sh in s.shapes:
                bbox = _clipped_bbox(sh, s.width, s.height)
                ci = data.classes.index(sh.label) if sh.label in data.classes else -1
                if bbox is None or ci < 0:
                    continue
                gt_boxes.append(list(bbox))
                gt_labels.append(ci)
            pred = self.predict(img)
            per_gt.append({"boxes": gt_boxes, "labels": gt_labels})
            per_pred.append({"boxes": pred.boxes, "labels": pred.labels, "scores": pred.scores})
        return detection_confusion_matrix(per_gt, per_pred, len(data.classes)).tolist()

    # ---- 推理 ----
    @torch.no_grad()
    def predict(self, image_bgr: np.ndarray, conf: float = 0.25, iou: float = 0.45) -> Prediction:
        self.net.eval()
        # 注意属性名是 score_thresh / nms_thresh（box_score_thresh 只是构造参数名）
        self.net.roi_heads.score_thresh = conf
        self.net.roi_heads.nms_thresh = iou
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        _install_cpu_mask_postprocess(self.net)  # 大原图贴 mask 的显存峰值防护
        try:
            out = self.net([x.to(self.device)])[0]
        finally:
            _restore_mask_postprocess(self.net)
        boxes = out["boxes"].cpu().numpy().tolist()
        scores = out["scores"].cpu().numpy().tolist()
        labels = (out["labels"].cpu().numpy() - 1).astype(int).tolist()
        masks = None
        if self.with_masks and "masks" in out:
            masks = [(out["masks"][i, 0] > 0.5).numpy() for i in range(len(boxes))]
        return Prediction(labels=labels, scores=scores, boxes=boxes, masks=masks)

    # ---- 权重 ----
    def load_weights(self, path) -> None:
        # path 为权重文件路径或 ckpt 字典（训练回调直传）
        if isinstance(path, dict):
            ckpt = path
        else:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.arch = ckpt["arch"]
        self.task = TaskType(ckpt["task"])
        self.with_masks = self.task is TaskType.SEGMENTATION
        self.num_classes = len(ckpt["classes"])
        self.classes = ckpt["classes"]
        self.industrial = ckpt.get("industrial_opt", False)
        self.mask_head_light = ckpt.get("mask_head_light", False)  # 供引擎导出透传
        self._min_size = ckpt.get("min_size", self.default_image_size)
        if "net_module" in ckpt:  # 模型工具剪枝后的整网结构：直接取用
            self.net = ckpt["net_module"]
        else:
            self.net = self._build(self.arch, self.num_classes, self._min_size,
                                   use_pretrained=False, industrial=self.industrial,
                                   mask_head_light=ckpt.get("mask_head_light", False),
                                   slim_head=ckpt.get("slim_head", False),
                                   anchor_base=ckpt.get("anchor_base"))
            from .modules import load_state_dict_compat

            try:
                load_state_dict_compat(self.net, ckpt["state_dict"])
            except RuntimeError as e:
                raise RuntimeError(
                    "权重结构与模型配置不匹配（常见原因：slim 头/轻量掩码头开关"
                    "与训练时不一致）：" + str(e)) from e
        self.net.to(self.device).eval()
