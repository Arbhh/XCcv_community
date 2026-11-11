"""分类模型封装：ResNet18 / ResNet50 / MobileNetV3-Small（torchvision）。"""

import logging
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision import transforms as T

from ..dataset.base import TaskType
from ..dataset.transforms import maybe_apply_ops
from ..io_utils import imread_unicode
from ..metrics.classification import confusion_matrix, topk_accuracy
from .base import BaseModel, Prediction, TrainConfig, TrainReporter
from .modules import CBAM, ECA, FocalLoss

log = logging.getLogger(__name__)


def _replace_se_with_eca(module: nn.Module) -> int:
    """递归将 MobileNetV3 中的 SqueezeExcitation 模块替换为 ECA（code.md）。"""
    from torchvision.ops import SqueezeExcitation

    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, SqueezeExcitation):
            setattr(module, name, ECA(child.fc1.in_channels))
            count += 1
        else:
            count += _replace_se_with_eca(child)
    return count

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class _FolderDataset(Dataset):
    """直接读取原始分类数据集（不再导出副本到训练输出目录）。

    训练侧按 augment_ops 在线增强（每次取样随机生成，与数据倍增共用同一批
    cv2 算子，行为与旧的"先烘焙副本"一致）；验证侧只做尺寸归一化。
    """

    def __init__(self, samples, classes: list[str], input_size,
                 augment_ops: dict | None, train: bool,
                 augment_probs: dict | None = None):
        self.samples = samples
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.targets = [self.class_to_idx[s.label] for s in samples]
        hw = (input_size, input_size) if isinstance(input_size, int) else tuple(input_size)
        self._hw = (int(hw[0]), int(hw[1]))  # (高, 宽)
        self._augment_ops = dict(augment_ops) if (train and augment_ops) else None
        self._augment_probs = dict(augment_probs or {})
        self._mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        s = self.samples[index]
        img = imread_unicode(s.image_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {s.image_path}")
        if self._augment_ops:  # 按每算子启用概率随机增强，其余取样保持原图
            img, _ = maybe_apply_ops(img, [], self._augment_ops, probs=self._augment_probs)
        img = cv2.resize(img, (self._hw[1], self._hw[0]), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
        x = (x - self._mean) / self._std
        return x, self.targets[index]


class ClassifierModel(BaseModel):
    family = "folder"
    supported_tasks = (TaskType.CLASSIFICATION,)
    default_image_size = 224

    def __init__(self, arch: str, num_classes: int, device: str = "auto",
                 use_pretrained: bool = True, industrial: bool = False):
        super().__init__(num_classes, device)
        self.arch = arch
        self.task = TaskType.CLASSIFICATION
        self.industrial = industrial
        self._frozen = False  # fit 中按 cfg.freeze_backbone 重建（保证"先冻结后插入"顺序）
        self.net = self._build(arch, num_classes, use_pretrained, industrial).to(self.device)
        self._input_size = self.default_image_size  # int 或 (高, 宽)

    @staticmethod
    def _build(arch: str, num_classes: int, use_pretrained: bool,
               industrial: bool = False, freeze_backbone: bool = False) -> nn.Module:
        """构造网络。

        冻结顺序是关键：先冻结原始骨干参数，再插入工业模块（CBAM/ECA）并
        重建分类头——新建模块 requires_grad 默认 True，注意力与分类头天然可训练。
        """
        weights = "DEFAULT" if use_pretrained else None
        channels = None
        if arch == "resnet18":
            net = models.resnet18(weights=weights)
            channels = {"layer3": 256, "layer4": 512}
        elif arch == "resnet50":
            net = models.resnet50(weights=weights)
            channels = {"layer3": 1024, "layer4": 2048}
        elif arch == "mobilenet_v3_small":
            net = models.mobilenet_v3_small(weights=weights)
        else:
            raise ValueError(f"未知分类模型: {arch}")
        if freeze_backbone and use_pretrained:
            for p in net.parameters():
                p.requires_grad_(False)
        else:  # 无预训练权重时冻结没有意义，自动忽略
            freeze_backbone = False
        if arch == "resnet18":
            net.fc = nn.Linear(net.fc.in_features, num_classes)
        elif arch == "resnet50":
            net.fc = nn.Linear(net.fc.in_features, num_classes)
        else:  # mobilenet_v3_small
            net.classifier[3] = nn.Linear(net.classifier[3].in_features, num_classes)
        if industrial:
            if channels:  # ResNet 系：layer3/layer4 输出后插 CBAM（code.md）
                for name, ch in channels.items():
                    setattr(net, name, nn.Sequential(getattr(net, name), CBAM(ch)))
            else:  # MobileNetV3：SE → ECA（code.md）
                _replace_se_with_eca(net.features)
        net._xccv_frozen = freeze_backbone
        return net

    # ---- 工具 ----
    def _resolve_input(self, cfg: TrainConfig):
        """训练输入尺寸：宽高都指定时用 (高, 宽)（与增强 resize 统一的归一化尺寸），
        否则回退方形 image_size / 模型默认值。"""
        if cfg.image_width > 0 and cfg.image_height > 0:
            return (cfg.image_height, cfg.image_width)
        return cfg.image_size or self.default_image_size

    def _transform(self, size) -> T.Compose:
        """ImageFolder 产出的已是 PIL 图，这里只做缩放、转张量与归一化。

        size 为 int 时方形拉伸，元组为 (高, 宽) 拉伸（与增强 resize 的归一化口径一致）。
        """
        hw = (size, size) if isinstance(size, int) else tuple(size)
        return T.Compose([
            T.Resize(hw),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    @staticmethod
    def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ---- 训练 ----
    def fit(self, data, cfg: TrainConfig, reporter: TrainReporter) -> dict:
        input_size = self._resolve_input(cfg)
        self._input_size = input_size
        log.info("分类训练: arch=%s 输入=%s 训练样本=%d 验证样本=%d 倍增=%s",
                 self.arch, input_size, len(data.train_samples),
                 len(data.val_samples), cfg.multiply_factor)
        resume = cfg.resume_from_pruned and getattr(self, "resume_weights", False)
        if resume:
            # 剪枝预训练模型：保留剪后网络结构直接训练（迁移学习）
            log.info("从剪枝预训练模型继续训练（不重建网络）")
            if cfg.freeze_backbone:  # 冻结骨干、放开分类头（在剪后网络上生效）
                for p in self.net.parameters():
                    p.requires_grad_(False)
                head = getattr(self.net, "fc", None)
                head = head if head is not None else self.net.classifier
                for p in head.parameters():
                    p.requires_grad_(True)
        elif cfg.industrial_opt != self.industrial or cfg.freeze_backbone != self._frozen:
            # 工业优化 / 冻结骨干开关变化：按需重建网络（保持"先冻结后插入"顺序）
            self.net = self._build(self.arch, self.num_classes, cfg.use_pretrained,
                                   cfg.industrial_opt, cfg.freeze_backbone).to(self.device)
            self.industrial = cfg.industrial_opt
            self._frozen = cfg.freeze_backbone
        # 直接读取原始数据集（训练侧在线增强），不再把数据集复制到输出目录
        train_set = _FolderDataset(data.train_samples, data.classes, input_size,
                                   cfg.augment_ops, train=True,
                                   augment_probs=cfg.augment_probs)
        val_set = _FolderDataset(data.val_samples, data.classes, input_size,
                                 None, train=False)

        loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                            num_workers=cfg.workers, pin_memory=self.device == "cuda")
        vloader = DataLoader(val_set, batch_size=cfg.batch_size,
                             num_workers=cfg.workers, pin_memory=self.device == "cuda")
        opt = torch.optim.AdamW(
            [p for p in self.net.parameters() if p.requires_grad],
            lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        # 预热半轮 + 余弦退火至 5%：小数据集上恒定学习率末期易震荡，
        # 退火收尾通常能再抬升一些最终精度
        steps_per_epoch = max(len(loader), 1)
        warmup_steps = max(1, steps_per_epoch // 2)
        total_steps = max(cfg.epochs * steps_per_epoch, warmup_steps + 1)

        def _lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * p))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)

        # 工业优化损失：类别不均衡(≥5 倍)用 Focal(γ=2, 类平衡权重)，否则 Label Smoothing 0.1
        class_counts = torch.bincount(torch.tensor(train_set.targets),
                                      minlength=self.num_classes)
        imbalance = (class_counts.max().float() /
                     class_counts.min().float().clamp(min=1)).item()
        if cfg.industrial_opt and imbalance >= 5:
            weights = (class_counts.sum() / class_counts.float()).to(self.device)
            weights = weights / weights.mean()
            loss_fn = FocalLoss(gamma=2.0, class_weights=weights)
            loss_type = f"focal(gamma=2, class-balanced, imbalance={imbalance:.1f})"
        elif cfg.industrial_opt:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
            loss_type = "label_smoothing(0.1)"
        else:
            loss_fn = nn.CrossEntropyLoss()
            loss_type = "cross_entropy"

        # 混合精度：GPU 上启用（缩短训练时间），CPU 自动关闭（界面可控）
        use_amp = cfg.use_amp and self.device == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        best_acc, best_state, history = -1.0, None, []
        stale, early_stopped = 0, False
        n_batches = max(len(loader), 1)
        for epoch in range(1, cfg.epochs + 1):
            if reporter.stop_requested:
                break  # 界面请求停止：保存已有最优权重
            self.net.train()
            total, count = 0.0, 0
            nan_steps = 0
            for bi, (xb, yb) in enumerate(loader, start=1):
                reporter.on_train_progress(epoch, bi, n_batches)  # 轮内进度上报
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss = loss_fn(self.net(xb), yb)
                if not torch.isfinite(loss):
                    # 数值防护：跳过非有限步（不计入平均），持续出现则中止
                    nan_steps += 1
                    if nan_steps <= 3:
                        log.warning("第 %d 轮第 %d/%d 步损失非有限，本步已跳过",
                                    epoch, bi, n_batches)
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.net.parameters() if p.requires_grad], 10.0)
                scaler.step(opt)
                scaler.update()
                sched.step()  # 预热/余弦按有效更新步推进（跳过的 NaN 步不计）
                total += loss.item() * len(yb)
                count += len(yb)
            if nan_steps:
                log.warning("第 %d 轮共 %d/%d 步损失非有限（已跳过）",
                            epoch, nan_steps, n_batches)
                if nan_steps >= max(5, 0.3 * n_batches):
                    raise RuntimeError(
                        f"第 {epoch} 轮有 {nan_steps}/{n_batches} 步损失为 NaN/Inf，已中止训练；"
                        "请把 platform_data/logs/training.log 发回分析")
            logs = {"train_loss": round(total / max(count, 1), 4)}

            if epoch % cfg.eval_interval == 0 or epoch == cfg.epochs:
                top1, _, vloss = self._evaluate(vloader, loss_fn)
                logs["val_top1"] = round(top1, 4)
                logs["val_loss"] = round(vloss, 4)
                if top1 >= best_acc:  # >= 保证极小数据集也能产出权重
                    best_acc = top1
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
                    "format": "xccv-classifier", "arch": self.arch,
                    "classes": data.classes, "image_size": self._input_size,
                    "state_dict": {k: v.detach().cpu()
                                   for k, v in self.net.state_dict().items()},
                    "industrial_opt": cfg.industrial_opt, "loss_type": loss_type,
                    "epoch": epoch,
                })
            if early_stopped:
                break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        # 保存最优权重 best.pt
        best_pt = cfg.output_dir / "best.pt"
        best_ckpt = {
            "format": "xccv-classifier", "arch": self.arch, "classes": data.classes,
            "image_size": self._input_size, "state_dict": self.net.state_dict(),
            "industrial_opt": cfg.industrial_opt, "loss_type": loss_type,
        }
        if cfg.resume_from_pruned:
            best_ckpt["net_module"] = self.net  # 剪后网络结构随文件保存
        torch.save(best_ckpt, best_pt)

        top1, top5, vloss, cm, n_eval = self._evaluate(vloader, loss_fn, with_cm=True)
        log.info("分类训练完成: arch=%s 最优 Top-1=%.4f 最终 Top-1=%.4f",
                 self.arch, best_acc, top1)
        return {
            "best_val_top1": round(best_acc, 4),
            "final_val_top1": round(top1, 4),
            "final_val_top5": round(top5, 4),
            "loss_type": loss_type,
            "industrial_opt": cfg.industrial_opt,
            "freeze_backbone": cfg.freeze_backbone and bool(getattr(self.net, "_xccv_frozen", False)),
            "trainable_params": sum(p.numel() for p in self.net.parameters() if p.requires_grad),
            "early_stopped": early_stopped,
            "class_imbalance": round(imbalance, 2),
            "history": history,
            "confusion_matrix": cm.tolist(),
            "classes": data.classes,
        }

    @torch.no_grad()
    def _evaluate(self, vloader, loss_fn, with_cm: bool = False):
        """返回 (top1, top5, val_loss[, 混淆矩阵, 评估样本数])。"""
        self.net.eval()
        all_probs, all_targets = [], []
        total, count = 0.0, 0
        for xb, yb in vloader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            logits = self.net(xb)
            total += loss_fn(logits, yb).item() * len(yb)
            count += len(yb)
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            all_targets.append(yb.cpu().numpy())
        probs = np.concatenate(all_probs)
        targets = np.concatenate(all_targets)
        top1 = topk_accuracy(probs, targets, k=1)
        top5 = topk_accuracy(probs, targets, k=5)
        if not with_cm:
            return top1, top5, total / max(count, 1)
        cm = confusion_matrix(targets, probs.argmax(axis=1), self.num_classes)
        return top1, top5, total / max(count, 1), cm, len(targets)

    # ---- 推理 ----
    @torch.no_grad()
    def predict(self, image_bgr: np.ndarray, conf: float = 0.0, iou: float = 0.0) -> Prediction:
        self.net.eval()
        rgb = self._bgr_to_rgb(image_bgr)
        pil = Image.fromarray(rgb)
        x = self._transform(self._input_size)(pil).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.net(x)[0], dim=0).cpu().numpy()
        top = int(np.argmax(probs))
        return Prediction(labels=[top], scores=[float(probs[top])],
                          probs=[float(p) for p in probs])

    # ---- 权重 ----
    def load_weights(self, path) -> None:
        # path 为权重文件路径或 ckpt 字典（训练回调直传）
        if isinstance(path, dict):
            ckpt = path
        else:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.arch = ckpt["arch"]
        self.num_classes = len(ckpt["classes"])
        self.classes = ckpt["classes"]
        self.industrial = ckpt.get("industrial_opt", False)
        restored = ckpt.get("image_size", self.default_image_size)
        if isinstance(restored, (list, tuple)):  # 非方形输入 (高, 宽)
            restored = (int(restored[0]), int(restored[1]))
        self._input_size = restored
        if "net_module" in ckpt:  # 模型工具剪枝后的整网结构：直接取用
            self.net = ckpt["net_module"]
        else:
            self.net = self._build(self.arch, self.num_classes, use_pretrained=False,
                                   industrial=self.industrial)
            from .modules import load_state_dict_compat

            load_state_dict_compat(self.net, ckpt["state_dict"])
        self.net.to(self.device).eval()
