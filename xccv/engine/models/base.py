"""模型抽象基类、训练配置与训练回调协议。"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from ..dataset.base import TaskType
from ..dataset.prepare import PreparedData, prepare
from ..io_utils import save_json


def resolve_device(device: str = "auto") -> str:
    """auto：有 NVIDIA GPU 用 GPU，否则 CPU。"""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


@dataclass
class TrainConfig:
    """一次训练的全部设置（训练平台各表单的落点）。"""

    task: TaskType
    dataset_dir: Path
    output_dir: Path            # 本次训练输出目录（权重、指标、导出数据）
    model_name: str
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_ratio: float = 0.2      # 验证集比例
    train_ratio: float = 0.0    # 训练集占比（0=自动 1-val_ratio；与 val_ratio 之和>1 时允许重叠）
    eval_interval: int = 1      # 每多少轮验证一次
    multiply_factor: int = 1    # 每轮重复学习次数（epoch 内训练集重复遍数）
    augment_ops: dict = field(default_factory=dict)  # {"random_brightness": {"max_delta": 30}, ...}
    augment_probs: dict = field(default_factory=dict)  # 每算子启用概率 {"random_flip_h": 0.5}；缺失按 0.5
    image_size: int = 0         # 0 表示用模型默认值（方形）；被宽高指定覆盖
    image_width: int = 0        # 训练输入宽（>0 且高>0 时生效，与增强 resize 统一）
    image_height: int = 0       # 训练输入高
    train_input: int = 0        # T9-2 两段式缩放（R-CNN 家族 opt-in）：0=跟随
    # 画布（默认现状）；显式设 32 的倍数且<画布短边时，画布增强保持画布口径，
    # 入网前由 GeneralizedRCNNTransform keep_ratio 缩到该尺寸
    device: str = "auto"
    seed: int = 42
    use_pretrained: bool = True
    workers: int = 4
    industrial_opt: bool = True  # 工业场景优化（注意力 + 改进损失，关闭即为基线对照）
    patience: int = 10           # 早停耐心：连续 N 次验证无提升则停止（0 关闭）
    # ---- 训练加速（默认全关，GPU 工控机上按需开启）----
    freeze_backbone: bool = False  # 冻结预训练骨干，仅训练头部/工业模块（快速迁移）
    rpn_reduced: bool = False      # R-CNN 系列：削减 RPN 候选框（2000→256），其他家族忽略
    use_amp: bool = True           # AMP 混合精度（仅 GPU 生效，CPU 自动关闭）
    mask_head_light: bool = False  # Mask R-CNN：轻量掩码头（4 层 256 → 2 层 128），其他家族忽略
    slim_head: bool = False        # T9-5 slim 头（仅 maskrcnn18）：FPN64+两层256FC+4×64 掩码
    anchor_small: bool = False     # 小目标锚框自适应（R-CNN 家族 opt-in）：底径=入网短边/80
    # ---- R-CNN 优化器（默认 AdamW 1e-4，可回退 SGD）----
    rcnn_optimizer: str = "adamw"    # R-CNN 家族优化器（adamw|sgd），其他家族忽略
    rcnn_learning_rate: float = 0.0  # 0=自动（AdamW→1e-4；sgd→learning_rate 字段）
    rcnn_weight_decay: float = 0.05  # AdamW 权重衰减（sgd 走 weight_decay 字段）
    tf32: bool = True              # GPU 提速：TF32 + cuDNN 自动调优（Ampere+ 显卡，CPU 无影响）
    save_epoch_weights: bool = True  # 每轮保存权重到共用 epoch_weights 文件夹（覆盖式）
    resume_from_pruned: bool = False  # 从项目自有的剪枝预训练模型开始训练（不重建网络）
    # ---- T9-3 切图训练（opt-in，R-CNN 家族；验证集始终整图评估）----
    tiling: bool = False          # 启用切图训练（需配合推理侧分块 tiled_infer）
    tile_size: int = 640          # tile 边长（像素）
    tile_overlap: float = 0.2     # 相邻 tile 重叠比例
    bg_tile_keep: float = 0.15    # 无目标 tile 的保留率（负样本防刷屏）
    min_visibility: float = 0.3   # 裁剪后可见面积比低于此值的目标丢弃
    mix_full_view: float = 0.1    # 整图混入比例（防"只见局部"分布偏移）
    # ---- 负样本（检测/分割全家族）----
    use_negatives: bool = True    # 负样本文件夹（OK/良品/负样本/背景…）内未标注图作背景训练
    # ---- YOLO 家族 ----
    p2_head: bool = False         # P2 小目标检测头（stride 4 第四检测层；其他家族忽略）
    # ---- O5（V0.7）颜色模式 ----
    color_mode: str = "color"     # "color"=彩色（现状）；"gray"=训练读图转灰再复制 3 通道
    # ---- E3（V0.7）确定性训练 ----
    deterministic: bool = False   # opt-in：种子全集+确定性算法开关（同机同种子结果可复现）


class TrainReporter:
    """训练进度回调。界面层继承它，把消息转发到 UI 线程。

    stop_requested 置 True 后，支持中断的模型（分类/R-CNN 家族）会在
    当前轮结束后停止训练并保存已有最优权重。
    """

    stop_requested: bool = False

    def on_epoch(self, epoch: int, logs: dict) -> None:
        """每完成一次验证触发，logs 如 {"train_loss": .., "val_top1": ..}。"""

    def on_train_progress(self, epoch: int, batch_index: int, batch_total: int) -> None:
        """每个批次训练后触发（轮内进度，用于总体进度条的实时百分比）。"""

    def on_finish(self, success: bool, message: str = "") -> None:
        """训练结束（无论成败）触发。"""


@dataclass
class Prediction:
    """单张图预测结果。

    分类任务填 labels/scores/probs；检测任务填 boxes/scores/labels；
    分割任务再填 masks（与 boxes 一一对应的布尔掩码）。
    OCR 任务（A2，V0.7）填 boxes/scores + texts（与 boxes 一一对应的
    识别文本；additive 契约，统一 JSON detections 同名透出）。
    """

    labels: list[int] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    boxes: list[list[float]] = field(default_factory=list)  # xyxy 像素坐标
    masks: list | None = None           # list[np.ndarray(H, W) bool]
    probs: list[float] | None = None    # 分类任务：全部类别概率
    texts: list[str] | None = None      # OCR 任务：逐框识别文本


def _with_run_meta(metrics: dict, cfg: TrainConfig, started: datetime,
                   ended: datetime, status: str, error: str = "") -> None:
    """把时间、状态、进度与参数快照写入指标字典（训练历史记录的数据源）。"""
    metrics["status"] = status
    metrics["start_time"] = started.strftime("%Y-%m-%d %H:%M:%S")
    metrics["end_time"] = ended.strftime("%Y-%m-%d %H:%M:%S")
    metrics["duration_sec"] = round((ended - started).total_seconds())
    metrics["epochs_planned"] = cfg.epochs
    if cfg.save_epoch_weights:
        metrics["epoch_weights_dir"] = str(Path(cfg.output_dir).parent / "epoch_weights")
    try:  # 数据集指纹（对比视图提示"数据集已修改"用）；失败不阻塞训练收尾
        from ..dataset.prepare import dataset_fingerprint

        metrics["dataset_fingerprint"] = dataset_fingerprint(cfg.dataset_dir)
    except Exception:
        pass
    metrics["config"] = {
        "model_name": cfg.model_name, "epochs": cfg.epochs,
        "batch_size": cfg.batch_size, "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay, "val_ratio": cfg.val_ratio,
        "train_ratio": cfg.train_ratio,
        "eval_interval": cfg.eval_interval, "multiply_factor": cfg.multiply_factor,
        "augment_ops": cfg.augment_ops, "augment_probs": cfg.augment_probs,
        "image_size": cfg.image_size,
        "image_width": cfg.image_width, "image_height": cfg.image_height,
        "device": cfg.device, "seed": cfg.seed, "industrial_opt": cfg.industrial_opt,
        "patience": cfg.patience, "use_pretrained": cfg.use_pretrained,
        "freeze_backbone": cfg.freeze_backbone, "rpn_reduced": cfg.rpn_reduced,
        "use_amp": cfg.use_amp, "mask_head_light": cfg.mask_head_light,
        "slim_head": cfg.slim_head,
        "anchor_small": getattr(cfg, "anchor_small", False),
        "rcnn_optimizer": cfg.rcnn_optimizer,
        "rcnn_learning_rate": cfg.rcnn_learning_rate,
        "rcnn_weight_decay": cfg.rcnn_weight_decay,
        "tf32": cfg.tf32, "save_epoch_weights": cfg.save_epoch_weights,
        "use_negatives": bool(getattr(cfg, "use_negatives", True)),
        "p2_head": bool(getattr(cfg, "p2_head", False)),
        "color_mode": str(getattr(cfg, "color_mode", "color")),
        "deterministic": bool(getattr(cfg, "deterministic", False)),
    }
    if error:
        metrics["error"] = error
    metrics.setdefault("history", [])


def negative_fp_stats(predict_fn, val_samples, conf: float = 0.25) -> dict:
    """验证集负样本（无标注的 OK 图）上的误检统计；无负样本时返回 {}。

    neg_fp_per_img=平均每张 OK 图的误检框数；neg_fp_img_rate=出现误检的
    OK 图占比（过检率）。predict_fn(image_bgr) -> Prediction，conf 为计数阈值。
    """
    from ..io_utils import imread_unicode

    n_box = n_hit = seen = 0
    for s in val_samples:
        if s.shapes:
            continue
        img = imread_unicode(s.image_path)
        if img is None:
            continue
        pred = predict_fn(img)
        k = sum(1 for sc in (pred.scores or []) if float(sc) >= conf)
        n_box += k
        n_hit += 1 if k else 0
        seen += 1
    if not seen:
        return {}
    return {"neg_fp_per_img": round(n_box / seen, 3),
            "neg_fp_img_rate": round(n_hit / seen, 4),
            "neg_val_images": seen}


class BaseModel(ABC):
    """所有模型的统一接口。子类声明 family（数据家族）与 supported_tasks。"""

    name: str = ""
    family: str = ""                 # "yolo" | "coco" | "folder"
    supported_tasks: tuple = ()
    default_image_size: int = 640

    def __init__(self, num_classes: int, device: str = "auto"):
        self.num_classes = num_classes
        self.device = resolve_device(device)
        self.classes: list[str] | None = None

    # ---- 对界面暴露的主流程 ----
    @staticmethod
    def _apply_gpu_tuning(cfg: TrainConfig) -> None:
        """GPU 提速（仅 CUDA 环境生效，CPU 无影响）：

        - TF32：Ampere+ 显卡上 FP32 矩阵乘/卷积零成本加速；
        - cudnn.benchmark：固定输入尺寸训练时自动搜索最优卷积算法并缓存。
        """
        if not cfg.tf32 or not torch.cuda.is_available():
            return
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    def train(self, cfg: TrainConfig, reporter: TrainReporter | None = None) -> dict:
        """模板方法：数据准备 -> 训练 -> 保存指标。界面只调这一个。"""
        reporter = reporter or TrainReporter()
        started = datetime.now()
        self._apply_gpu_tuning(cfg)
        try:
            cfg.output_dir.mkdir(parents=True, exist_ok=True)
            data = prepare(
                task=cfg.task,
                family=self.family,
                dataset_dir=cfg.dataset_dir,
                val_ratio=cfg.val_ratio,
                train_ratio=cfg.train_ratio or None,
                seed=cfg.seed,
                multiply_factor=cfg.multiply_factor,
                augment_ops=cfg.augment_ops,
                augment_probs=cfg.augment_probs,
                use_negatives=bool(getattr(cfg, "use_negatives", True)),
            )
            # 轮次权重存档：共用文件夹（output_dir 的上级 = 模型储存路径），
            # 数据准备成功后清空旧训练内容，本次训练逐轮覆盖写入
            self.epoch_store = None
            if cfg.save_epoch_weights:
                from ..epoch_weights import EpochWeightStore

                self.epoch_store = EpochWeightStore(Path(cfg.output_dir).parent)
                self.epoch_store.reset()
                logging.getLogger(__name__).info(
                    "轮次权重存档: %s（每轮保存，新训练自动覆盖）", self.epoch_store.dir)
            torch.manual_seed(cfg.seed)
            np.random.seed(cfg.seed)
            import random as _random

            _random.seed(cfg.seed)
            if getattr(cfg, "deterministic", False):
                # E3（V0.7）opt-in 确定性训练：种子全集 + 确定性算法开关。
                # warn_only=True：个别算子不支持确定性实现时降级告警而非中断
                # （VIMO FAQ"个别算子不支持"口径）；性能有损失，默认关。
                import os as _os

                _os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
                torch.use_deterministic_algorithms(True, warn_only=True)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(cfg.seed)
            metrics = self.fit(data, cfg, reporter)
            _with_run_meta(metrics, cfg, started, datetime.now(),
                           status="interrupted" if reporter.stop_requested else "completed")
            save_json(cfg.output_dir / "metrics.json", metrics)
            # E2（V0.7）：run_manifest.json（环境/数据指纹 + 完整参数快照）
            from .manifest import write_run_manifest

            write_run_manifest(cfg, metrics, cfg.output_dir)
            reporter.on_finish(True)
            return metrics
        except Exception as e:  # 训练失败要把原因带回界面，同时留档到历史记录
            try:
                failed: dict = {}
                _with_run_meta(failed, cfg, started, datetime.now(),
                               status="failed", error=f"{type(e).__name__}: {e}")
                save_json(cfg.output_dir / "metrics.json", failed)
            except Exception:
                pass
            reporter.on_finish(False, f"{type(e).__name__}: {e}")
            raise

    @abstractmethod
    def fit(self, data: PreparedData, cfg: TrainConfig, reporter: TrainReporter) -> dict:
        """在已导出的数据上训练，返回最终指标字典（含混淆矩阵与 history）。"""

    @abstractmethod
    def predict(self, image_bgr: np.ndarray, conf: float = 0.25, iou: float = 0.45) -> Prediction:
        """对单张 BGR 图推理。"""

    @abstractmethod
    def load_weights(self, path: Path) -> None:
        """加载本模型保存的权重（测试平台用），同时恢复 classes。"""
