"""模型注册表：平台可选模型的名称、任务与创建/加载入口。"""

import logging
import time
from pathlib import Path

import torch

from ..dataset.base import TaskType
from .base import BaseModel
from .classifier import ClassifierModel
from .rcnn import RCNNModel
from .yolo import YoloModel

log = logging.getLogger(__name__)

# name -> (支持的任务, 中文说明, 创建函数(task, num_classes, device, pretrained))
MODEL_SPECS: dict[str, tuple] = {
    "resnet18": (TaskType.CLASSIFICATION, "ResNet18 分类",
                 lambda task, n, dev, pre: ClassifierModel("resnet18", n, dev, pre)),
    "resnet50": (TaskType.CLASSIFICATION, "ResNet50 分类",
                 lambda task, n, dev, pre: ClassifierModel("resnet50", n, dev, pre)),
    "mobilenet_v3_small": (TaskType.CLASSIFICATION, "MobileNetV3-Small 分类（轻量）",
                           lambda task, n, dev, pre: ClassifierModel("mobilenet_v3_small", n, dev, pre)),
    "yolo11n": (TaskType.DETECTION, "YOLO11n 检测（轻量）",
                lambda task, n, dev, pre: YoloModel("yolo11n", task, n, dev, pre)),
    "yolo11s": (TaskType.DETECTION, "YOLO11s 检测",
                lambda task, n, dev, pre: YoloModel("yolo11s", task, n, dev, pre)),
    "yolo11n-seg": (TaskType.SEGMENTATION, "YOLO11n 分割（轻量）",
                    lambda task, n, dev, pre: YoloModel("yolo11n", TaskType.SEGMENTATION, n, dev, pre)),
    "yolo11s-seg": (TaskType.SEGMENTATION, "YOLO11s 分割",
                    lambda task, n, dev, pre: YoloModel("yolo11s", TaskType.SEGMENTATION, n, dev, pre)),
    "fasterrcnn": (TaskType.DETECTION, "Faster R-CNN 检测",
                   lambda task, n, dev, pre: RCNNModel("fasterrcnn", TaskType.DETECTION, n, dev, pre)),
    "maskrcnn": (TaskType.SEGMENTATION, "Mask R-CNN 实例分割",
                 lambda task, n, dev, pre: RCNNModel("maskrcnn", TaskType.SEGMENTATION, n, dev, pre)),
    "maskrcnn18": (TaskType.SEGMENTATION, "Mask R-CNN 实例分割（ResNet18 轻量）",
                   lambda task, n, dev, pre: RCNNModel("maskrcnn18", TaskType.SEGMENTATION, n, dev, pre)),
}


def list_models(task: TaskType | str | None = None) -> list[dict]:
    """列出可选模型；task 给定时只列出支持该任务的。"""
    if task is not None:
        task = TaskType(task)
    return [
        {"name": name, "task": info[0].value, "desc": info[1]}
        for name, info in MODEL_SPECS.items()
        if task is None or info[0] is task
    ]


def create_model(name: str, task: TaskType | str, num_classes: int,
                 device: str = "auto", use_pretrained: bool = True) -> BaseModel:
    task = TaskType(task)
    if name not in MODEL_SPECS:
        raise ValueError(f"未知模型: {name}，可选: {', '.join(MODEL_SPECS)}")
    supported, _, factory = MODEL_SPECS[name]
    if task not in (supported,):
        raise ValueError(f"模型 {name} 不支持任务 {task.label}（仅支持 {supported.label}）")
    return factory(task, num_classes, device, use_pretrained)


def load_model(weights_path: Path, device: str = "auto") -> BaseModel:
    """按权重文件自动识别模型类型并加载（测试平台 / 批量预标注入口）。

    平台自有格式（xccv-classifier / xccv-rcnn）直接重建；
    否则按 ultralytics YOLO 权重处理。
    """
    path = Path(weights_path)
    t0 = time.perf_counter()
    log.info("加载权重: %s (%.1f MB)", path, path.stat().st_size / 1e6)

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        log.warning("权重文件 torch.load 失败，按 ultralytics YOLO 权重重试", exc_info=True)
        ckpt = None
    if isinstance(ckpt, dict) and ckpt.get("format") == "xccv-classifier":
        log.info("格式: xccv-classifier (arch=%s, %s 类, 加载耗时 %.1fs)",
                 ckpt.get("arch"), len(ckpt.get("classes") or []), time.perf_counter() - t0)
        model = ClassifierModel("resnet18", 2, device, use_pretrained=False)  # 占位，load_weights 会重建
        model.load_weights(ckpt)
        return model
    if isinstance(ckpt, dict) and ckpt.get("format") == "xccv-rcnn":
        log.info("格式: xccv-rcnn (arch=%s, task=%s, 加载耗时 %.1fs)",
                 ckpt.get("arch"), ckpt.get("task"), time.perf_counter() - t0)
        model = RCNNModel("fasterrcnn", TaskType.DETECTION, 1, device, use_pretrained=False)
        model.load_weights(ckpt)
        return model
    log.info("格式: ultralytics YOLO (加载耗时 %.1fs)", time.perf_counter() - t0)
    model = YoloModel("yolo11n", TaskType.DETECTION, 1, device, use_pretrained=False)
    model.load_weights(path)
    return model
