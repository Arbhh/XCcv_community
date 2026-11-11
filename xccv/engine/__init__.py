"""算法引擎对外接口。"""

from .offline import setup_offline_weights

setup_offline_weights()  # 优先启用随包基准权重（完全离线运行）

from .dataset.base import TaskType
from .dataset import AUGMENT_OPS, Sample, Shape

# torch 延迟导入（T1-4）：models 子包经 registry→rcnn/yolo 牵引 torch，而引擎包
# 被全部平台窗口在模块级导入——包本身必须保持轻量，模型 API 走 PEP 562 惰性转发
#（from ..engine import load_model 与 xccv.engine.load_model 两种写法都照常工作）。
_LAZY_MODELS = ("Prediction", "TrainConfig", "TrainReporter",
                "create_model", "list_models", "load_model")

__all__ = [
    "TaskType", "Sample", "Shape", "AUGMENT_OPS",
    "Prediction", "TrainConfig", "TrainReporter",
    "create_model", "list_models", "load_model",
]


def __getattr__(name):
    if name in _LAZY_MODELS:
        from . import models as _models

        return getattr(_models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_MODELS))
