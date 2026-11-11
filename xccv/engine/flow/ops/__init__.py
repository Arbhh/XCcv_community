"""传统视觉算子库（ops）。

导入本包即完成全部算子注册（各子模块底部的 import 副作用）。
算子实现纪律：
- 纯 OpenCV/NumPy，禁 Qt、禁 UI 调用（反馈走 ctx.warn / 快照）；
- 参数全部走 PARAMS schema（中文标签 + tooltip），执行前自动校验；
- 彩色/灰度行为写在每个算子 docstring；
- 图像引用传递，禁止对输入全图 .copy()。

分类（前端节点库分组用）：图像源 / 预处理 / 定位 / 逻辑 / AI。
"""


def ensure_odd(value: int, ctx, label: str) -> int:
    """核数类参数自动补奇（任务书约定：非奇数 +1，并记录人话警告）。"""
    v = int(value)
    if v % 2 == 0:
        v += 1
        ctx.warn(f"「{label}」{value} 为偶数，已自动取奇数 {v}")
    return max(1, v)


def to_gray(mat):
    """彩色输入转灰度（threshold/canny/sobel/clahe/fft 等单通道算子用）。

    灰度输入原样返回；彩色 BGR 按亮度加权转换（cv2 标准）。
    """
    import cv2

    if mat.ndim == 2:
        return mat
    return cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)


def load_all() -> None:
    """导入全部算子子模块完成注册（幂等，import 天然幂等）。

    画布节点库 / 算子工作台初始化时调用；五平台主流程**不**调用——
    保持主界面冷启动不引入 flow/ops 的导入开销（Q6 性能纪律）。
    旧过滤链的 8 个薄委托节点在 engine/flow/legacy_adapter.py，
    import 该模块即完成注册（不归本函数管）。
    自定义算子（platform_data/operators/*.op.json，V0.6 T4-4）在此首次读盘
    注册（懒加载同一时机）；之后由工作台「刷新」显式热重载。
    """
    from . import (  # noqa: F401
        ai_bridge, crop, listops, logic, output, source,
    )
    from .. import custom_ops

    try:
        custom_ops.ensure_loaded()
    except Exception:  # noqa: BLE001  坏的自定义算子不得拖垮内置算子库
        import logging

        logging.getLogger(__name__).warning("自定义算子加载失败", exc_info=True)
