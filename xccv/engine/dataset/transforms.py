"""图像增强算子：图像与标注点同步变换，保证检测/分割标注始终有效。

分类任务 shapes 恒为空列表，同样接口可复用。
"""

import random

import cv2
import numpy as np

from .base import Shape


def _scale_shapes(shapes: list[Shape], sx: float, sy: float) -> list[Shape]:
    return [
        Shape(label=s.label, shape_type=s.shape_type, points=[[x * sx, y * sy] for x, y in s.points])
        for s in shapes
    ]


def resize(image: np.ndarray, shapes: list[Shape], width: int, height: int,
           keep_ratio: bool = False, rng=None):
    """调整图像大小。

    keep_ratio=False（默认，平台统一口径）：直接拉伸到 宽×高 —— 训练、
    验证与预览所见一致，图像真实比例随设定改变（原图 4:3 设为 1:1 即显示 1:1）；
    训练输入尺寸与 resize 宽×高统一后，模型侧二次缩放为恒等。
    keep_ratio=True：等比缩放到恰好装进 宽×高 框内（只缩不变形）。
    """
    h, w = image.shape[:2]
    if keep_ratio:
        scale = min(width / w, height / h)
        nw, nh = max(2, round(w * scale)), max(2, round(h * scale))
        if (nw, nh) == (w, h):
            return image, shapes
        out = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return out, _scale_shapes(shapes, nw / w, nh / h)
    out = cv2.resize(image, (int(width), int(height)), interpolation=cv2.INTER_LINEAR)
    return out, _scale_shapes(shapes, width / w, height / h)


def _make_flip(direction: str):
    """生成固定方向的翻转算子（horizontal / vertical）。"""

    def random_flip(image: np.ndarray, shapes: list[Shape],
                    probability: float = 1.0, rng=None):
        """整图翻转（是否启用由每算子"概率"列控制），标注点同步镜像。

        probability 形参仅为兼容旧项目字段（旧值 0.5 仍生效），界面已不展示。
        """
        rng = rng or random
        if rng.random() >= probability:
            return image, shapes
        h, w = image.shape[:2]
        if direction == "horizontal":
            out = cv2.flip(image, 1)
            new_shapes = [Shape(label=s.label, shape_type=s.shape_type,
                                points=[[w - x, y] for x, y in s.points])
                          for s in shapes]
        else:
            out = cv2.flip(image, 0)
            new_shapes = [Shape(label=s.label, shape_type=s.shape_type,
                                points=[[x, h - y] for x, y in s.points])
                          for s in shapes]
        return out, new_shapes

    random_flip.__name__ = f"random_flip_{direction}"
    return random_flip


random_flip_h = _make_flip("horizontal")
random_flip_v = _make_flip("vertical")


def random_rescale(image: np.ndarray, shapes: list[Shape], min_scale: float = 0.8,
                   max_scale: float = 1.2, rng=None):
    """随机比例缩放：整图按 [min_scale, max_scale] 内随机比例缩放（可缩可放）。"""
    rng = rng or random
    h, w = image.shape[:2]
    scale = rng.uniform(min(min_scale, max_scale), max(min_scale, max_scale))
    nw, nh = max(2, round(w * scale)), max(2, round(h * scale))
    return resize(image, shapes, nw, nh)


def random_brightness(image: np.ndarray, shapes: list[Shape], max_delta: float = 30, rng=None):
    """随机调整亮度：全图加一个 [-max_delta, max_delta] 的随机偏移。"""
    rng = rng or random
    delta = rng.uniform(-max_delta, max_delta)
    out = np.clip(image.astype(np.float32) + delta, 0, 255).astype(np.uint8)
    return out, shapes


def random_contrast(image: np.ndarray, shapes: list[Shape], low: float = 0.6, high: float = 1.4, rng=None):
    """随机调整对比度：以各通道均值为基准随机缩放。"""
    rng = rng or random
    factor = rng.uniform(low, high)
    mean = image.mean(axis=(0, 1), keepdims=True)
    out = np.clip((image.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)
    return out, shapes


def _rotate_shapes(shapes: list[Shape], w: int, h: int, clockwise: bool) -> list[Shape]:
    """90° 旋转的标注点同步（连续坐标 + 边界裁剪；旋转后 new_w=h、new_h=w）。"""
    out = []
    for s in shapes:
        pts = []
        for x, y in s.points:
            nx, ny = (h - y, x) if clockwise else (y, w - x)
            pts.append([min(max(nx, 0.0), float(h)), min(max(ny, 0.0), float(w))])
        out.append(Shape(label=s.label, shape_type=s.shape_type, points=pts))
    return out


def rotate90(image: np.ndarray, shapes: list[Shape], rng=None):
    """随机 90°/270° 旋转（工件任意朝向场景），标注点同步旋转。

    旋转后宽高互换：数据集图像长宽比一致时训练输入仍一致；长宽比混杂时
    由训练前守卫拦截，或启用 random_crop（裁剪比例=归一化尺寸比例）兜底。
    """
    rng = rng or random
    h, w = image.shape[:2]
    if rng.random() < 0.5:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), _rotate_shapes(shapes, w, h, True)
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), _rotate_shapes(shapes, w, h, False)


def random_crop(image: np.ndarray, shapes: list[Shape], ratio: float = 1.0,
                min_cover: float = 0.5, rng=None):
    """随机裁剪到指定宽高比（rotate90 的兜底：比例=归一化尺寸比例时，
    旋转后接本算子可使训练输入长宽比重新一致）。

    图像宽高比已与目标一致（±2%）时原样返回；否则裁出面积 ≥min_cover、
    宽高比=ratio 的窗口（随机位置），标注点同步平移并裁到窗内。
    """
    rng = rng or random
    h, w = image.shape[:2]
    ratio = max(1e-3, float(ratio))
    if abs(w / h - ratio) / ratio < 0.02:
        return image, shapes
    cover = rng.uniform(min(min_cover, 1.0), 1.0)
    area = w * h * cover
    cw, ch = (area * ratio) ** 0.5, (area / ratio) ** 0.5
    if cw > w or ch > h:  # 窗口超出图像（极端比例）时等比缩小到贴边
        scale = min(w / cw, h / ch)
        cw, ch = cw * scale, ch * scale
    cw, ch = max(2, int(cw)), max(2, int(ch))
    x1 = rng.randint(0, w - cw)
    y1 = rng.randint(0, h - ch)
    out = image[y1:y1 + ch, x1:x1 + cw]
    new_shapes = []
    for s in shapes:
        pts = [[min(max(x - x1, 0.0), float(cw)), min(max(y - y1, 0.0), float(ch))]
               for x, y in s.points]
        new_shapes.append(Shape(label=s.label, shape_type=s.shape_type, points=pts))
    return out, new_shapes


def random_hsv_shift(image: np.ndarray, shapes: list[Shape], h_delta: float = 10,
                     s_delta: float = 0.25, v_delta: float = 0.25, rng=None):
    """随机 HSV 色彩扰动（工业光照/色温变化的刚需）。光度变换，标注不变。"""
    rng = rng or random
    dh = rng.uniform(-abs(h_delta), abs(h_delta))
    ds = rng.uniform(-abs(s_delta), abs(s_delta))
    dv = rng.uniform(-abs(v_delta), abs(v_delta))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = np.mod(hsv[..., 0] + dh, 180.0)
    hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + ds), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * (1.0 + dv), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), shapes


def gaussian_blur(image: np.ndarray, shapes: list[Shape], ksize: int = 3,
                  sigma: float = 0.6, rng=None):
    """轻微高斯模糊（模拟失焦；sigma 越小越轻）。光度变换，标注不变。"""
    k = max(1, int(ksize) | 1)  # 强制奇数
    if k <= 1:
        return image, shapes
    return cv2.GaussianBlur(image, (k, k), sigmaX=max(0.0, float(sigma))), shapes


def gaussian_noise(image: np.ndarray, shapes: list[Shape], sigma: float = 3, rng=None):
    """随机高斯噪声（模拟传感器噪声）。光度变换，标注不变。"""
    sigma = abs(float(sigma))
    if sigma <= 0:
        return image, shapes
    noise = np.random.normal(0.0, sigma, image.shape)
    out = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out, shapes


def random_erase(image: np.ndarray, shapes: list[Shape], max_ratio: float = 0.1, rng=None):
    """随机矩形擦除 CutOut（模拟遮挡/反光），区域填充图像均值灰色。

    遮挡语义下标注保持不变（目标框内的被遮挡部分由模型学习容忍）。
    """
    rng = rng or random
    h, w = image.shape[:2]
    mr = min(abs(float(max_ratio)), 0.9)
    eh = max(1, int(h * rng.uniform(0.02, mr)))
    ew = max(1, int(w * rng.uniform(0.02, mr)))
    x1 = rng.randint(0, w - ew)
    y1 = rng.randint(0, h - eh)
    out = image.copy()
    out[y1:y1 + eh, x1:x1 + ew] = image.mean(axis=(0, 1)).astype(np.uint8)
    return out, shapes


# 算子注册表：名称 -> (函数, 默认参数)。界面与数据倍增统一从这里取。
# 顺序即应用顺序：几何类在前（rotate90 之后接 random_crop 兜底），光度类在后。
AUGMENT_OPS: dict[str, tuple] = {
    "resize": (resize, {"width": 640, "height": 640}),
    "random_rescale": (random_rescale, {"min_scale": 0.8, "max_scale": 1.2}),
    "random_flip_h": (random_flip_h, {}),
    "random_flip_v": (random_flip_v, {}),
    "rotate90": (rotate90, {}),
    "random_crop": (random_crop, {"ratio": 1.0, "min_cover": 0.7}),
    "random_brightness": (random_brightness, {"max_delta": 20}),
    "random_contrast": (random_contrast, {"low": 0.8, "high": 1.2}),
    "random_hsv_shift": (random_hsv_shift, {"h_delta": 6, "s_delta": 0.15, "v_delta": 0.15}),
    "gaussian_blur": (gaussian_blur, {"ksize": 3, "sigma": 0.6}),
    "gaussian_noise": (gaussian_noise, {"sigma": 3}),
    "random_erase": (random_erase, {"max_ratio": 0.1}),
}


def apply_ops(image: np.ndarray, shapes: list[Shape], ops: dict, rng=None):
    """按给定策略依次应用增强。ops 形如 {"random_brightness": {"max_delta": 40}, ...}。"""
    for name, params in ops.items():
        fn, defaults = AUGMENT_OPS[name]
        image, shapes = fn(image, shapes, **{**defaults, **(params or {})}, rng=rng)
    return image, shapes


# 单算子默认启用概率：每个算子独立掷硬币（旧版的"整套一掷"由 per-op 概率取代）
AUGMENT_PROB = 0.5


def maybe_apply_ops(image: np.ndarray, shapes: list[Shape], ops: dict, rng=None,
                    probs: dict | None = None):
    """按每算子启用概率随机应用增强（随机增强，与数据倍增设置解耦）。

    probs 形如 {"random_flip_h": 0.5, ...}；未列出的算子按 AUGMENT_PROB。
    每个算子独立决定是否应用：训练中每个 epoch 看到的是各算子随机组合。
    rng 为 None 时使用全局 random 模块。
    """
    if not ops:
        return image, shapes
    rng = rng if rng is not None else random
    probs = probs or {}
    chosen = {name: params for name, params in ops.items()
              if rng.random() < float(probs.get(name, AUGMENT_PROB))}
    if not chosen:
        return image, shapes
    return apply_ops(image, shapes, chosen, rng=rng)
