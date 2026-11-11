"""推理结果可视化渲染：bbox、掩码、类别+置信度标签（支持中文类别名）。"""

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger(__name__)

# 类别配色（BGR），按类别索引循环使用
_PALETTE = [
    (230, 159, 0), (0, 114, 178), (86, 180, 233), (0, 158, 115),
    (240, 228, 66), (213, 94, 0), (204, 121, 167), (0, 0, 0),
    (89, 89, 89), (166, 206, 227),
]

_FONT_PATHS = ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]
_font_cache: dict = {}


def _font(size: int = 16):
    if size not in _font_cache:
        font = None
        for path in _FONT_PATHS:
            if Path(path).exists():
                try:
                    font = ImageFont.truetype(path, size)
                    break
                except OSError:
                    continue
        _font_cache[size] = font or ImageFont.load_default()
    return _font_cache[size]


def class_color(index: int) -> tuple:
    return _PALETTE[index % len(_PALETTE)]


def _align_mask(mask, image_hw):
    """把掩码对齐到图像尺寸；无法对齐时返回 None（防御，T1-1 双保险）。

    推理侧应已把掩码还原到原图尺寸；这里兜底防止任何来源的分辨率不一致
    再把 IndexError 漏成未捕获异常：形状不符先尝试最近邻缩放，仍失败跳过。
    """
    mask = np.asarray(mask)
    h, w = int(image_hw[0]), int(image_hw[1])
    if mask.ndim != 2 or mask.size == 0:
        return None
    if mask.shape == (h, w):
        return mask.astype(bool)
    try:
        resized = cv2.resize(mask.astype(np.uint8), (w, h),
                             interpolation=cv2.INTER_NEAREST)
        return resized > 0
    except cv2.error:
        return None


def draw_predictions(image_bgr: np.ndarray, prediction, classes: list[str],
                     task: str, show_labels: bool = True,
                     show_masks: bool = True) -> np.ndarray:
    """把一次预测结果画到图片副本上并返回。

    task: "detection" / "segmentation" / "classification"
    """
    out = image_bgr.copy()

    if task == "classification":
        if prediction.labels and show_labels:
            label = classes[prediction.labels[0]] if prediction.labels[0] < len(classes) else "?"
            score = prediction.scores[0] if prediction.scores else 0.0
            out = _draw_text(out, f"{label}  {score:.2f}", (8, 8), class_color(0))
        return out

    # 判空不用真值（防御：上游若给 3 维 ndarray，`if masks:` 真值歧义抛错，P150）
    if show_masks and prediction.masks is not None and len(prediction.masks) > 0:
        h, w = out.shape[:2]
        for raw_mask, idx in zip(prediction.masks, prediction.labels):
            color = class_color(idx)
            mask = _align_mask(raw_mask, (h, w))
            if mask is None:
                shape = getattr(raw_mask, "shape", None)
                log.warning("掩码形状 %s 与图像 (%d, %d) 不符且无法对齐，已跳过该实例",
                            shape, h, w)
                continue
            overlay = out.copy()
            overlay[mask] = color
            out = cv2.addWeighted(overlay, 0.4, out, 0.6, 0)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, color, 2)

    for box, idx, score in zip(prediction.boxes, prediction.labels, prediction.scores):
        color = class_color(idx)
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if show_labels:
            name = classes[idx] if idx < len(classes) else str(idx)
            out = _draw_text(out, f"{name}  {score:.2f}", (x1, max(y1 - 22, 0)), color)
    return out


def _draw_text(image_bgr: np.ndarray, text: str, origin_xy, color_bgr) -> np.ndarray:
    """用 PIL 绘制文字（cv2.putText 不支持中文类别名）。"""
    x, y = origin_xy
    img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font = _font(16)
    left, top, right, bottom = draw.textbbox((x, y), text, font=font, stroke_width=1)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.rectangle([left - 2, top - 1, right + 2, bottom + 1], fill=color_rgb)
    draw.text((x, y), text, font=font, fill=(255, 255, 255), stroke_width=1,
              stroke_fill=(0, 0, 0))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
