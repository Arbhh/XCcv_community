"""通用小工具：JSON 读写、中文路径安全的图片读写。"""

import json
from pathlib import Path

import cv2
import numpy as np


def save_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def imread_unicode(path, color: bool = True) -> np.ndarray | None:
    """cv2.imread 在 Windows 中文路径下会失败，这里用 fromfile+imdecode 代替。

    color=False（O5，V0.7 灰度训练）：IMREAD_GRAYSCALE 解码后复制回 3 通道
    ——灰度信息保留、下游张量化/增强/骨干零改动（"灰度复制 3 通道"口径）。
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    if color:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def imwrite_unicode(path, image: np.ndarray) -> bool:
    """cv2.imwrite 在 Windows 中文路径下会失败，这里用 imencode+tofile 代替。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, image)
    if ok:
        buf.tofile(str(path))
    return bool(ok)
