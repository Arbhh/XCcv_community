# MIT License
# Copyright (c) Kentaro Wada

import base64
import io

import numpy as np
import PIL.ExifTags
import PIL.Image
import PIL.ImageOps


def img_data_to_pil(img_data):
    f = io.BytesIO()
    f.write(img_data)
    img_pil = PIL.Image.open(f)
    return img_pil


def img_data_to_arr(img_data):
    img_pil = img_data_to_pil(img_data)
    img_arr = np.array(img_pil)
    return img_arr


def img_b64_to_arr(img_b64):
    img_data = base64.b64decode(img_b64)
    img_arr = img_data_to_arr(img_data)
    return img_arr


def img_pil_to_data(img_pil):
    f = io.BytesIO()
    img_pil.save(f, format="PNG")
    img_data = f.getvalue()
    return img_data


def img_arr_to_b64(img_arr):
    img_data = img_arr_to_data(img_arr)
    img_b64 = base64.b64encode(img_data).decode("utf-8")
    return img_b64


def img_arr_to_data(img_arr):
    img_pil = PIL.Image.fromarray(img_arr)
    img_data = img_pil_to_data(img_pil)
    return img_data


def img_data_to_png_data(img_data):
    with io.BytesIO() as f:
        f.write(img_data)
        img = PIL.Image.open(f)

        with io.BytesIO() as f:
            img.save(f, "PNG")
            f.seek(0)
            return f.read()


def img_qt_to_arr(img_qt):
    """QImage → RGB uint8 ndarray（H, W, 3）。

    2026-11-06 统一修复：非 RGB888 一律 convertToFormat(Format_RGB888)。
    原实现 bits() 直读——toImage() 常给 ARGB32_Premultiplied（预乘 alpha
    + BGRA 序），直读出现整体亮度偏移/通道错位（实测灰度图 qt=111 vs 源
    =124，SAM 编码受污染把小目标解成大背景块）；灰度图则 depth=8 按 1
    通道 reshape，调用方按 RGB 采样错位。RGB888 是非预乘、RGB 字节序，
    所有按 RGB 消费的调用方（SAM 编码/准星亮度/批量预标注）一致正确。
    """
    from PyQt5 import QtGui as _QtGui

    if img_qt.format() != _QtGui.QImage.Format_RGB888:
        img_qt = img_qt.convertToFormat(_QtGui.QImage.Format_RGB888)
    w, h = img_qt.width(), img_qt.height()
    bpl = img_qt.bytesPerLine()   # 行按 4 字节对齐，w*3 非对齐时必须逐行取
    if bpl == w * 3:
        bytes_ = img_qt.bits().asstring(w * h * 3)
        arr = np.frombuffer(bytes_, dtype=np.uint8).reshape((h, w, 3))
    else:
        arr = np.empty((h, w, 3), dtype=np.uint8)
        raw = img_qt.bits().asstring(bpl * h)
        for y in range(h):
            row = raw[y * bpl:y * bpl + w * 3]
            arr[y] = np.frombuffer(row, dtype=np.uint8).reshape((w, 3))
    return np.ascontiguousarray(arr[..., ::-1])   # 内存 RGB888 → 输出 RGB


def apply_exif_orientation(image):
    try:
        exif = image._getexif()
    except AttributeError:
        exif = None

    if exif is None:
        return image

    exif = {PIL.ExifTags.TAGS[k]: v for k, v in exif.items() if k in PIL.ExifTags.TAGS}

    orientation = exif.get("Orientation", None)

    if orientation == 1:
        # do nothing
        return image
    elif orientation == 2:
        # left-to-right mirror
        return PIL.ImageOps.mirror(image)
    elif orientation == 3:
        # rotate 180
        return image.transpose(PIL.Image.ROTATE_180)
    elif orientation == 4:
        # top-to-bottom mirror
        return PIL.ImageOps.flip(image)
    elif orientation == 5:
        # top-to-left mirror
        return PIL.ImageOps.mirror(image.transpose(PIL.Image.ROTATE_270))
    elif orientation == 6:
        # rotate 270
        return image.transpose(PIL.Image.ROTATE_270)
    elif orientation == 7:
        # top-to-right mirror
        return PIL.ImageOps.mirror(image.transpose(PIL.Image.ROTATE_90))
    elif orientation == 8:
        # rotate 90
        return image.transpose(PIL.Image.ROTATE_90)
    else:
        return image
