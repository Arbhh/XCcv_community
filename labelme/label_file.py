import base64
import contextlib
import io
import json
import os.path as osp

import PIL.Image
from loguru import logger

from labelme import __version__
from labelme import utils

PIL.Image.MAX_IMAGE_PIXELS = None


@contextlib.contextmanager
def open(name, mode):
    assert mode in ["r", "w"]
    encoding = "utf-8"
    yield io.open(name, mode, encoding=encoding)
    return


class LabelFileError(Exception):
    pass


class LabelFile(object):
    suffix = ".json"

    def __init__(self, filename=None):
        self.shapes = []
        self.imagePath = None
        self.imageData = None
        if filename is not None:
            self.load(filename)
        self.filename = filename

    @staticmethod
    def load_image_file(filename):
        try:
            image_pil = PIL.Image.open(filename)
        except IOError:
            logger.error("Failed opening image file: {}".format(filename))
            return

        # apply orientation to image according to exif
        image_pil = utils.apply_exif_orientation(image_pil)
        if image_pil.mode not in ("RGB", "RGBA"):
            # 灰度(L)/调色板等单通道图统一转 RGB——灰度 JPEG 在
            # PIL/Qt/cv2 间存在亮度范围解释分歧（实测整图偏暗 ~10 灰度、
            # 62% 像素 diff>8），AI(SAM) 编码随之把小目标解成大背景块
            image_pil = image_pil.convert("RGB")

        with io.BytesIO() as f:
            ext = osp.splitext(filename)[1].lower()
            if ext in [".jpg", ".jpeg"]:
                format = "JPEG"
            else:
                format = "PNG"
            if format == "JPEG":
                # 高保真重存（2026-11-06）：默认 q75+4:2:0 使 imageData 与
                # 原图平均差 ~21 灰度，AI(SAM) 编码在小目标上把重压缩噪声
                # 当纹理 → 解出大背景块；q95 + 4:4:4 差异回落到 ~1
                image_pil.save(f, format=format, quality=95,
                               subsampling=0)
            else:
                image_pil.save(f, format=format)
            f.seek(0)
            return f.read()

    def load(self, filename):
        keys = [
            "version",
            "imageData",
            "imagePath",
            "shapes",  # polygonal annotations
            "flags",  # image level flags
            "imageHeight",
            "imageWidth",
        ]
        shape_keys = [
            "label",
            "points",
            "group_id",
            "shape_type",
            "flags",
            "description",
            "mask",
        ]
        try:
            # open 为本模块自定义包装（io.open+utf-8）；勿加 encoding 参数
            with open(filename, "r") as f:
                data = json.load(f)

            if data["imageData"] is not None:
                imageData = base64.b64decode(data["imageData"])
            else:
                # relative path from label file to relative path from cwd
                imagePath = osp.join(osp.dirname(filename), data["imagePath"])
                imageData = self.load_image_file(imagePath)
            flags = data.get("flags") or {}
            imagePath = data["imagePath"]
            self._check_image_height_and_width(
                base64.b64encode(imageData).decode("utf-8"),
                data.get("imageHeight"),
                data.get("imageWidth"),
            )
            shapes = [
                dict(
                    label=s["label"],
                    points=s["points"],
                    shape_type=s.get("shape_type", "polygon"),
                    flags=s.get("flags", {}),
                    description=s.get("description"),
                    group_id=s.get("group_id"),
                    mask=utils.img_b64_to_arr(s["mask"]).astype(bool)
                    if s.get("mask")
                    else None,
                    other_data={k: v for k, v in s.items() if k not in shape_keys},
                )
                for s in data["shapes"]
            ]
        except Exception as e:
            raise LabelFileError(e)

        otherData = {}
        for key, value in data.items():
            if key not in keys:
                otherData[key] = value

        # Only replace data after everything is loaded.
        self.flags = flags
        self.shapes = shapes
        self.imagePath = imagePath
        self.imageData = imageData
        self.filename = filename
        self.otherData = otherData

    @staticmethod
    def _check_image_height_and_width(imageData, imageHeight, imageWidth):
        img_arr = utils.img_b64_to_arr(imageData)
        if imageHeight is not None and img_arr.shape[0] != imageHeight:
            logger.error(
                "imageHeight does not match with imageData or imagePath, "
                "so getting imageHeight from actual image."
            )
            imageHeight = img_arr.shape[0]
        if imageWidth is not None and img_arr.shape[1] != imageWidth:
            logger.error(
                "imageWidth does not match with imageData or imagePath, "
                "so getting imageWidth from actual image."
            )
            imageWidth = img_arr.shape[1]
        return imageHeight, imageWidth

    def save(
        self,
        filename,
        shapes,
        imagePath,
        imageHeight,
        imageWidth,
        imageData=None,
        otherData=None,
        flags=None,
    ):
        if imageData is not None:
            imageData = base64.b64encode(imageData).decode("utf-8")
            imageHeight, imageWidth = self._check_image_height_and_width(
                imageData, imageHeight, imageWidth
            )
        if otherData is None:
            otherData = {}
        if flags is None:
            flags = {}
        data = dict(
            version=__version__,
            flags=flags,
            shapes=shapes,
            imagePath=imagePath,
            imageData=imageData,
            imageHeight=imageHeight,
            imageWidth=imageWidth,
        )
        for key, value in otherData.items():
            assert key not in data
            data[key] = value
        try:
            # open 为本模块自定义包装（io.open+utf-8）；勿加 encoding 参数
            with open(filename, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.filename = filename
        except Exception as e:
            raise LabelFileError(e)

    @staticmethod
    def is_label_file(filename):
        return osp.splitext(filename)[1].lower() == LabelFile.suffix
