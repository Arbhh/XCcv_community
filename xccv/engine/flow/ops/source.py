"""图像源类算子（批次十五附带入库：image_loader 是全库最简节点，
供算子工作台/模板冒烟直接从文件取图；文件夹逐张源按 D4 决策缓做，
批处理继续走测试平台/流程图平台页的批量线程——此处留接口占位注释）。
"""

from ...io_utils import imread_unicode
from ..ir import ImageData
from ..node_base import ParamSpec, Port, VisionNode
from ..registry import register_node


@register_node
class ImageLoader(VisionNode):
    """从文件加载一张图片（支持中文路径）。

    输出 BGR uint8；is_grayscale 勾选时转单通道灰度。
    """

    NAME = "image_loader"
    TITLE = "读入图片"
    CATEGORY = "图像源"
    INPUTS = ()
    OUTPUTS = (Port("image", "image", label="图像"),)
    PARAMS = {
        "path": ParamSpec("path", "图片路径", "path", "",
                          tooltip="待加载的图片文件（jpg/png/bmp 等）"),
        "is_grayscale": ParamSpec("is_grayscale", "转为灰度", "bool", False,
                                  tooltip="勾选后输出单通道灰度图"),
    }

    def run(self, inputs, params, ctx):
        path = str(params.get("path") or "")
        if not path:
            raise ValueError("未设置图片路径")
        try:
            img = imread_unicode(path)
        except OSError as e:
            raise ValueError(f"无法读取图片：{path}（{e}）") from e
        if img is None:
            raise ValueError(f"无法读取图片：{path}")
        if params.get("is_grayscale") and img.ndim == 3:
            import cv2

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return {"image": ImageData(mat=img, meta={"source_node": self.NAME,
                                                  "path": path})}


# 占位（D4 决策：文件夹逐张源缓做）——批处理走测试平台/流程图平台页批量线程；
# 二期如需会话游标语义，在此文件内新增 image_folder_source（ctx 维护游标）。
