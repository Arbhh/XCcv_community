"""标注平台子包（基于内置的 labelme 修改版，AI 标注功能已移除）。

注意：必须在导入任何 Qt/labelme 模块之前固定 Qt 绑定为 PyQt5，
否则 qtpy 可能选择其他绑定，与 labelme 产生冲突。
"""

import os

os.environ.setdefault("QT_API", "pyqt5")
