"""XCcv 标注平台语言设置与翻译层（切换中英文，重启后生效）。

翻译查找顺序（MergedZhTranslator）：
1. 补充映射（填补官方 zh_CN.qm 缺失/未完成的常用键）
2. 官方 zh_CN.qm
3. 均未命中 → 返回 None（Qt 回退到英文原文）

注意：PyQt 中返回空串会被 Qt 当作“已翻译的空文本”阻断回退链，
因此未命中必须返回 None 而不是空串。
"""

import os

from PyQt5 import QtCore

_SETTINGS_ORG = "XCcv"
_SETTINGS_APP = "annotation"

# (context, source) -> 中文；仅收录官方 zh_CN.qm 缺失或未完成的常用键
_ZH_SUPPLEMENT = {
    ("MainWindow", "Save"): "保存",
    ("MainWindow", "&Save"): "保存(&S)",
    ("MainWindow", "Save As"): "另存为",
    ("MainWindow", "&Delete"): "删除(&D)",
    ("MainWindow", "Delete File"): "删除文件",
    ("MainWindow", "Zoom in"): "放大",
    ("MainWindow", "Zoom out"): "缩小",
    ("MainWindow", "Zoom to Original"): "原始大小",
    ("MainWindow", "Fit Window"): "适应窗口",
    ("MainWindow", "Fit Width"): "适应宽度",
    ("MainWindow", "Help"): "帮助",
    ("MainWindow", "&Help"): "帮助(&H)",
    ("MainWindow", "&File"): "文件(&F)",
    ("MainWindow", "&Edit"): "编辑(&E)",
    ("MainWindow", "&View"): "视图(&V)",
    ("MainWindow", "File List"): "文件列表",
    ("MainWindow", "Search Filename"): "搜索文件名",
    ("MainWindow", "Labels"): "标签",
    ("MainWindow", "Polygons"): "多边形",
    ("MainWindow", "Flags"): "标志",
    ("MainWindow", "Error opening file"): "打开文件出错",
    ("MainWindow", "Open"): "打开",
    ("MainWindow", "&Open"): "打开(&O)",
    ("MainWindow", "Close"): "关闭",
    ("MainWindow", "&Close"): "关闭(&C)",
    ("MainWindow", "&Quit"): "退出(&Q)",
    ("MainWindow", "Quit"): "退出",
    ("MainWindow", "Create Circle"): "创建圆形",
    ("MainWindow", "Create LineStrip"): "创建线条",
    ("MainWindow", "Create Point"): "创建点",
    ("MainWindow", "Create Line"): "创建线",
    ("MainWindow", "Keep Previous Annotation"): "保留上一标注",
    ("MainWindow", "Brightness Contrast"): "亮度对比度",
    ("MainWindow", "Show All Polygons"): "显示全部多边形",
    ("MainWindow", "Hide All Polygons"): "隐藏全部多边形",
    ("MainWindow", "Toggle All Polygons"): "切换全部多边形",
    ("MainWindow", "Open &Recent"): "最近打开",
    ("MainWindow", "Language"): "语言",
}

_LANG_NAMES = {"zh_CN": "简体中文", "en": "English"}


class MergedZhTranslator(QtCore.QTranslator):
    """官方 zh_CN.qm + 补充映射合并为一个翻译层。

    查找顺序：补充映射 → 官方 qm → 返回 None（Qt 回退到英文原文）。
    注意：PyQt 中返回空串会被 Qt 当作“已翻译的空文本”阻断回退链，
    因此未命中必须返回 None 而不是空串。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._official = QtCore.QTranslator(self)
        path = os.path.join(os.path.dirname(__file__), "translate")
        self._official.load("zh_CN", path)

    def translate(self, context, source_text, disambiguation=None, n=-1):
        # XCcv：C++ 虚回调传入的 const char* 参数是 bytes，统一解码为 str
        try:
            if isinstance(context, bytes):
                context = context.decode("utf-8", "replace")
            if isinstance(source_text, bytes):
                source_text = source_text.decode("utf-8", "replace")
            zh = _ZH_SUPPLEMENT.get((context, source_text))
            if zh is not None:
                return zh
            t = self._official.translate(context, source_text, disambiguation, n)
            return t if t else None
        except Exception:
            # 防御：个别键无法编码/查询时回退原文，绝不让翻译层拖垮进程
            return None


def language_name(lang: str) -> str:
    return _LANG_NAMES.get(lang, lang)


def current_language() -> str:
    settings = QtCore.QSettings(_SETTINGS_ORG, _SETTINGS_APP)
    return str(settings.value("language", "zh_CN"))


def set_language(lang: str) -> None:
    settings = QtCore.QSettings(_SETTINGS_ORG, _SETTINGS_APP)
    settings.setValue("language", lang)


_installed_translators: list = []  # 持有引用，防止 Python 包装对象被回收后 Qt 调用虚函数崩溃


def apply_language(app) -> None:
    """按用户设置安装翻译（必须在任何窗口部件创建之前调用）。"""
    if current_language() != "zh_CN":
        return
    translator = MergedZhTranslator(app)
    app.installTranslator(translator)
    _installed_translators.append(translator)  # 关键：保活 Python 包装对象
