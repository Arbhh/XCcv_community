# -*- coding: utf-8 -*-

import functools
import html
import math
import os
import os.path as osp
import re
import webbrowser

import imgviz
import natsort
import numpy as np
from loguru import logger
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

from labelme import __appname__
from labelme.config import get_config
from labelme.label_file import LabelFile
from labelme.language import (
    apply_language, current_language, language_name, set_language,
)
from labelme.label_file import LabelFileError
from labelme.shape import Shape
from labelme.widgets import BrightnessContrastDialog
from labelme.widgets import Canvas
from labelme.widgets import FileDialogPreview
from labelme.widgets import LabelDialog
from labelme.widgets import LabelListWidget
from labelme.widgets import LabelListWidgetItem
from labelme.widgets import ToolBar
from labelme.widgets import UniqueLabelQListWidget
from labelme.widgets import ZoomWidget

from . import utils

# FIXME
# - [medium] Set max zoom value to something big enough for FitWidth/Window

# TODO(unknown):
# - Zoom is too "steppy".


LABEL_COLORMAP = imgviz.label_colormap()


class MainWindow(QtWidgets.QMainWindow):
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = 0, 1, 2

    def __init__(
        self,
        config=None,
        filename=None,
        output=None,
        output_file=None,
        output_dir=None,
    ):
        if output is not None:
            logger.warning("argument output is deprecated, use output_file instead")
            if output_file is None:
                output_file = output

        # see labelme/config/default_config.yaml for valid configuration
        if config is None:
            config = get_config()
        self._config = config

        # XCcv：语言设置（必须在任何窗口部件创建之前安装翻译）
        apply_language(QtWidgets.QApplication.instance())

        # set default shape colors
        Shape.line_color = QtGui.QColor(*self._config["shape"]["line_color"])
        Shape.fill_color = QtGui.QColor(*self._config["shape"]["fill_color"])
        Shape.select_line_color = QtGui.QColor(
            *self._config["shape"]["select_line_color"]
        )
        Shape.select_fill_color = QtGui.QColor(
            *self._config["shape"]["select_fill_color"]
        )
        Shape.vertex_fill_color = QtGui.QColor(
            *self._config["shape"]["vertex_fill_color"]
        )
        Shape.hvertex_fill_color = QtGui.QColor(
            *self._config["shape"]["hvertex_fill_color"]
        )

        # Set point size from config file
        Shape.point_size = self._config["shape"]["point_size"]

        super(MainWindow, self).__init__()
        self.setWindowTitle(__appname__)

        # Whether we need to save or not.
        self.dirty = False

        self._noSelectionSlot = False

        self._copied_shapes = None

        # Main widgets and related state.
        self.labelDialog = LabelDialog(
            parent=self,
            labels=self._config["labels"],
            sort_labels=self._config["sort_labels"],
            show_text_field=self._config["show_label_text_field"],
            completion=self._config["label_completion"],
            fit_to_content=self._config["fit_to_content"],
            flags=self._config["label_flags"],
        )

        self.labelList = LabelListWidget()
        self.lastOpenDir = None
        # XCcv V0.6 T5-6：上次目录记忆（ui_prefs，跨启动）
        try:
            from xccv.ui_widgets import load_prefs

            _saved_dir = load_prefs().get("annotation_last_dir")
            if _saved_dir and osp.exists(str(_saved_dir)):
                self.lastOpenDir = str(_saved_dir)
        except Exception:  # noqa: BLE001  偏好读取失败不影响启动
            pass

        self.flag_dock = self.flag_widget = None
        self.flag_dock = QtWidgets.QDockWidget(self.tr("Flags"), self)
        self.flag_dock.setObjectName("Flags")
        self.flag_widget = QtWidgets.QListWidget()
        if config["flags"]:
            self.loadFlags({k: False for k in config["flags"]})
        self.flag_dock.setWidget(self.flag_widget)
        self.flag_widget.itemChanged.connect(self.setDirty)

        self.labelList.itemSelectionChanged.connect(self.labelSelectionChanged)
        self.labelList.itemDoubleClicked.connect(self._edit_label)
        self.labelList.itemChanged.connect(self.labelItemChanged)
        self.labelList.itemDropped.connect(self.labelOrderChanged)
        self.shape_dock = QtWidgets.QDockWidget(self.tr("Polygon Labels"), self)
        self.shape_dock.setObjectName("Labels")
        self.shape_dock.setWidget(self.labelList)

        self.uniqLabelList = UniqueLabelQListWidget()
        self.uniqLabelList.setToolTip(
            self.tr(
                "Select label to start annotating for it. " "Press 'Esc' to deselect."
            )
        )
        if self._config["labels"]:
            for label in self._config["labels"]:
                item = self.uniqLabelList.createItemFromLabel(label)
                self.uniqLabelList.addItem(item)
                rgb = self._get_rgb_by_label(label)
                self.uniqLabelList.setItemLabel(item, label, rgb)
        self.label_dock = QtWidgets.QDockWidget(self.tr("Label List"), self)
        self.label_dock.setObjectName("Label List")
        self.label_dock.setWidget(self.uniqLabelList)

        self.fileSearch = QtWidgets.QLineEdit()
        self.fileSearch.setPlaceholderText(self.tr("Search Filename"))
        self.fileSearch.textChanged.connect(self.fileSearchChanged)
        # XCcv：文件列表改为树形（子文件夹可展开，图片显示原名）
        self.fileListWidget = QtWidgets.QTreeWidget()
        self.fileListWidget.setHeaderHidden(True)
        self.fileListWidget.itemSelectionChanged.connect(self.fileSelectionChanged)
        # XCcv V0.6 T5-6：右键「打开所在文件夹」
        self.fileListWidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.fileListWidget.customContextMenuRequested.connect(
            self._file_list_menu)
        self._file_items: dict = {}   # 完整路径 -> 树节点
        self._file_order: list = []   # 展平后的图片路径顺序（导航用）
        fileListLayout = QtWidgets.QVBoxLayout()
        fileListLayout.setContentsMargins(0, 0, 0, 0)
        fileListLayout.setSpacing(0)
        fileListLayout.addWidget(self.fileSearch)
        self._fltDone = QtWidgets.QCheckBox(self.tr("已标注"))
        self._fltTodo = QtWidgets.QCheckBox(self.tr("未标注"))
        self._fltDone.toggled.connect(self.fileSearchChanged)
        self._fltTodo.toggled.connect(self.fileSearchChanged)
        _fltRow = QtWidgets.QHBoxLayout()
        _fltRow.setContentsMargins(4, 2, 4, 2)
        _fltRow.addWidget(self._fltDone)
        _fltRow.addWidget(self._fltTodo)
        _fltRow.addStretch(1)
        fileListLayout.addLayout(_fltRow)
        fileListLayout.addWidget(self.fileListWidget)
        self.file_dock = QtWidgets.QDockWidget(self.tr("File List"), self)
        self.file_dock.setObjectName("Files")
        fileListWidget = QtWidgets.QWidget()
        fileListWidget.setLayout(fileListLayout)
        self.file_dock.setWidget(fileListWidget)

        self.zoomWidget = ZoomWidget()
        self.setAcceptDrops(True)

        self.canvas = self.labelList.canvas = Canvas(
            epsilon=self._config["epsilon"],
            double_click=self._config["canvas"]["double_click"],
            num_backups=self._config["canvas"]["num_backups"],
            crosshair=self._config["canvas"]["crosshair"],
        )
        self.canvas.zoomRequest.connect(self.zoomRequest)
        self.canvas.mouseMoved.connect(
            lambda pos: self.status(f"Mouse is at: x={pos.x()}, y={pos.y()}")
        )

        scrollArea = QtWidgets.QScrollArea()
        scrollArea.setWidget(self.canvas)
        scrollArea.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scrollArea.verticalScrollBar(),
            Qt.Horizontal: scrollArea.horizontalScrollBar(),
        }
        self.canvas.scrollRequest.connect(self.scrollRequest)

        self.canvas.on_right_click = self._undo_last_action  # XCcv：右键撤销
        self.canvas.on_exit_drawing = self._exit_drawing_mode  # XCcv：退出标注走标准切换
        self.canvas.newShape.connect(self.newShape)
        self.canvas.shapeMoved.connect(self.setDirty)
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)
        # 批注修复：AI 后台加载/状态提示（原同步加载卡死 UI 数秒）
        self.canvas.aiModelReady.connect(
            lambda name: self._on_ai_model_ready(name))
        self.canvas.aiModelFailed.connect(
            lambda msg: self.statusBar().showMessage(
                f"AI 模型加载失败：{msg}", 10000))
        self.canvas.aiStatus.connect(
            lambda msg: self.statusBar().showMessage(msg, 5000))

        self.setCentralWidget(scrollArea)

        features = QtWidgets.QDockWidget.DockWidgetFeatures()
        for dock in ["flag_dock", "label_dock", "shape_dock", "file_dock"]:
            if self._config[dock]["closable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetClosable
            if self._config[dock]["floatable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetFloatable
            if self._config[dock]["movable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetMovable
            getattr(self, dock).setFeatures(features)
            if self._config[dock]["show"] is False:
                getattr(self, dock).setVisible(False)

        self.addDockWidget(Qt.RightDockWidgetArea, self.flag_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.label_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.shape_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.file_dock)

        # Actions
        action = functools.partial(utils.newAction, self)
        shortcuts = self._config["shortcuts"]
        quit = action(
            self.tr("&Quit"),
            self.close,
            shortcuts["quit"],
            "quit",
            self.tr("Quit application"),
        )
        open_ = action(
            self.tr("&Open\n"),
            self.openFile,
            shortcuts["open"],
            "open",
            self.tr("Open image or label file"),
        )
        opendir = action(
            self.tr("Open Dir"),
            self.openDirDialog,
            shortcuts["open_dir"],
            "open",
            self.tr("Open Dir"),
        )
        archiveToClass = action(
            "归类到类别",
            self.archiveToClass,
            "Ctrl+Shift+C",
            "open",
            "把当前图片（含同名 json）移动到类别子文件夹",
        )
        openNextImg = action(
            self.tr("&Next Image"),
            self.openNextImg,
            shortcuts["open_next"],
            "next",
            self.tr("Open next (hold Ctl+Shift to copy labels)"),
            enabled=False,
        )
        openPrevImg = action(
            self.tr("&Prev Image"),
            self.openPrevImg,
            shortcuts["open_prev"],
            "prev",
            self.tr("Open prev (hold Ctl+Shift to copy labels)"),
            enabled=False,
        )
        save = action(
            self.tr("&Save\n"),
            self.saveFile,
            shortcuts["save"],
            "save",
            self.tr("Save labels to file"),
            enabled=False,
        )
        saveAs = action(
            self.tr("&Save As"),
            self.saveFileAs,
            shortcuts["save_as"],
            "save-as",
            self.tr("Save labels to a different file"),
            enabled=False,
        )

        deleteFile = action(
            self.tr("&Delete File"),
            self.deleteFile,
            shortcuts["delete_file"],
            "delete",
            self.tr("Delete current label file"),
            enabled=False,
        )

        changeOutputDir = action(
            self.tr("&Change Output Dir"),
            slot=self.changeOutputDirDialog,
            shortcut=shortcuts["save_to"],
            icon="open",
            tip=self.tr("Change where annotations are loaded/saved"),
        )

        saveAuto = action(
            text=self.tr("Save &Automatically"),
            slot=lambda x: self.actions.saveAuto.setChecked(x),
            icon="save",
            tip=self.tr("Save automatically"),
            checkable=True,
            enabled=True,
        )
        saveAuto.setChecked(self._config["auto_save"])

        saveWithImageData = action(
            text=self.tr("Save With Image Data"),
            slot=self.enableSaveImageWithData,
            tip=self.tr("Save image data in label file"),
            checkable=True,
            checked=self._config["store_data"],
        )

        close = action(
            self.tr("&Close"),
            self.closeFile,
            shortcuts["close"],
            "close",
            self.tr("Close current file"),
        )

        toggle_keep_prev_mode = action(
            self.tr("Keep Previous Annotation"),
            self.toggleKeepPrevMode,
            shortcuts["toggle_keep_prev_mode"],
            None,
            self.tr('Toggle "keep previous annotation" mode'),
            checkable=True,
        )
        toggle_keep_prev_mode.setChecked(self._config["keep_prev"])

        # XCcv V0.6 T5-6：标签文字显隐（C）与灰度显示（G）
        showLabelTexts = action(
            self.tr("Show Label Texts"),
            self.toggleLabelTexts,
            shortcuts.get("show_label_texts", "C"),
            None,
            self.tr("在标注目标左上角显示/隐藏类别文字"),
            checkable=True,
        )
        showLabelTexts.setChecked(True)
        grayscaleDisplay = action(
            self.tr("Grayscale Display"),
            self.toggleGrayscale,
            shortcuts.get("grayscale_display", "G"),
            None,
            self.tr("以灰度方式显示当前图片（不改标注）"),
            checkable=True,
        )
        createMode = action(
            self.tr("Create Polygons"),

            lambda: self.toggleDrawMode(False, createMode="polygon"),
            shortcuts["create_polygon"],
            "objects",
            self.tr("Start drawing polygons"),
            enabled=False,
        )
        createRectangleMode = action(
            self.tr("Create Rectangle"),
            lambda: self.toggleDrawMode(False, createMode="rectangle"),
            shortcuts["create_rectangle"],
            "objects",
            self.tr("Start drawing rectangles"),
            enabled=False,
        )
        createCircleMode = action(
            self.tr("Create Circle"),
            lambda: self.toggleDrawMode(False, createMode="circle"),
            shortcuts["create_circle"],
            "objects",
            self.tr("Start drawing circles"),
            enabled=False,
        )
        createLineMode = action(
            self.tr("Create Line"),
            lambda: self.toggleDrawMode(False, createMode="line"),
            shortcuts["create_line"],
            "objects",
            self.tr("Start drawing lines"),
            enabled=False,
        )
        createPointMode = action(
            self.tr("Create Point"),
            lambda: self.toggleDrawMode(False, createMode="point"),
            shortcuts["create_point"],
            "objects",
            self.tr("Start drawing points"),
            enabled=False,
        )
        createLineStripMode = action(
            self.tr("Create LineStrip"),
            lambda: self.toggleDrawMode(False, createMode="linestrip"),
            shortcuts["create_linestrip"],
            "objects",
            self.tr("Start drawing linestrip. Ctrl+LeftClick ends creation."),
            enabled=False,
        )
        # XCcv V0.6 T5-3：AI 智能多边形（SAM 点提示实时提取）
        createAiPolygonMode = action(
            self.tr("创建智能多边形"),
            lambda: self.toggleDrawMode(False, createMode="ai_polygon"),
            shortcuts["create_ai_polygon"],
            "objects",
            self.tr(
                "AI 智能多边形：左键点目标=并入，Shift+左键=挖除，"
                "右键撤销上一锚点，再次右键/Esc 退出，Ctrl+左键或 Enter 完成"
            ),
            enabled=False,
        )
        # XCcv V0.6 T5-5：画笔标注
        createBrushMode = action(
            self.tr("创建画笔"),
            lambda: self.toggleDrawMode(False, createMode="brush"),
            shortcuts["create_brush"],
            "objects",
            self.tr(
                "画笔标注：左键拖动涂画，松开自动转多边形；"
                "滚轮或 [ ] 调节粗细，右键/Esc 取消本笔"
            ),
            enabled=False,
        )
        editMode = action(
            self.tr("Edit Polygons"),
            self.setEditMode,
            shortcuts["edit_polygon"],
            "edit",
            self.tr("Move and edit the selected polygons"),
            enabled=False,
        )

        delete = action(
            self.tr("Delete Polygons"),
            self.deleteSelectedShape,
            shortcuts["delete_polygon"],
            "cancel",
            self.tr("Delete the selected polygons"),
            enabled=False,
        )
        duplicate = action(
            self.tr("Duplicate Polygons"),
            self.duplicateSelectedShape,
            shortcuts["duplicate_polygon"],
            "copy",
            self.tr("Create a duplicate of the selected polygons"),
            enabled=False,
        )
        copy = action(
            self.tr("Copy Polygons"),
            self.copySelectedShape,
            shortcuts["copy_polygon"],
            "copy_clipboard",
            self.tr("Copy selected polygons to clipboard"),
            enabled=False,
        )
        paste = action(
            self.tr("Paste Polygons"),
            self.pasteSelectedShape,
            shortcuts["paste_polygon"],
            "paste",
            self.tr("Paste copied polygons"),
            enabled=False,
        )
        undoLastPoint = action(
            self.tr("Undo last point"),
            self.canvas.undoLastPoint,
            shortcuts["undo_last_point"],
            "undo",
            self.tr("Undo last drawn point"),
            enabled=False,
        )
        removePoint = action(
            text=self.tr("Remove Selected Point"),
            slot=self.removeSelectedPoint,
            shortcut=shortcuts["remove_selected_point"],
            icon="edit",
            tip=self.tr("Remove selected point from polygon"),
            enabled=False,
        )

        undo = action(
            self.tr("Undo\n"),
            self.undoShapeEdit,
            shortcuts["undo"],
            "undo",
            self.tr("Undo last add and edit of shape"),
            enabled=False,
        )

        hideAll = action(
            self.tr("&Hide\nPolygons"),
            functools.partial(self.togglePolygons, False),
            shortcuts["hide_all_polygons"],
            icon="eye",
            tip=self.tr("Hide all polygons"),
            enabled=False,
        )
        showAll = action(
            self.tr("&Show\nPolygons"),
            functools.partial(self.togglePolygons, True),
            shortcuts["show_all_polygons"],
            icon="eye",
            tip=self.tr("Show all polygons"),
            enabled=False,
        )
        toggleAll = action(
            self.tr("&Toggle\nPolygons"),
            functools.partial(self.togglePolygons, None),
            shortcuts["toggle_all_polygons"],
            icon="eye",
            tip=self.tr("Toggle all polygons"),
            enabled=False,
        )

        help = action(
            self.tr("&Tutorial"),
            self.tutorial,
            icon="help",
            tip=self.tr("Show tutorial page"),
        )

        zoom = QtWidgets.QWidgetAction(self)
        zoomBoxLayout = QtWidgets.QVBoxLayout()
        zoomLabel = QtWidgets.QLabel(self.tr("Zoom"))
        zoomLabel.setAlignment(Qt.AlignCenter)
        zoomBoxLayout.addWidget(zoomLabel)
        zoomBoxLayout.addWidget(self.zoomWidget)
        zoom.setDefaultWidget(QtWidgets.QWidget())
        zoom.defaultWidget().setLayout(zoomBoxLayout)
        self.zoomWidget.setWhatsThis(
            str(
                self.tr(
                    "Zoom in or out of the image. Also accessible with "
                    "{} and {} from the canvas."
                )
            ).format(
                utils.fmtShortcut(
                    "{},{}".format(shortcuts["zoom_in"], shortcuts["zoom_out"])
                ),
                utils.fmtShortcut(self.tr("Ctrl+Wheel")),
            )
        )
        self.zoomWidget.setEnabled(False)

        zoomIn = action(
            self.tr("Zoom &In"),
            functools.partial(self.addZoom, 1.1),
            shortcuts["zoom_in"],
            "zoom-in",
            self.tr("Increase zoom level"),
            enabled=False,
        )
        zoomOut = action(
            self.tr("&Zoom Out"),
            functools.partial(self.addZoom, 0.9),
            shortcuts["zoom_out"],
            "zoom-out",
            self.tr("Decrease zoom level"),
            enabled=False,
        )
        zoomOrg = action(
            self.tr("&Original size"),
            functools.partial(self.setZoom, 100),
            shortcuts["zoom_to_original"],
            "zoom",
            self.tr("Zoom to original size"),
            enabled=False,
        )
        keepPrevScale = action(
            self.tr("切换图片时保持缩放"),
            self.enableKeepPrevScale,
            tip=self.tr("开启：切回看过的图恢复它上次的缩放/滚动位置；"
                        "关闭：每张图自动适应窗口"),
            checkable=True,
            checked=self._config["keep_prev_scale"],
            enabled=True,
        )
        fitWindow = action(
            self.tr("&Fit Window"),
            self.setFitWindow,
            shortcuts["fit_window"],
            "fit-window",
            self.tr("Zoom follows window size"),
            checkable=True,
            enabled=False,
        )
        fitWidth = action(
            self.tr("Fit &Width"),
            self.setFitWidth,
            shortcuts["fit_width"],
            "fit-width",
            self.tr("Zoom follows window width"),
            checkable=True,
            enabled=False,
        )
        brightnessContrast = action(
            self.tr("&Brightness Contrast"),
            self.brightnessContrast,
            None,
            "color",
            self.tr("Adjust brightness and contrast"),
            enabled=False,
        )
        # Group zoom controls into a list for easier toggling.
        zoomActions = (
            self.zoomWidget,
            zoomIn,
            zoomOut,
            zoomOrg,
            fitWindow,
            fitWidth,
        )
        self.zoomMode = self.FIT_WINDOW
        fitWindow.setChecked(Qt.Checked)
        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        edit = action(
            self.tr("&Edit Label"),
            self._edit_label,
            shortcuts["edit_label"],
            "edit",
            self.tr("Modify the label of the selected polygon"),
            enabled=False,
        )

        fill_drawing = action(
            self.tr("Fill Drawing Polygon"),
            self.canvas.setFillDrawing,
            None,
            "color",
            self.tr("Fill polygon while drawing"),
            checkable=True,
            enabled=True,
        )
        if self._config["canvas"]["fill_drawing"]:
            fill_drawing.trigger()

        # Label list context menu.
        labelMenu = QtWidgets.QMenu()
        utils.addActions(labelMenu, (edit, delete))
        self.labelList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.labelList.customContextMenuRequested.connect(self.popLabelListMenu)

        # Store actions for further handling.
        self.actions = utils.struct(
            saveAuto=saveAuto,
            saveWithImageData=saveWithImageData,
            changeOutputDir=changeOutputDir,
            save=save,
            saveAs=saveAs,
            open=open_,
            close=close,
            deleteFile=deleteFile,
            toggleKeepPrevMode=toggle_keep_prev_mode,
            delete=delete,
            edit=edit,
            duplicate=duplicate,
            copy=copy,
            paste=paste,
            undoLastPoint=undoLastPoint,
            undo=undo,
            removePoint=removePoint,
            createMode=createMode,
            editMode=editMode,
            createRectangleMode=createRectangleMode,
            createCircleMode=createCircleMode,
            createLineMode=createLineMode,
            createPointMode=createPointMode,
            createLineStripMode=createLineStripMode,
            createAiPolygonMode=createAiPolygonMode,
            createBrushMode=createBrushMode,
            showLabelTexts=showLabelTexts,
            grayscaleDisplay=grayscaleDisplay,
            zoom=zoom,
            zoomIn=zoomIn,
            zoomOut=zoomOut,
            zoomOrg=zoomOrg,
            keepPrevScale=keepPrevScale,
            fitWindow=fitWindow,
            fitWidth=fitWidth,
            brightnessContrast=brightnessContrast,
            zoomActions=zoomActions,
            openNextImg=openNextImg,
            openPrevImg=openPrevImg,
            fileMenuActions=(open_, opendir, archiveToClass, save, saveAs, close, quit),
            tool=(),
            # XXX: need to add some actions here to activate the shortcut
            editMenu=(
                edit,
                duplicate,
                copy,
                paste,
                delete,
                None,
                undo,
                undoLastPoint,
                None,
                removePoint,
                None,
                toggle_keep_prev_mode,
            ),
            # menu shown at right click
            menu=(
                createMode,
                createRectangleMode,
                createCircleMode,
                createLineMode,
                createPointMode,
                createLineStripMode,
                createAiPolygonMode,
                createBrushMode,
                editMode,
                edit,
                duplicate,
                copy,
                paste,
                delete,
                undo,
                undoLastPoint,
                removePoint,
            ),
            onLoadActive=(
                close,
                createMode,
                createRectangleMode,
                createCircleMode,
                createLineMode,
                createPointMode,
                createLineStripMode,
                createAiPolygonMode,
                createBrushMode,
                editMode,
                brightnessContrast,
            ),
            onShapesPresent=(saveAs, hideAll, showAll, toggleAll),
        )

        self.canvas.vertexSelected.connect(self.actions.removePoint.setEnabled)

        self.menus = utils.struct(
            file=self.menu(self.tr("&File")),
            edit=self.menu(self.tr("&Edit")),
            view=self.menu(self.tr("&View")),
            help=self.menu(self.tr("&Help")),
            recentFiles=QtWidgets.QMenu(self.tr("Open &Recent")),
            labelList=labelMenu,
        )

        # XCcv：语言设置菜单（切换中英文，重启后生效）
        settings_menu = self.menu("Language 语言")
        lang_group = QtWidgets.QActionGroup(self)
        for lang_id in ("zh_CN", "en"):
            lang_act = action(
                language_name(lang_id),
                lambda _, lid=lang_id: self.setLanguage(lid),
                None,
                None,
                "Takes effect after restart. 重启后生效",
                checkable=True,
            )
            lang_act.setChecked(current_language() == lang_id)
            lang_group.addAction(lang_act)
            settings_menu.addAction(lang_act)

        utils.addActions(
            self.menus.file,
            (
                open_,
                openNextImg,
                openPrevImg,
                opendir,
                self.menus.recentFiles,
                save,
                saveAs,
                saveAuto,
                changeOutputDir,
                saveWithImageData,
                close,
                deleteFile,
                None,
                quit,
            ),
        )
        utils.addActions(self.menus.help, (help,))
        utils.addActions(
            self.menus.view,
            (
                self.flag_dock.toggleViewAction(),
                self.label_dock.toggleViewAction(),
                self.shape_dock.toggleViewAction(),
                self.file_dock.toggleViewAction(),
                None,
                fill_drawing,
                None,
                hideAll,
                showAll,
                toggleAll,
                showLabelTexts,
                grayscaleDisplay,
                None,
                zoomIn,
                zoomOut,
                zoomOrg,
                keepPrevScale,
                None,
                fitWindow,
                fitWidth,
                None,
                brightnessContrast,
            ),
        )

        self.menus.file.aboutToShow.connect(self.updateFileMenu)

        # Custom context menu for the canvas widget:
        utils.addActions(self.canvas.menus[0], self.actions.menu)
        utils.addActions(
            self.canvas.menus[1],
            (
                action("&Copy here", self.copyShape),
                action("&Move here", self.moveShape),
            ),
        )

        self._aiModelAction = self._build_ai_model_action()
        self.tools = self.toolbar("Tools")
        self.actions.tool = (
            open_,
            opendir,
            openPrevImg,
            openNextImg,
            save,
            deleteFile,
            None,
            createMode,
            editMode,
            duplicate,
            delete,
            undo,
            brightnessContrast,
            None,
            fitWindow,
            zoom,
            self._aiModelAction,
            None,
        )

        self.statusBar().showMessage(str(self.tr("%s started.")) % __appname__)
        self.statusBar().show()

        if output_file is not None and self._config["auto_save"]:
            logger.warning(
                "If `auto_save` argument is True, `output_file` argument "
                "is ignored and output filename is automatically "
                "set as IMAGE_BASENAME.json."
            )
        self.output_file = output_file
        self.output_dir = output_dir

        # Application state.
        self.image = QtGui.QImage()
        self.imagePath = None
        self.recentFiles = []
        self.maxRecent = 7
        self.otherData = None
        self.zoom_level = 100
        self.fit_window = False
        self.zoom_values = {}  # key=filename, value=(zoom_mode, zoom_value)
        self.brightnessContrast_values = {}
        self.scroll_values = {
            Qt.Horizontal: {},
            Qt.Vertical: {},
        }  # key=filename, value=scroll_value

        if filename is not None and osp.isdir(filename):
            self.importDirImages(filename, load=False)
        else:
            self.filename = filename

        if config["file_search"]:
            self.fileSearch.setText(config["file_search"])
            self.fileSearchChanged()

        # XXX: Could be completely declarative.
        # Restore application settings.
        self.settings = QtCore.QSettings("labelme", "labelme")
        self.recentFiles = self.settings.value("recentFiles", []) or []
        size = self.settings.value("window/size", QtCore.QSize(600, 500))
        position = self.settings.value("window/position", QtCore.QPoint(0, 0))
        state = self.settings.value("window/state", QtCore.QByteArray())
        self.resize(size)
        self.move(position)
        # or simply:
        # self.restoreGeometry(settings['window/geometry']
        self.restoreState(state)

        # Populate the File menu dynamically.
        self.updateFileMenu()
        # Since loading the file may take some time,
        # make sure it runs in the background.
        if self.filename is not None:
            self.queueEvent(functools.partial(self.loadFile, self.filename))

        # Callbacks:
        self.zoomWidget.valueChanged.connect(self.paintCanvas)

        self.populateModeActions()

        # self.firstStart = True
        # if self.firstStart:
        #    QWhatsThis.enterWhatsThisMode()

    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            utils.addActions(menu, actions)
        return menu

    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName("%sToolBar" % title)
        # toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        if actions:
            utils.addActions(toolbar, actions)
        self.addToolBar(Qt.TopToolBarArea, toolbar)
        return toolbar

    # Support Functions

    def noShapes(self):
        return not len(self.labelList)

    def populateModeActions(self):
        tool, menu = self.actions.tool, self.actions.menu
        self.tools.clear()
        utils.addActions(self.tools, tool)
        self.canvas.menus[0].clear()
        utils.addActions(self.canvas.menus[0], menu)
        self.menus.edit.clear()
        actions = (
            self.actions.createMode,
            self.actions.createRectangleMode,
            self.actions.createCircleMode,
            self.actions.createLineMode,
            self.actions.createPointMode,
            self.actions.createLineStripMode,
            # V0.6 修复循环 4（用户批注）：S=智能多边形 / B=画笔——必须挂进
            # 编辑菜单 QAction 才归属窗口，快捷键才会激活（708 行 XXX 同坑）
            self.actions.createAiPolygonMode,
            self.actions.createBrushMode,
            self.actions.editMode,
        )
        utils.addActions(self.menus.edit, actions + self.actions.editMenu)
        # 批注：AI 标注参数入口（点数上限/容差/悬停预览，用户可调）
        from PyQt5.QtWidgets import QAction as _QA

        self._act_ai_params = _QA(self.tr("AI 标注参数…"), self)
        self._act_ai_params.triggered.connect(self._ai_params_dialog)
        self.menus.edit.addSeparator()
        self.menus.edit.addAction(self._act_ai_params)
        # 批注第二批 F2/F7/F5：粗细键 / 智能标注 L / 批量改名
        self._act_brush_wider = _QA(self.tr("画笔加粗（+/=）"), self)
        self._act_brush_wider.setShortcut(QtGui.QKeySequence("+"))
        self._act_brush_wider.triggered.connect(
            lambda: self.canvas.setBrushWidth(
                self.canvas._brush_width + 4))
        self._act_brush_narrower = _QA(self.tr("画笔变细（-）"), self)
        self._act_brush_narrower.setShortcut(QtGui.QKeySequence("-"))
        self._act_brush_narrower.triggered.connect(
            lambda: self.canvas.setBrushWidth(
                self.canvas._brush_width - 4))
        self.menus.edit.addAction(self._act_brush_wider)
        self.menus.edit.addAction(self._act_brush_narrower)
        self.menus.edit.addSeparator()
        self._act_smart = _QA(self.tr("智能标注当前图（L）"), self)
        self._act_smart.setShortcut(QtGui.QKeySequence("L"))
        self._act_smart.triggered.connect(self.smartAnnotateSingle)
        self.menus.edit.addAction(self._act_smart)
        self._act_rename = _QA(self.tr("按标签批量改名…"), self)
        self._act_rename.triggered.connect(self.batchRenameLabels)
        self.menus.edit.addAction(self._act_rename)

    def setDirty(self):
        # Even if we autosave the file, we keep the ability to undo
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)

        if self._config["auto_save"] or self.actions.saveAuto.isChecked():
            label_file = osp.splitext(self.imagePath)[0] + ".json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            self.saveLabels(label_file)
            return
        self.dirty = True
        self.actions.save.setEnabled(True)
        title = __appname__
        if self.filename is not None:
            title = "{} - {}*".format(title, self.filename)
        self.setWindowTitle(title)

    def setClean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)
        self.actions.createMode.setEnabled(True)
        self.actions.createRectangleMode.setEnabled(True)
        self.actions.createCircleMode.setEnabled(True)
        self.actions.createLineMode.setEnabled(True)
        self.actions.createPointMode.setEnabled(True)
        self.actions.createLineStripMode.setEnabled(True)
        self.actions.createAiPolygonMode.setEnabled(True)
        self.actions.createBrushMode.setEnabled(True)
        title = __appname__
        if self.filename is not None:
            title = "{} - {}".format(title, self.filename)
        self.setWindowTitle(title)

        if self.hasLabelFile():
            self.actions.deleteFile.setEnabled(True)
        else:
            self.actions.deleteFile.setEnabled(False)

    def toggleActions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.actions.zoomActions:
            z.setEnabled(value)
        for action in self.actions.onLoadActive:
            action.setEnabled(value)

    def queueEvent(self, function):
        QtCore.QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)

    def resetState(self):
        self.labelList.clear()
        self.filename = None
        self.imagePath = None
        self.imageData = None
        self.labelFile = None
        self.otherData = None
        self.canvas.resetState()

    def currentItem(self):
        items = self.labelList.selectedItems()
        if items:
            return items[0]
        return None

    def addRecentFile(self, filename):
        if filename in self.recentFiles:
            self.recentFiles.remove(filename)
        elif len(self.recentFiles) >= self.maxRecent:
            self.recentFiles.pop()
        self.recentFiles.insert(0, filename)

    # Callbacks

    def undoShapeEdit(self):
        self.canvas.restoreShape()
        self.labelList.clear()
        self.loadShapes(self.canvas.shapes)
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)

    def tutorial(self):
        url = "https://github.com/labelmeai/labelme/tree/main/examples/tutorial"  # NOQA
        webbrowser.open(url)

    def toggleDrawingSensitive(self, drawing=True):
        """Toggle drawing sensitive.

        In the middle of drawing, toggling between modes should be disabled.
        """
        self.actions.editMode.setEnabled(not drawing)
        self.actions.undoLastPoint.setEnabled(drawing)
        self.actions.undo.setEnabled(not drawing)
        self.actions.delete.setEnabled(not drawing)

    def toggleDrawMode(self, edit=True, createMode="polygon"):
        draw_actions = {
            "polygon": self.actions.createMode,
            "rectangle": self.actions.createRectangleMode,
            "circle": self.actions.createCircleMode,
            "point": self.actions.createPointMode,
            "line": self.actions.createLineMode,
            "linestrip": self.actions.createLineStripMode,
            "ai_polygon": self.actions.createAiPolygonMode,
            "brush": self.actions.createBrushMode,
        }

        self.canvas.setEditing(edit)
        self.canvas.createMode = createMode
        if not edit and createMode == "ai_polygon":
            self._init_ai_model()   # 进入智能多边形前确保 SAM 就绪（T5-3）
        if edit:
            for draw_action in draw_actions.values():
                draw_action.setEnabled(True)
        else:
            for draw_mode, draw_action in draw_actions.items():
                draw_action.setEnabled(createMode != draw_mode)
        self.actions.editMode.setEnabled(not edit)

    def setEditMode(self):
        self.toggleDrawMode(True)

    # ---- XCcv V0.6 T5-6：交互精选 ----
    def toggleLabelTexts(self):
        self.canvas.setLabelsVisible(self.actions.showLabelTexts.isChecked())

    def toggleGrayscale(self):
        """灰度显示开关：仅改显示，不改图片与标注。"""
        pm = self.canvas.pixmap
        if pm is None or pm.isNull():
            return
        if self.actions.grayscaleDisplay.isChecked():
            self._color_pixmap = pm
            img = pm.toImage().convertToFormat(QtGui.QImage.Format_Grayscale8)
            self.canvas.loadPixmap(QtGui.QPixmap.fromImage(img),
                                   clear_shapes=False)
        else:
            if getattr(self, "_color_pixmap", None) is not None:
                self.canvas.loadPixmap(self._color_pixmap, clear_shapes=False)

    def _apply_grayscale_if_enabled(self):
        if getattr(self, "actions", None) is not None \
                and self.actions.grayscaleDisplay.isChecked() \
                and self.canvas.pixmap is not None:
            self.toggleGrayscale()

    def _file_list_menu(self, pos):
        """文件列表右键：打开所在文件夹（T5-6）。"""
        import subprocess

        item = self.fileListWidget.itemAt(pos)
        if item is None:
            return
        filename = str(item.data(0, QtCore.Qt.UserRole) or "")
        menu = QtWidgets.QMenu(self)
        act_open = menu.addAction(self.tr("Open Containing Folder"))
        chosen = menu.exec_(self.fileListWidget.mapToGlobal(pos))
        if chosen is act_open and filename:
            folder = osp.dirname(filename)
            if osp.isdir(folder):
                if hasattr(os, "startfile"):
                    os.startfile(folder)  # noqa: S606  资源管理器
                else:
                    subprocess.Popen(["xdg-open", folder])

    def _init_ai_model(self):
        """T5-3：智能多边形进入前初始化 SAM（权重缺失=动作提示，不进入）。"""
        if getattr(self.canvas, "_sam", None) is not None:
            return
        try:
            from xccv.annotation import ai_assist

            ready = [m for m in ai_assist.available_models() if m["available"]]
            if not ready:
                msg = (
                    "AI 智能多边形不可用：缺少权重文件。请将 sam2.1_t.pt 或 "
                    "mobile_sam.pt 放到 " + str(ai_assist.weights_dir()))
                self.statusBar().showMessage(msg, 10000)
                self.setEditMode()
                return
            name = self._selected_ai_model() \
                or (self._config.get("ai") or {}).get("default") \
                or ready[0]["name"]
            if name not in [m["name"] for m in ready]:
                name = ready[0]["name"]
            # 批注修复：模型+嵌入后台加载（原同步初始化卡死 UI 数秒）
            self.canvas.initializeAiModelAsync(name)
            self.statusBar().showMessage(
                f"AI 智能多边形后台加载中（{name}）：就绪后 左键并入 / "
                "Shift+左键挖除 / 右键撤销 / Ctrl+左键完成", 10000)
        except Exception as e:  # noqa: BLE001
            self.statusBar().showMessage(f"AI 模型初始化失败：{e}", 10000)
            self.setEditMode()

    def smartAnnotateSingle(self):
        """批注第二批 F7：L=用平台模型标注当前图（批量链路的单图版）。"""
        import threading as _th

        if self.imagePath is None or self.canvas.pixmap is None:
            self.statusBar().showMessage("智能标注：请先打开图片", 5000)
            return
        from xccv.ui_widgets import load_prefs, save_prefs

        prefs = load_prefs()
        ai = self._config.get("ai") or {}
        weights = ai.get("batch_model") or prefs.get("ai_batch_model") or ""
        if not weights or not osp.exists(weights):
            weights, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "选择模型权重（.pt）", "",
                "模型权重 (*.pt)")
            if not weights:
                return
            save_prefs({"ai_batch_model": weights})   # 记住，下次免选
        self._config["ai"] = {"batch_model": weights}
        self.statusBar().showMessage("智能标注推理中…")
        import cv2 as _cv2

        arr = utils.img_qt_to_arr(self.canvas.pixmap.toImage())
        arr = _cv2.cvtColor(arr, _cv2.COLOR_RGB2BGR)

        class _Sig(QtCore.QObject):
            done = QtCore.pyqtSignal(object)

        sig = _Sig()
        sig.done.connect(self._on_smart_annotate_done)
        self._smart_annotate_sig = sig  # 防局部变量被回收导致信号丢失

        def _work():
            try:
                from xccv.engine.models.registry import load_model

                model = load_model(weights, device="auto")
                task = getattr(getattr(model, "task", None), "value", "")
                if task not in ("detection", "segmentation"):
                    raise ValueError(
                        f"需要检测/分割模型，所选模型任务为「{task}」")
                pred = model.predict(arr)
                from xccv.annotation import auto_annotate

                shapes = auto_annotate.prediction_to_shapes(
                    pred, list(model.classes or []))
                sig.done.emit(shapes)
            except Exception as e:  # noqa: BLE001
                sig.done.emit(e)

        _th.Thread(target=_work, daemon=True).start()

    def _on_smart_annotate_done(self, result):
        if isinstance(result, Exception):
            self.statusBar().showMessage(
                f"智能标注失败：{result}", 8000)
            return
        from labelme.shape import Shape

        added = 0
        for d in result:
            sh = Shape(shape_type=d["shape_type"])
            for x, y in d["points"]:
                sh.addPoint(QtCore.QPointF(float(x), float(y)))
            if d["shape_type"] != "points":
                sh.close()
            sh.label = d["label"]
            if d.get("score") is not None:
                try:
                    sh.score = d["score"]
                except Exception:  # noqa: BLE001
                    pass
            self.canvas.shapes.append(sh)
            if self.uniqLabelList.findItemByLabel(sh.label) is None:
                _item = self.uniqLabelList.createItemFromLabel(sh.label)
                self.uniqLabelList.addItem(_item)
                self.uniqLabelList.setItemLabel(
                    _item, sh.label, self._get_rgb_by_label(sh.label))
            added += 1
        self.setDirty()
        self.canvas.update()
        self.statusBar().showMessage(
            f"智能标注完成：新增 {added} 个目标（追加模式）", 8000)

    def batchRenameLabels(self):
        """批注第二批 F5（降级版）：当前图按标签批量改名。"""
        from PyQt5.QtWidgets import (QComboBox, QDialog, QDialogButtonBox,
                                     QFormLayout, QLineEdit)

        labels = sorted({s.label for s in self.canvas.shapes if s.label})
        if not labels:
            self.statusBar().showMessage(
                "批量改名：当前图没有已标注目标", 5000)
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("按标签批量改名（当前图）")
        form = QFormLayout(dlg)
        cb_old = QComboBox()
        cb_old.addItems(labels)
        le_new = QLineEdit()
        form.addRow("原标签", cb_old)
        form.addRow("新标签", le_new)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if not dlg.exec_():
            return
        old, new = cb_old.currentText(), le_new.text().strip()
        if not new or new == old:
            return
        n = 0
        for sh in self.canvas.shapes:
            if sh.label == old:
                sh.label = new
                n += 1
        if n:
            self.setDirty()
            self.canvas.update()
            self._sync_classes_txt()
        self.statusBar().showMessage(
            f"批量改名完成：{old} → {new}（{n} 个）", 8000)

    def _on_ai_model_ready(self, name):
        self.canvas._apply_pending_model(name)
        want = self._selected_ai_model()
        if want and want != name:
            # 加载期间用户又换了模型：继续加载最新所选
            self.canvas.initializeAiModelAsync(want)
            return
        self.statusBar().showMessage(
            f"AI 智能多边形已就绪（{name}）：左键并入 / Shift+左键挖除 / "
            "右键撤销 / 双击或 Ctrl+左键完成", 8000)

    # ---- AI 模型选择与参数（第二批：用户可选模型、可调参数，持久化）----
    _AI_DEFAULTS = {
        "max_points": 100, "tolerance": 0.004, "hover": True,
        "hover_ms": 120, "first_ratio": 0.9, "largest_only": True,
        "zoom": True, "zoom_win": 256, "zoom_trigger": 0.05,
        "mask_threshold": 0.0, "pick": "auto", "device": "auto",
        "half": True,
    }

    def _ai_params(self):
        from xccv.ui_widgets import load_prefs

        p = dict(self._AI_DEFAULTS)
        p.update(load_prefs().get("ai_params") or {})
        return p

    def _apply_ai_params(self, p):
        c = self.canvas
        c._ai_max_points = int(p["max_points"])
        c._ai_tolerance = float(p["tolerance"])
        c._ai_hover_enabled = bool(p["hover"])
        c._ai_hover_interval = max(0.03, int(p["hover_ms"]) / 1000.0)
        c._ai_first_ratio = float(p["first_ratio"])
        c._ai_largest_only = bool(p["largest_only"])
        try:
            from xccv.annotation import ai_assist

            ai_assist.set_params(
                zoom=bool(p["zoom"]), zoom_win=int(p["zoom_win"]),
                zoom_trigger=float(p["zoom_trigger"]),
                mask_threshold=float(p["mask_threshold"]),
                pick=str(p["pick"]), device=str(p["device"]),
                half=bool(p["half"]))
        except Exception as e:  # noqa: BLE001
            logger.warning("AI 参数应用失败：{}", e)
        if hasattr(c, "_ai_invalidate"):
            c._ai_invalidate()

    def _build_ai_model_action(self):
        act = QtWidgets.QWidgetAction(self)
        box = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(box)
        lay.setContentsMargins(2, 0, 2, 0)
        lab = QtWidgets.QLabel(self.tr("AI 模型"))
        lab.setAlignment(Qt.AlignCenter)
        lay.addWidget(lab)
        self._aiModelCombo = QtWidgets.QComboBox()
        lay.addWidget(self._aiModelCombo)
        act.setDefaultWidget(box)
        self._fill_ai_models(self._aiModelCombo,
                             self._ai_params().get("model"))
        self._aiModelCombo.currentIndexChanged.connect(
            self._on_ai_model_changed)
        self._apply_ai_params(self._ai_params())
        return act

    def _fill_ai_models(self, cb, want=None):
        try:
            from xccv.annotation import ai_assist

            models = ai_assist.available_models()
        except Exception:  # noqa: BLE001
            models = []
        want = want or (self._config.get("ai") or {}).get("default")
        cb.blockSignals(True)
        cb.clear()
        sel = -1
        for m in models:
            cb.addItem(m["name"], m["name"])
            i = cb.count() - 1
            cb.setItemData(i, m["title"] if m["available"]
                           else m["message"], Qt.ToolTipRole)
            if not m["available"]:
                cb.model().item(i).setEnabled(False)
            elif sel < 0 or m["name"] == want:
                sel = i
        if sel >= 0:
            cb.setCurrentIndex(sel)
        cb.blockSignals(False)

    def _selected_ai_model(self):
        cb = getattr(self, "_aiModelCombo", None)
        return cb.currentData() if cb is not None else None

    def _on_ai_model_changed(self):
        from xccv.ui_widgets import save_prefs

        name = self._selected_ai_model()
        if not name:
            return
        p = self._ai_params()
        p["model"] = name
        save_prefs({"ai_params": p})
        if getattr(self.canvas, "_sam", None) is not None \
                or self.canvas.createMode in ("ai_polygon", "ai_mask"):
            self.canvas.initializeAiModelAsync(name)
            self.statusBar().showMessage(f"AI 模型切换中：{name}", 5000)

    def _ai_params_dialog(self):
        """AI 标注参数：模型 / 推理 / 提示 / 小目标 / 轮廓 / 预览。"""
        from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDialog,
                                     QDialogButtonBox, QDoubleSpinBox,
                                     QFormLayout, QGroupBox, QSpinBox,
                                     QVBoxLayout)

        p = self._ai_params()
        dlg = QDialog(self)
        dlg.setWindowTitle("AI 标注参数")
        root = QVBoxLayout(dlg)

        def group(title):
            g = QGroupBox(title)
            f = QFormLayout(g)
            root.addWidget(g)
            return f

        def combo(items, cur):
            c = QComboBox()
            for text, data in items:
                c.addItem(text, data)
            i = max(0, c.findData(cur))
            c.setCurrentIndex(i)
            return c

        f1 = group("模型")
        cb_model = QComboBox()
        self._fill_ai_models(cb_model, self._selected_ai_model())
        cb_dev = combo([("自动（GPU 优先）", "auto"), ("GPU", "cuda"),
                        ("CPU", "cpu")], p["device"])
        cb_half = QCheckBox("GPU 半精度（更快）")
        cb_half.setChecked(bool(p["half"]))
        f1.addRow("SAM 模型", cb_model)
        f1.addRow("推理设备", cb_dev)
        f1.addRow(cb_half)

        f2 = group("提示与候选")
        sp_ratio = QSpinBox()
        sp_ratio.setRange(50, 90)
        sp_ratio.setSingleStep(5)
        sp_ratio.setSuffix(" %")
        sp_ratio.setValue(int(round(float(p["first_ratio"]) * 100)))
        cb_pick = combo([("自动", "auto"), ("小目标优先", "anchor"),
                         ("置信度优先", "score")], p["pick"])
        sp_thr = QDoubleSpinBox()
        sp_thr.setRange(-3.0, 3.0)
        sp_thr.setSingleStep(0.25)
        sp_thr.setDecimals(2)
        sp_thr.setValue(float(p["mask_threshold"]))
        sp_thr.setToolTip("调高边界更紧，调低更松（默认 0）")
        f2.addRow("首锚权重", sp_ratio)
        f2.addRow("候选选择", cb_pick)
        f2.addRow("掩膜阈值", sp_thr)

        f3 = group("小目标")
        cb_zoom = QCheckBox("放大裁剪重解")
        cb_zoom.setChecked(bool(p["zoom"]))
        cb_win = combo([(str(v), v) for v in (128, 192, 256, 384, 512)],
                       int(p["zoom_win"]))
        sp_trig = QDoubleSpinBox()
        sp_trig.setRange(0.01, 0.5)
        sp_trig.setSingleStep(0.01)
        sp_trig.setDecimals(2)
        sp_trig.setValue(float(p["zoom_trigger"]))
        f3.addRow(cb_zoom)
        f3.addRow("裁剪窗边长", cb_win)
        f3.addRow("触发面积占比", sp_trig)

        f4 = group("轮廓与预览")
        sp_pts = QSpinBox()
        sp_pts.setRange(10, 500)
        sp_pts.setValue(int(p["max_points"]))
        sp_tol = QDoubleSpinBox()
        sp_tol.setRange(0.00005, 0.05)
        sp_tol.setDecimals(5)
        sp_tol.setSingleStep(0.0005)
        sp_tol.setValue(float(p["tolerance"]))
        cb_cc = QCheckBox("仅保留最大连通域")
        cb_cc.setChecked(bool(p["largest_only"]))
        cb_hover = QCheckBox("悬停实时预览")
        cb_hover.setChecked(bool(p["hover"]))
        sp_hms = QSpinBox()
        sp_hms.setRange(30, 1000)
        sp_hms.setSuffix(" ms")
        sp_hms.setValue(int(p["hover_ms"]))
        f4.addRow("多边形顶点上限", sp_pts)
        f4.addRow("轮廓简化容差", sp_tol)
        f4.addRow(cb_cc)
        f4.addRow(cb_hover)
        f4.addRow("悬停刷新间隔", sp_hms)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel
                              | QDialogButtonBox.RestoreDefaults)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)

        def _defaults():
            d = self._AI_DEFAULTS
            cb_dev.setCurrentIndex(max(0, cb_dev.findData(d["device"])))
            cb_half.setChecked(d["half"])
            sp_ratio.setValue(int(d["first_ratio"] * 100))
            cb_pick.setCurrentIndex(max(0, cb_pick.findData(d["pick"])))
            sp_thr.setValue(d["mask_threshold"])
            cb_zoom.setChecked(d["zoom"])
            cb_win.setCurrentIndex(max(0, cb_win.findData(d["zoom_win"])))
            sp_trig.setValue(d["zoom_trigger"])
            sp_pts.setValue(d["max_points"])
            sp_tol.setValue(d["tolerance"])
            cb_cc.setChecked(d["largest_only"])
            cb_hover.setChecked(d["hover"])
            sp_hms.setValue(d["hover_ms"])

        bb.button(QDialogButtonBox.RestoreDefaults).clicked.connect(_defaults)
        bb.button(QDialogButtonBox.Ok).setText("确定")
        bb.button(QDialogButtonBox.Cancel).setText("取消")
        bb.button(QDialogButtonBox.RestoreDefaults).setText("恢复默认")
        root.addWidget(bb)
        if not dlg.exec_():
            return
        from xccv.ui_widgets import save_prefs

        p.update({
            "device": cb_dev.currentData(), "half": cb_half.isChecked(),
            "first_ratio": sp_ratio.value() / 100.0,
            "pick": cb_pick.currentData(),
            "mask_threshold": sp_thr.value(), "zoom": cb_zoom.isChecked(),
            "zoom_win": int(cb_win.currentData()),
            "zoom_trigger": sp_trig.value(), "max_points": sp_pts.value(),
            "tolerance": sp_tol.value(), "largest_only": cb_cc.isChecked(),
            "hover": cb_hover.isChecked(), "hover_ms": sp_hms.value(),
        })
        model = cb_model.currentData()
        if model:
            p["model"] = model
        save_prefs({"ai_params": p})
        self._apply_ai_params(p)
        if model and model != self._selected_ai_model():
            i = self._aiModelCombo.findData(model)
            if i >= 0:
                self._aiModelCombo.setCurrentIndex(i)   # 触发切换加载
        self.statusBar().showMessage("AI 标注参数已更新", 5000)

    def updateFileMenu(self):
        current = self.filename

        def exists(filename):
            return osp.exists(str(filename))

        menu = self.menus.recentFiles
        menu.clear()
        files = [f for f in self.recentFiles if f != current and exists(f)]
        for i, f in enumerate(files):
            icon = utils.newIcon("labels")
            action = QtWidgets.QAction(
                icon, "&%d %s" % (i + 1, QtCore.QFileInfo(f).fileName()), self
            )
            action.triggered.connect(functools.partial(self.loadRecent, f))
            menu.addAction(action)

    def popLabelListMenu(self, point):
        self.menus.labelList.exec_(self.labelList.mapToGlobal(point))

    def validateLabel(self, label):
        # no validation
        if self._config["validate_label"] is None:
            return True

        for i in range(self.uniqLabelList.count()):
            label_i = self.uniqLabelList.item(i).data(Qt.UserRole)
            if self._config["validate_label"] in ["exact"]:
                if label_i == label:
                    return True
        return False

    def _edit_label(self, value=None):
        if not self.canvas.editing():
            return

        items = self.labelList.selectedItems()
        if not items:
            logger.warning("No label is selected, so cannot edit label.")
            return

        shape = items[0].shape()

        if len(items) == 1:
            edit_text = True
            edit_flags = True
            edit_group_id = True
            edit_description = True
        else:
            edit_text = all(item.shape().label == shape.label for item in items[1:])
            edit_flags = all(item.shape().flags == shape.flags for item in items[1:])
            edit_group_id = all(
                item.shape().group_id == shape.group_id for item in items[1:]
            )
            edit_description = all(
                item.shape().description == shape.description for item in items[1:]
            )

        if not edit_text:
            self.labelDialog.edit.setDisabled(True)
            self.labelDialog.labelList.setDisabled(True)
        if not edit_flags:
            for i in range(self.labelDialog.flagsLayout.count()):
                self.labelDialog.flagsLayout.itemAt(i).setDisabled(True)
        if not edit_group_id:
            self.labelDialog.edit_group_id.setDisabled(True)
        if not edit_description:
            self.labelDialog.editDescription.setDisabled(True)

        text, flags, group_id, description = self.labelDialog.popUp(
            text=shape.label if edit_text else "",
            flags=shape.flags if edit_flags else None,
            group_id=shape.group_id if edit_group_id else None,
            description=shape.description if edit_description else None,
        )

        if not edit_text:
            self.labelDialog.edit.setDisabled(False)
            self.labelDialog.labelList.setDisabled(False)
        if not edit_flags:
            for i in range(self.labelDialog.flagsLayout.count()):
                self.labelDialog.flagsLayout.itemAt(i).setDisabled(False)
        if not edit_group_id:
            self.labelDialog.edit_group_id.setDisabled(False)
        if not edit_description:
            self.labelDialog.editDescription.setDisabled(False)

        if text is None:
            assert flags is None
            assert group_id is None
            assert description is None
            return

        if not self.validateLabel(text):
            self.errorMessage(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            return

        self.canvas.storeShapes()
        for item in items:
            shape: Shape = item.shape()

            if edit_text:
                shape.label = text
            if edit_flags:
                shape.flags = flags
            if edit_group_id:
                shape.group_id = group_id
            if edit_description:
                shape.description = description

            self._update_shape_color(shape)
            if shape.group_id is None:
                item.setText(
                    '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                        html.escape(shape.label), *shape.fill_color.getRgb()[:3]
                    )
                )
            else:
                item.setText("{} ({})".format(shape.label, shape.group_id))
            self.setDirty()
            if self.uniqLabelList.findItemByLabel(shape.label) is None:
                item = self.uniqLabelList.createItemFromLabel(shape.label)
                self.uniqLabelList.addItem(item)
                rgb = self._get_rgb_by_label(shape.label)
                self.uniqLabelList.setItemLabel(item, shape.label, rgb)

    def fileSearchChanged(self):
        self.importDirImages(
            self.lastOpenDir,
            pattern=self.fileSearch.text(),
            load=False,
        )

    def fileSelectionChanged(self):
        items = self.fileListWidget.selectedItems()
        if not items:
            return
        filename = items[0].data(0, QtCore.Qt.UserRole)
        if not filename:
            return

        if not self.mayContinue():
            return
        self.loadFile(filename)

    # React to canvas signals.
    def shapeSelectionChanged(self, selected_shapes):
        self._noSelectionSlot = True
        for shape in self.canvas.selectedShapes:
            shape.selected = False
        self.labelList.clearSelection()
        self.canvas.selectedShapes = selected_shapes
        for shape in self.canvas.selectedShapes:
            shape.selected = True
            item = self.labelList.findItemByShape(shape)
            self.labelList.selectItem(item)
            self.labelList.scrollToItem(item)
        self._noSelectionSlot = False
        n_selected = len(selected_shapes)
        self.actions.delete.setEnabled(n_selected)
        self.actions.duplicate.setEnabled(n_selected)
        self.actions.copy.setEnabled(n_selected)
        self.actions.edit.setEnabled(n_selected)

    def addLabel(self, shape):
        if shape.group_id is None:
            text = shape.label
        else:
            text = "{} ({})".format(shape.label, shape.group_id)
        label_list_item = LabelListWidgetItem(text, shape)
        self.labelList.addItem(label_list_item)
        if self.uniqLabelList.findItemByLabel(shape.label) is None:
            item = self.uniqLabelList.createItemFromLabel(shape.label)
            self.uniqLabelList.addItem(item)
            rgb = self._get_rgb_by_label(shape.label)
            self.uniqLabelList.setItemLabel(item, shape.label, rgb)
        self.labelDialog.addLabelHistory(shape.label)
        for action in self.actions.onShapesPresent:
            action.setEnabled(True)

        self._update_shape_color(shape)
        label_list_item.setText(
            '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                html.escape(text), *shape.fill_color.getRgb()[:3]
            )
        )

    def _update_shape_color(self, shape):
        r, g, b = self._get_rgb_by_label(shape.label)
        shape.line_color = QtGui.QColor(r, g, b)
        shape.vertex_fill_color = QtGui.QColor(r, g, b)
        shape.hvertex_fill_color = QtGui.QColor(255, 255, 255)
        shape.fill_color = QtGui.QColor(r, g, b, 128)
        shape.select_line_color = QtGui.QColor(255, 255, 255)
        shape.select_fill_color = QtGui.QColor(r, g, b, 155)

    def _get_rgb_by_label(self, label):
        if self._config["shape_color"] == "auto":
            item = self.uniqLabelList.findItemByLabel(label)
            if item is None:
                item = self.uniqLabelList.createItemFromLabel(label)
                self.uniqLabelList.addItem(item)
                rgb = self._get_rgb_by_label(label)
                self.uniqLabelList.setItemLabel(item, label, rgb)
            label_id = self.uniqLabelList.indexFromItem(item).row() + 1
            label_id += self._config["shift_auto_shape_color"]
            return LABEL_COLORMAP[label_id % len(LABEL_COLORMAP)]
        elif (
            self._config["shape_color"] == "manual"
            and self._config["label_colors"]
            and label in self._config["label_colors"]
        ):
            return self._config["label_colors"][label]
        elif self._config["default_shape_color"]:
            return self._config["default_shape_color"]
        return (0, 255, 0)

    def remLabels(self, shapes):
        for shape in shapes:
            item = self.labelList.findItemByShape(shape)
            self.labelList.removeItem(item)

    def loadShapes(self, shapes, replace=True):
        self._noSelectionSlot = True
        for shape in shapes:
            self.addLabel(shape)
        self.labelList.clearSelection()
        self._noSelectionSlot = False
        self.canvas.loadShapes(shapes, replace=replace)

    def loadLabels(self, shapes):
        s = []
        for shape in shapes:
            label = shape["label"]
            points = shape["points"]
            shape_type = shape["shape_type"]
            flags: dict = shape["flags"] or {}
            description = shape.get("description", "")
            group_id = shape["group_id"]
            other_data = shape["other_data"]

            if not points:
                # skip point-empty shape
                continue

            shape = Shape(
                label=label,
                shape_type=shape_type,
                group_id=group_id,
                description=description,
                mask=shape["mask"],
            )
            for x, y in points:
                shape.addPoint(QtCore.QPointF(x, y))
            shape.close()

            default_flags = {}
            if self._config["label_flags"]:
                for pattern, keys in self._config["label_flags"].items():
                    if re.match(pattern, label):
                        for key in keys:
                            default_flags[key] = False
            shape.flags = default_flags
            shape.flags.update(flags)
            shape.other_data = other_data

            s.append(shape)
        self.loadShapes(s)

    def loadFlags(self, flags):
        self.flag_widget.clear()
        for key, flag in flags.items():
            item = QtWidgets.QListWidgetItem(key)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if flag else Qt.Unchecked)
            self.flag_widget.addItem(item)

    def saveLabels(self, filename):
        lf = LabelFile()

        def format_shape(s):
            data = s.other_data.copy()
            data.update(
                dict(
                    label=s.label,
                    points=[(p.x(), p.y()) for p in s.points],
                    group_id=s.group_id,
                    description=s.description,
                    shape_type=s.shape_type,
                    flags=s.flags,
                    mask=None
                    if s.mask is None
                    else utils.img_arr_to_b64(s.mask.astype(np.uint8)),
                )
            )
            return data

        shapes = [format_shape(item.shape()) for item in self.labelList]
        flags = {}
        for i in range(self.flag_widget.count()):
            item = self.flag_widget.item(i)
            key = item.text()
            flag = item.checkState() == Qt.Checked
            flags[key] = flag
        try:
            imagePath = osp.relpath(self.imagePath, osp.dirname(filename))
            imageData = self.imageData if self._config["store_data"] else None
            if osp.dirname(filename) and not osp.exists(osp.dirname(filename)):
                os.makedirs(osp.dirname(filename))
            lf.save(
                filename=filename,
                shapes=shapes,
                imagePath=imagePath,
                imageData=imageData,
                imageHeight=self.image.height(),
                imageWidth=self.image.width(),
                otherData=self.otherData,
                flags=flags,
            )
            self.labelFile = lf
            item = self._file_items.get(self.imagePath)
            if item is not None:
                item.setCheckState(0, Qt.Checked)
            # disable allows next and previous image to proceed
            # self.filename = filename
            self._sync_classes_txt()
            return True
        except LabelFileError as e:
            self.errorMessage(
                self.tr("Error saving label data"), self.tr("<b>%s</b>") % e
            )
            return False

    # ==== XCcv 平台扩展：类别表同步与分类归档 ====
    def _sync_classes_txt(self):
        """把当前用到的标签同步写入数据集根目录的 classes.txt（每行一个类别）。"""
        if not self.filename:
            return
        root = self.lastOpenDir or osp.dirname(self.filename)
        if not root or not osp.isdir(root):
            return
        txt_path = osp.join(root, "classes.txt")
        classes = []
        if osp.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                classes = [line.strip() for line in f if line.strip()]
        for i in range(self.uniqLabelList.count()):
            label = self.uniqLabelList.item(i).data(Qt.UserRole)
            if label and label not in classes:
                classes.append(label)
        classes = sorted(classes)
        if classes:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(classes) + "\n")
            # 同时维护 label.txt（内容与 classes.txt 一致）
            with open(osp.join(root, "label.txt"), "w",
                      encoding="utf-8") as f:
                f.write("\n".join(classes) + "\n")

    def archiveToClass(self):
        """把当前图片（含同名 json）移动到类别子文件夹。"""
        if not self.filename:
            self.errorMessage("归类到类别", "请先打开图片")
            return
        root = self.lastOpenDir or osp.dirname(self.filename)
        class_dirs = sorted(
            d for d in os.listdir(root) if osp.isdir(osp.join(root, d))
        )
        name, ok = QtWidgets.QInputDialog.getItem(
            self,
            "归类到类别",
            "选择或输入类别（将作为子文件夹名）：",
            class_dirs,
            0,
            True,
        )
        if not ok or not name.strip():
            return
        self._archive_to_class_dir(name.strip())

    def _archive_to_class_dir(self, class_name):
        root = self.lastOpenDir or osp.dirname(self.filename)
        dest_dir = osp.join(root, class_name)
        os.makedirs(dest_dir, exist_ok=True)
        src_json = osp.splitext(self.filename)[0] + ".json"
        dst = osp.join(dest_dir, osp.basename(self.filename))
        if osp.abspath(dst) == osp.abspath(self.filename):
            self.statusBar().showMessage("图片已在该类别中")
            return
        if osp.exists(dst):
            self.errorMessage(
                "归类到类别", "目标已存在同名文件"
            )
            return
        import shutil

        shutil.move(self.filename, dst)
        if osp.exists(src_json):
            shutil.move(
                src_json, osp.join(dest_dir, osp.basename(src_json))
            )
        if self.lastOpenDir:
            self.importDirImages(self.lastOpenDir, load=False)
        self.loadFile(dst)
        self.statusBar().showMessage("已归类到 %s" % class_name)

    def duplicateSelectedShape(self):
        self.copySelectedShape()
        self.pasteSelectedShape()

    def pasteSelectedShape(self):
        self.loadShapes(self._copied_shapes, replace=False)
        self.setDirty()

    def copySelectedShape(self):
        self._copied_shapes = [s.copy() for s in self.canvas.selectedShapes]
        self.actions.paste.setEnabled(len(self._copied_shapes) > 0)

    def labelSelectionChanged(self):
        if self._noSelectionSlot:
            return
        if self.canvas.editing():
            selected_shapes = []
            for item in self.labelList.selectedItems():
                selected_shapes.append(item.shape())
            if selected_shapes:
                self.canvas.selectShapes(selected_shapes)
            else:
                self.canvas.deSelectShape()

    def labelItemChanged(self, item):
        shape = item.shape()
        self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)

    def labelOrderChanged(self):
        self.setDirty()
        self.canvas.loadShapes([item.shape() for item in self.labelList])

    # Callback functions:

    def _undo_last_action(self) -> None:
        """XCcv：画布右键触发的撤销（非绘制状态撤销上一步编辑）。"""
        if self.actions.undo.isEnabled():
            self.actions.undo.trigger()

    def setLanguage(self, lang: str) -> None:
        """XCcv：切换界面语言（重启后生效）。"""
        set_language(lang)
        self.status("语言设置已保存，重启应用后生效 / Saved. Restart to apply.")

    def _exit_drawing_mode(self) -> None:
        """XCcv：退出标注状态时走标准切换流程，恢复所有绘制动作的可用状态。"""
        self.setEditMode()

    def newShape(self):
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """
        items = self.uniqLabelList.selectedItems()
        text = None
        if items:
            text = items[0].data(Qt.UserRole)
        flags = {}
        group_id = None
        description = ""
        if self._config["display_label_popup"] or not text:
            previous_text = self.labelDialog.edit.text()
            text, flags, group_id, description = self.labelDialog.popUp(text)
            if not text:
                self.labelDialog.edit.setText(previous_text)

        if text and not self.validateLabel(text):
            self.errorMessage(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            text = ""
        if text:
            self.labelList.clearSelection()
            shape = self.canvas.setLastLabel(text, flags)
            shape.group_id = group_id
            shape.description = description
            self.addLabel(shape)
            self.actions.editMode.setEnabled(True)
            self.actions.undoLastPoint.setEnabled(False)
            self.actions.undo.setEnabled(True)
            self.setDirty()
        else:
            self.canvas.undoLastLine()
            self.canvas.shapesBackups.pop()

    def scrollRequest(self, delta, orientation):
        units = -delta * 0.1  # natural scroll
        bar = self.scrollBars[orientation]
        value = bar.value() + bar.singleStep() * units
        self.setScroll(orientation, value)

    def setScroll(self, orientation, value):
        self.scrollBars[orientation].setValue(int(value))
        self.scroll_values[orientation][self.filename] = value

    def setZoom(self, value):
        self.actions.fitWidth.setChecked(False)
        self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(value)
        self.zoom_values[self.filename] = (self.zoomMode, value)

    def addZoom(self, increment=1.1):
        zoom_value = self.zoomWidget.value() * increment
        if increment > 1:
            zoom_value = math.ceil(zoom_value)
        else:
            zoom_value = math.floor(zoom_value)
        self.setZoom(zoom_value)

    def zoomRequest(self, delta, pos):
        canvas_width_old = self.canvas.width()
        units = 1.1
        if delta < 0:
            units = 0.9
        self.addZoom(units)

        canvas_width_new = self.canvas.width()
        if canvas_width_old != canvas_width_new:
            canvas_scale_factor = canvas_width_new / canvas_width_old

            x_shift = round(pos.x() * canvas_scale_factor) - pos.x()
            y_shift = round(pos.y() * canvas_scale_factor) - pos.y()

            self.setScroll(
                Qt.Horizontal,
                self.scrollBars[Qt.Horizontal].value() + x_shift,
            )
            self.setScroll(
                Qt.Vertical,
                self.scrollBars[Qt.Vertical].value() + y_shift,
            )

    def setFitWindow(self, value=True):
        if value:
            self.actions.fitWidth.setChecked(False)
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
        if value:
            self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

    def enableKeepPrevScale(self, enabled):
        self._config["keep_prev_scale"] = enabled
        self.actions.keepPrevScale.setChecked(enabled)
        try:   # XCcv（第三批补充）：跨会话记住用户选择
            from xccv.ui_widgets import save_prefs

            save_prefs({"keep_prev_scale": bool(enabled)})
        except Exception:  # noqa: BLE001  偏好写失败不影响功能
            pass

    def onNewBrightnessContrast(self, qimage):
        self.canvas.loadPixmap(QtGui.QPixmap.fromImage(qimage), clear_shapes=False)

    def brightnessContrast(self, value):
        dialog = BrightnessContrastDialog(
            utils.img_data_to_pil(self.imageData),
            self.onNewBrightnessContrast,
            parent=self,
        )
        brightness, contrast = self.brightnessContrast_values.get(
            self.filename, (None, None)
        )
        if brightness is not None:
            dialog.slider_brightness.setValue(brightness)
        if contrast is not None:
            dialog.slider_contrast.setValue(contrast)
        dialog.exec_()

        brightness = dialog.slider_brightness.value()
        contrast = dialog.slider_contrast.value()
        self.brightnessContrast_values[self.filename] = (brightness, contrast)

    def togglePolygons(self, value):
        flag = value
        for item in self.labelList:
            if value is None:
                flag = item.checkState() == Qt.Unchecked
            item.setCheckState(Qt.Checked if flag else Qt.Unchecked)

    def loadFile(self, filename=None):
        """Load the specified file, or the last opened file if None."""
        # changing fileListWidget loads file
        current_item = self.fileListWidget.currentItem()
        current_path = current_item.data(0, QtCore.Qt.UserRole) if current_item else None
        if filename in self.imageList and current_path != filename:
            item = self._file_items.get(filename)
            if item is not None:
                self.fileListWidget.setCurrentItem(item)
            self.fileListWidget.repaint()
            return

        self.resetState()
        self.canvas.setEnabled(False)
        if filename is None:
            filename = self.settings.value("filename", "")
        filename = str(filename)
        if not QtCore.QFile.exists(filename):
            self.errorMessage(
                self.tr("Error opening file"),
                self.tr("No such file: <b>%s</b>") % filename,
            )
            return False
        # assumes same name, but json extension
        self.status(str(self.tr("Loading %s...")) % osp.basename(str(filename)))
        label_file = osp.splitext(filename)[0] + ".json"
        if self.output_dir:
            label_file_without_path = osp.basename(label_file)
            label_file = osp.join(self.output_dir, label_file_without_path)
        if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file):
            try:
                self.labelFile = LabelFile(label_file)
            except LabelFileError as e:
                self.errorMessage(
                    self.tr("Error opening file"),
                    self.tr(
                        "<p><b>%s</b></p>"
                        "<p>Make sure <i>%s</i> is a valid label file."
                    )
                    % (e, label_file),
                )
                self.status(self.tr("Error reading %s") % label_file)
                return False
            self.imageData = self.labelFile.imageData
            self.imagePath = osp.join(
                osp.dirname(label_file),
                self.labelFile.imagePath,
            )
            self.otherData = self.labelFile.otherData
            if not self.imageData:
                # XCcv：store_data=false 时 json 不内嵌图片，改从图片文件读取
                if self.imagePath and QtCore.QFile.exists(self.imagePath):
                    self.imageData = LabelFile.load_image_file(self.imagePath)
        else:
            self.imageData = LabelFile.load_image_file(filename)
            if self.imageData:
                self.imagePath = filename
            self.labelFile = None
        image = (
            QtGui.QImage.fromData(self.imageData)
            if self.imageData
            else QtGui.QImage()
        )

        if image.isNull():
            formats = [
                "*.{}".format(fmt.data().decode())
                for fmt in QtGui.QImageReader.supportedImageFormats()
            ]
            self.errorMessage(
                self.tr("Error opening file"),
                self.tr(
                    "<p>Make sure <i>{0}</i> is a valid image file.<br/>"
                    "Supported image formats: {1}</p>"
                ).format(filename, ",".join(formats)),
            )
            self.status(self.tr("Error reading %s") % filename)
            return False
        self.image = image
        self.filename = filename
        if self._config["keep_prev"]:
            prev_shapes = self.canvas.shapes
        self.canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
        self._apply_grayscale_if_enabled()   # T5-6：灰度显示状态延续
        flags = {k: False for k in self._config["flags"] or []}
        if self.labelFile:
            self.loadLabels(self.labelFile.shapes)
            if self.labelFile.flags is not None:
                flags.update(self.labelFile.flags)
        self.loadFlags(flags)
        if self._config["keep_prev"] and self.noShapes():
            self.loadShapes(prev_shapes, replace=False)
            self.setDirty()
        else:
            self.setClean()
        self.canvas.setEnabled(True)
        # set zoom values
        # XCcv 修复（第三批补充）：按文件记忆的缩放此前无条件恢复——用户在
        # A 图放大后切 B 再回 A，上一张倍率"粘"住，且「保持缩放」开关对其
        # 无效（等于无法关闭）。改为仅在开关开启时恢复；关闭时每张图自动
        # 适应窗口（滚动记忆同口径，避免恢复旧滚动错位）
        is_initial_load = not self.zoom_values
        if self._config["keep_prev_scale"] and self.filename in self.zoom_values:
            self.zoomMode = self.zoom_values[self.filename][0]
            self.setZoom(self.zoom_values[self.filename][1])
        elif is_initial_load or not self._config["keep_prev_scale"]:
            self.adjustScale(initial=True)
        # set scroll values
        if self._config["keep_prev_scale"]:
            for orientation in self.scroll_values:
                if self.filename in self.scroll_values[orientation]:
                    self.setScroll(
                        orientation,
                        self.scroll_values[orientation][self.filename],
                    )
        # set brightness contrast values
        dialog = BrightnessContrastDialog(
            utils.img_data_to_pil(self.imageData),
            self.onNewBrightnessContrast,
            parent=self,
        )
        brightness, contrast = self.brightnessContrast_values.get(
            self.filename, (None, None)
        )
        if self._config["keep_prev_brightness"] and self.recentFiles:
            brightness, _ = self.brightnessContrast_values.get(
                self.recentFiles[0], (None, None)
            )
        if self._config["keep_prev_contrast"] and self.recentFiles:
            _, contrast = self.brightnessContrast_values.get(
                self.recentFiles[0], (None, None)
            )
        if brightness is not None:
            dialog.slider_brightness.setValue(brightness)
        if contrast is not None:
            dialog.slider_contrast.setValue(contrast)
        self.brightnessContrast_values[self.filename] = (brightness, contrast)
        if brightness is not None or contrast is not None:
            dialog.onNewValue(None)
        self.paintCanvas()
        self.addRecentFile(self.filename)
        self.toggleActions(True)
        self.canvas.setFocus()
        self.status(str(self.tr("Loaded %s")) % osp.basename(str(filename)))
        return True

    def resizeEvent(self, event):
        if (
            self.canvas
            and not self.image.isNull()
            and self.zoomMode != self.MANUAL_ZOOM
        ):
            self.adjustScale()
        super(MainWindow, self).resizeEvent(event)

    def paintCanvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()

    def adjustScale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        value = int(100 * value)
        self.zoomWidget.setValue(value)
        self.zoom_values[self.filename] = (self.zoomMode, value)

    def scaleFitWindow(self):
        """Figure out the size of the pixmap to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.centralWidget().width() - e
        h1 = self.centralWidget().height() - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() - 2.0
        return w / self.canvas.pixmap.width()

    def enableSaveImageWithData(self, enabled):
        self._config["store_data"] = enabled
        self.actions.saveWithImageData.setChecked(enabled)

    def closeEvent(self, event):
        if not self.mayContinue():
            event.ignore()
        self.settings.setValue("filename", self.filename if self.filename else "")
        self.settings.setValue("window/size", self.size())
        self.settings.setValue("window/position", self.pos())
        self.settings.setValue("window/state", self.saveState())
        self.settings.setValue("recentFiles", self.recentFiles)
        # ask the use for where to save the labels
        # self.settings.setValue('window/geometry', self.saveGeometry())

    def dragEnterEvent(self, event):
        extensions = [
            ".%s" % fmt.data().decode().lower()
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        if event.mimeData().hasUrls():
            items = [i.toLocalFile() for i in event.mimeData().urls()]
            if any([i.lower().endswith(tuple(extensions)) for i in items]):
                event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if not self.mayContinue():
            event.ignore()
            return
        items = [i.toLocalFile() for i in event.mimeData().urls()]
        self.importDroppedImageFiles(items)

    # User Dialogs #

    def loadRecent(self, filename):
        if self.mayContinue():
            self.loadFile(filename)

    def openPrevImg(self, _value=False):
        keep_prev = self._config["keep_prev"]
        if QtWidgets.QApplication.keyboardModifiers() == (
            Qt.ControlModifier | Qt.ShiftModifier
        ):
            self._config["keep_prev"] = True

        if not self.mayContinue():
            return

        if len(self.imageList) <= 0:
            return

        if self.filename is None:
            return

        currIndex = self.imageList.index(self.filename)
        if currIndex - 1 >= 0:
            filename = self.imageList[currIndex - 1]
            if filename:
                self.loadFile(filename)

        self._config["keep_prev"] = keep_prev

    def openNextImg(self, _value=False, load=True):
        keep_prev = self._config["keep_prev"]
        if QtWidgets.QApplication.keyboardModifiers() == (
            Qt.ControlModifier | Qt.ShiftModifier
        ):
            self._config["keep_prev"] = True

        if not self.mayContinue():
            return

        if len(self.imageList) <= 0:
            return

        filename = None
        if self.filename is None:
            filename = self.imageList[0]
        else:
            currIndex = self.imageList.index(self.filename)
            if currIndex + 1 < len(self.imageList):
                filename = self.imageList[currIndex + 1]
            else:
                filename = self.imageList[-1]
        self.filename = filename

        if self.filename and load:
            self.loadFile(self.filename)

        self._config["keep_prev"] = keep_prev

    def openFile(self, _value=False):
        if not self.mayContinue():
            return
        path = osp.dirname(str(self.filename)) if self.filename else "."
        formats = [
            "*.{}".format(fmt.data().decode())
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        filters = self.tr("Image & Label files (%s)") % " ".join(
            formats + ["*%s" % LabelFile.suffix]
        )
        fileDialog = FileDialogPreview(self)
        fileDialog.setFileMode(FileDialogPreview.ExistingFile)
        fileDialog.setNameFilter(filters)
        fileDialog.setWindowTitle(
            self.tr("%s - Choose Image or Label file") % __appname__,
        )
        fileDialog.setWindowFilePath(path)
        fileDialog.setViewMode(FileDialogPreview.Detail)
        if fileDialog.exec_():
            fileName = fileDialog.selectedFiles()[0]
            if fileName:
                self.loadFile(fileName)

    def changeOutputDirDialog(self, _value=False):
        default_output_dir = self.output_dir
        if default_output_dir is None and self.filename:
            default_output_dir = osp.dirname(self.filename)
        if default_output_dir is None:
            default_output_dir = self.currentPath()

        output_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("%s - Save/Load Annotations in Directory") % __appname__,
            default_output_dir,
            QtWidgets.QFileDialog.ShowDirsOnly
            | QtWidgets.QFileDialog.DontResolveSymlinks,
        )
        output_dir = str(output_dir)

        if not output_dir:
            return

        self.output_dir = output_dir

        self.statusBar().showMessage(
            self.tr("%s . Annotations will be saved/loaded in %s")
            % ("Change Annotations Dir", self.output_dir)
        )
        self.statusBar().show()

        current_filename = self.filename
        self.importDirImages(self.lastOpenDir, load=False)

        if current_filename in self.imageList:
            # retain currently selected file
            item = self._file_items.get(current_filename)
            if item is not None:
                self.fileListWidget.setCurrentItem(item)
            self.fileListWidget.repaint()

    def saveFile(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        if self.labelFile:
            # DL20180323 - overwrite when in directory
            self._saveFile(self.labelFile.filename)
        elif self.output_file:
            self._saveFile(self.output_file)
            self.close()
        else:
            self._saveFile(self.saveFileDialog())

    def saveFileAs(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        self._saveFile(self.saveFileDialog())

    def saveFileDialog(self):
        caption = self.tr("%s - Choose File") % __appname__
        filters = self.tr("Label files (*%s)") % LabelFile.suffix
        if self.output_dir:
            dlg = QtWidgets.QFileDialog(self, caption, self.output_dir, filters)
        else:
            dlg = QtWidgets.QFileDialog(self, caption, self.currentPath(), filters)
        dlg.setDefaultSuffix(LabelFile.suffix[1:])
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.setOption(QtWidgets.QFileDialog.DontConfirmOverwrite, False)
        dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, False)
        basename = osp.basename(osp.splitext(self.filename)[0])
        if self.output_dir:
            default_labelfile_name = osp.join(
                self.output_dir, basename + LabelFile.suffix
            )
        else:
            default_labelfile_name = osp.join(
                self.currentPath(), basename + LabelFile.suffix
            )
        filename = dlg.getSaveFileName(
            self,
            self.tr("Choose File"),
            default_labelfile_name,
            self.tr("Label files (*%s)") % LabelFile.suffix,
        )
        if isinstance(filename, tuple):
            filename, _ = filename
        return filename

    def _saveFile(self, filename):
        if filename and self.saveLabels(filename):
            self.addRecentFile(filename)
            self.setClean()

    def closeFile(self, _value=False):
        if not self.mayContinue():
            return
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)
        self.actions.saveAs.setEnabled(False)

    def getLabelFile(self):
        if self.filename.lower().endswith(".json"):
            label_file = self.filename
        else:
            label_file = osp.splitext(self.filename)[0] + ".json"

        return label_file

    def deleteFile(self):
        mb = QtWidgets.QMessageBox
        msg = self.tr(
            "You are about to permanently delete this label file, " "proceed anyway?"
        )
        answer = mb.warning(self, self.tr("Attention"), msg, mb.Yes | mb.No)
        if answer != mb.Yes:
            return

        label_file = self.getLabelFile()
        if osp.exists(label_file):
            os.remove(label_file)
            logger.info("Label file is removed: {}".format(label_file))

            item = self.fileListWidget.currentItem()
            if item is not None:
                item.setCheckState(0, Qt.Unchecked)

            self.resetState()

    # Message Dialogs. #
    def hasLabels(self):
        if self.noShapes():
            self.errorMessage(
                "No objects labeled",
                "You must label at least one object to save the file.",
            )
            return False
        return True

    def hasLabelFile(self):
        if self.filename is None:
            return False

        label_file = self.getLabelFile()
        return osp.exists(label_file)

    def mayContinue(self):
        if not self.dirty:
            return True
        mb = QtWidgets.QMessageBox
        msg = self.tr('Save annotations to "{}" before closing?').format(self.filename)
        answer = mb.question(
            self,
            self.tr("Save annotations?"),
            msg,
            mb.Save | mb.Discard | mb.Cancel,
            mb.Save,
        )
        if answer == mb.Discard:
            return True
        elif answer == mb.Save:
            self.saveFile()
            return True
        else:  # answer == mb.Cancel
            return False

    def errorMessage(self, title, message):
        return QtWidgets.QMessageBox.critical(
            self, title, "<p><b>%s</b></p>%s" % (title, message)
        )

    def currentPath(self):
        return osp.dirname(str(self.filename)) if self.filename else "."

    def toggleKeepPrevMode(self):
        self._config["keep_prev"] = not self._config["keep_prev"]

    def removeSelectedPoint(self):
        self.canvas.removeSelectedPoint()
        self.canvas.update()
        if not self.canvas.hShape.points:
            self.canvas.deleteShape(self.canvas.hShape)
            self.remLabels([self.canvas.hShape])
            if self.noShapes():
                for action in self.actions.onShapesPresent:
                    action.setEnabled(False)
        self.setDirty()

    def deleteSelectedShape(self):
        yes, no = QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No
        msg = self.tr(
            "You are about to permanently delete {} polygons, " "proceed anyway?"
        ).format(len(self.canvas.selectedShapes))
        if yes == QtWidgets.QMessageBox.warning(
            self, self.tr("Attention"), msg, yes | no, yes
        ):
            self.remLabels(self.canvas.deleteSelected())
            self.setDirty()
            if self.noShapes():
                for action in self.actions.onShapesPresent:
                    action.setEnabled(False)

    def copyShape(self):
        self.canvas.endMove(copy=True)
        for shape in self.canvas.selectedShapes:
            self.addLabel(shape)
        self.labelList.clearSelection()
        self.setDirty()

    def moveShape(self):
        self.canvas.endMove(copy=False)
        self.setDirty()

    def openDirDialog(self, _value=False, dirpath=None):
        if not self.mayContinue():
            return

        defaultOpenDirPath = dirpath if dirpath else "."
        if self.lastOpenDir and osp.exists(self.lastOpenDir):
            defaultOpenDirPath = self.lastOpenDir
        else:
            defaultOpenDirPath = osp.dirname(self.filename) if self.filename else "."

        targetDirPath = str(
            QtWidgets.QFileDialog.getExistingDirectory(
                self,
                self.tr("%s - Open Directory") % __appname__,
                defaultOpenDirPath,
                QtWidgets.QFileDialog.ShowDirsOnly
                | QtWidgets.QFileDialog.DontResolveSymlinks,
            )
        )
        self.importDirImages(targetDirPath)

    @property
    def imageList(self):
        """展平的图片完整路径顺序（根目录在前，子文件夹随后）。"""
        return list(self._file_order)

    def importDroppedImageFiles(self, imageFiles):
        extensions = [
            ".%s" % fmt.data().decode().lower()
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]

        self.filename = None
        for file in imageFiles:
            if file in self.imageList or not file.lower().endswith(tuple(extensions)):
                continue
            label_file = osp.splitext(file)[0] + ".json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            item = QtWidgets.QTreeWidgetItem([osp.basename(file)])
            item.setData(0, QtCore.Qt.UserRole, file)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
            if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file):
                item.setCheckState(0, Qt.Checked)
            else:
                item.setCheckState(0, Qt.Unchecked)
            self.fileListWidget.addTopLevelItem(item)
            self._file_items[file] = item
            self._file_order.append(file)

        if len(self.imageList) > 1:
            self.actions.openNextImg.setEnabled(True)
            self.actions.openPrevImg.setEnabled(True)

        self.openNextImg()

    def importDirImages(self, dirpath, pattern=None, load=True):
        self.actions.openNextImg.setEnabled(True)
        self.actions.openPrevImg.setEnabled(True)

        if not self.mayContinue() or not dirpath:
            return

        self.lastOpenDir = dirpath
        # label.txt 自动加载
        _label_txt = osp.join(dirpath, "label.txt")
        if osp.exists(_label_txt):
            try:
                with open(_label_txt, "r", encoding="utf-8") as _f:
                    for _line in _f:
                        _lb = _line.strip()
                        if _lb and self.uniqLabelList.findItemByLabel(
                                _lb) is None:
                            _item = self.uniqLabelList.createItemFromLabel(
                                _lb)
                            self.uniqLabelList.addItem(_item)
                            self.uniqLabelList.setItemLabel(
                                _item, _lb, self._get_rgb_by_label(_lb))
            except OSError:
                pass
        # XCcv V0.6 T5-6：记录上次目录（ui_prefs）
        try:
            from xccv.ui_widgets import save_prefs

            save_prefs({"annotation_last_dir": str(dirpath)})
        except Exception:  # noqa: BLE001
            pass
        self.filename = None
        self.fileListWidget.clear()
        self._file_items = {}
        self._file_order = []

        filenames = self.scanAllImages(dirpath)
        if pattern:
            try:
                filenames = [f for f in filenames if re.search(pattern, f)]
            except re.error:
                pass
        _done = getattr(self, "_fltDone", None)
        _todo = getattr(self, "_fltTodo", None)
        if _done is not None and _todo is not None \
                and _done.isChecked() != _todo.isChecked():
            want = _done.isChecked()
            filenames = [f for f in filenames
                         if self._has_label_file(f) == want]

        def _make_file_item(filename):
            label_file = osp.splitext(filename)[0] + ".json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            item = QtWidgets.QTreeWidgetItem([osp.basename(filename)])
            item.setData(0, QtCore.Qt.UserRole, filename)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
            if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file):
                item.setCheckState(0, Qt.Checked)
            else:
                item.setCheckState(0, Qt.Unchecked)
            self._file_items[filename] = item
            self._file_order.append(filename)
            return item

        # XCcv：根目录图片作为顶层节点；子文件夹显示为可展开节点（默认收起）
        groups: dict = {}
        for filename in filenames:
            rel_dir = osp.dirname(osp.relpath(filename, dirpath))
            groups.setdefault("" if rel_dir in (".", "") else rel_dir, []).append(filename)
        for filename in groups.get("", []):
            self.fileListWidget.addTopLevelItem(_make_file_item(filename))
        for rel_dir in sorted(groups):
            if rel_dir == "":
                continue
            folder_item = QtWidgets.QTreeWidgetItem([osp.basename(rel_dir)])
            folder_item.setFlags(Qt.ItemIsEnabled)
            self.fileListWidget.addTopLevelItem(folder_item)
            for filename in groups[rel_dir]:
                folder_item.addChild(_make_file_item(filename))
        self.openNextImg(load=load)

    def _has_label_file(self, filename):
        label_file = osp.splitext(filename)[0] + ".json"
        if self.output_dir:
            label_file = osp.join(self.output_dir, osp.basename(label_file))
        return QtCore.QFile.exists(label_file) and \
            LabelFile.is_label_file(label_file)

    def scanAllImages(self, folderPath):
        extensions = [
            ".%s" % fmt.data().decode().lower()
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]

        images = []
        for root, dirs, files in os.walk(folderPath):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    relativePath = os.path.normpath(osp.join(root, file))
                    images.append(relativePath)
        images = natsort.os_sorted(images)
        return images


# ---- Cython 编译态槽包装（信号槽参数裁剪修复，见 xccv/qt_slots.py）----
from xccv.qt_slots import apply_auto_slot_module  # noqa: E402
apply_auto_slot_module(globals())
