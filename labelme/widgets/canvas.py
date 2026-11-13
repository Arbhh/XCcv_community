import collections
import threading
import time
from typing import Optional

import imgviz
from loguru import logger
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets

# XCcv V0.6 T5-2：AI 标注经 xccv.annotation.ai_assist（ultralytics SAM 预测器，
# 编码按图缓存）。保持 osam 形状契约——历史 AI 分支（finalise 精化等）继续可用；
# 权重缺失时 initializeAiModel 调用点给中文提示（T5-3）。
import types as _types

try:
    from xccv.annotation import ai_assist as _ai_assist
except Exception:  # noqa: BLE001  独立测试环境无 xccv 包时保持可导入
    _ai_assist = None

# V0.6 修复循环 4：types 必须接 ai_assist 的真实 dataclass——占位 object()
# 会让 `osam.types.Prompt(...)` 构造即 TypeError（S 智能多边形确认路径必崩）
if _ai_assist is not None:
    _osam_types = _types.SimpleNamespace(
        Model=_ai_assist._UltralyticsSam,
        ImageEmbedding=_ai_assist.ImageEmbedding,
        GenerateResponse=_ai_assist.GenerateResponse,
        GenerateRequest=_ai_assist.GenerateRequest,
        Prompt=_ai_assist.Prompt,
        Point=_ai_assist.Point,
        BoundingBox=_ai_assist.BoundingBox,
    )
else:  # 无 xccv 包的独立环境：保留占位（仅导入期兼容，AI 功能不可用）
    _osam_types = _types.SimpleNamespace(
        Model=object, ImageEmbedding=object, GenerateResponse=object,
        GenerateRequest=object, Prompt=object, Point=object, BoundingBox=object,
    )
osam = _types.SimpleNamespace(
    types=_osam_types,
    apis=_types.SimpleNamespace(
        get_model_type_by_name=(
            (lambda name: _ai_assist.get_model_type_by_name(name))
            if _ai_assist is not None else (lambda name: None)
        ),
        generate=(
            (lambda request: _ai_assist.generate(request))
            if _ai_assist is not None else None
        ),
    ),
)
import numpy as np
from labelme._automation import polygon_from_mask
import labelme.utils
from labelme.shape import Shape

# TODO(unknown):
# - [maybe] Find optimal epsilon value.


CURSOR_DEFAULT = QtCore.Qt.ArrowCursor  # type: ignore[attr-defined]
CURSOR_POINT = QtCore.Qt.PointingHandCursor  # type: ignore[attr-defined]
CURSOR_DRAW = QtCore.Qt.CrossCursor  # type: ignore[attr-defined]
CURSOR_MOVE = QtCore.Qt.ClosedHandCursor  # type: ignore[attr-defined]
CURSOR_GRAB = QtCore.Qt.OpenHandCursor  # type: ignore[attr-defined]

MOVE_SPEED = 5.0


class Canvas(QtWidgets.QWidget):
    # 批注修复：AI 后台加载/状态/悬停预览经信号回主线程（原同步实现卡 UI）
    aiModelReady = QtCore.pyqtSignal(str)
    aiModelFailed = QtCore.pyqtSignal(str)
    aiStatus = QtCore.pyqtSignal(str)
    aiHoverReady = QtCore.pyqtSignal(object)
    aiDecodeDone = QtCore.pyqtSignal(object, object, object)
    aiEmbeddingReady = QtCore.pyqtSignal(object)
    zoomRequest = QtCore.pyqtSignal(int, QtCore.QPoint)
    scrollRequest = QtCore.pyqtSignal(int, int)
    newShape = QtCore.pyqtSignal()
    selectionChanged = QtCore.pyqtSignal(list)
    shapeMoved = QtCore.pyqtSignal()
    drawingPolygon = QtCore.pyqtSignal(bool)
    vertexSelected = QtCore.pyqtSignal(bool)
    mouseMoved = QtCore.pyqtSignal(QtCore.QPointF)

    CREATE, EDIT = 0, 1

    # polygon, rectangle, line, or point
    _createMode = "polygon"

    _fill_drawing = False

    def __init__(self, *args, **kwargs):
        self.epsilon = kwargs.pop("epsilon", 10.0)
        self.double_click = kwargs.pop("double_click", "close")
        if self.double_click not in [None, "close"]:
            raise ValueError(
                "Unexpected value for double_click event: {}".format(self.double_click)
            )
        self.num_backups = kwargs.pop("num_backups", 10)
        _default_crosshair = {
            "polygon": False,
            "rectangle": True,
            "circle": False,
            "line": False,
            "point": False,
            "linestrip": False,
            "ai_polygon": True,    # 批注修复：AI/画笔瞄准必需十字准星
            "ai_mask": True,
            "brush": True,   # V0.6 修复循环 4：画笔模式（缺键 paintEvent KeyError）
        }
        _crosshair_cfg = kwargs.pop("crosshair", None)
        if isinstance(_crosshair_cfg, dict):
            _default_crosshair.update(_crosshair_cfg)
        self._crosshair = _default_crosshair
        super(Canvas, self).__init__(*args, **kwargs)
        # Initialise local state.
        self.mode = self.EDIT
        self.shapes = []
        self.shapesBackups = []
        self.current = None
        self.selectedShapes = []  # save the selected shapes here
        self.selectedShapesCopy = []
        # self.line represents:
        #   - createMode == 'polygon': edge from last point to current
        #   - createMode == 'rectangle': diagonal line of the rectangle
        #   - createMode == 'line': the line
        #   - createMode == 'point': the point
        self.line = Shape()
        self.prevPoint = QtCore.QPoint()
        self.prevMovePoint = QtCore.QPoint()
        self.offsets = QtCore.QPoint(), QtCore.QPoint()
        self.scale = 1.0
        self.pixmap = QtGui.QPixmap()
        self.visible = {}
        self._hideBackround = False
        self.hideBackround = False
        self.hShape = None
        self.prevhShape = None
        self.hVertex = None
        self.prevhVertex = None
        self.hEdge = None
        self.prevhEdge = None
        self.movingShape = False
        self.snapping = True
        self.hShapeIsSelected = False
        self._painter = QtGui.QPainter()
        self._cursor = CURSOR_DEFAULT
        # Menus:
        # 0: right-click without selection and dragging of shapes
        # 1: right-click with selection and dragging of shapes
        self.menus = (QtWidgets.QMenu(), QtWidgets.QMenu())
        # XCcv：右键撤销回调（主窗口注入）与画布平移状态
        self.on_right_click = None
        self.on_exit_drawing = None  # 退出标注状态回调（主窗口走标准切换流程）
        self._panning = False
        # Set widget options.
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.WheelFocus)

        self._sam: Optional[osam.types.Model] = None
        self._sam_embedding: collections.OrderedDict[
            bytes, osam.types.ImageEmbedding
        ] = collections.OrderedDict()
        # XCcv V0.6 T5-3/T5-5：智能多边形提示点与预览、画笔状态
        self._ai_prompts: list = []        # [(QPointF, label)]，label 1=并入 0=挖除
        self._ai_preview_shape = None      # 悬停预览（半透明多边形）
        self._ai_last_hover_t = 0.0
        self._ai_last_mask = None
        self._brush_width = 24             # 画笔直径（像素，[ ] 或滚轮调节）
        self._show_labels = True           # T5-6：类别文字显隐（C）
        self._brush_pts: list = []         # 当前笔画轨迹（图像坐标）
        # 右键擦除轨迹：右键拖动=挖除
        self._brush_erase_pts: list = []
        self._brush_erasing = False
        self._brush_last_cursor = None    # 画笔局部重绘：上一帧圈心
        # 批注修复（用户可调参数）：AI 多边形点数上限/简化容差/悬停预览开关
        self._ai_max_points = 100
        self._ai_tolerance = 0.004
        self._ai_hover_enabled = True
        self._ai_loading = False
        self._ai_hover_busy = False
        self._sam_lock = threading.Lock()
        self.aiHoverReady.connect(self._ai_hover_apply)
        self._px_img = None            # QImage 缓存（准星取背景色）
        # 智能多边形交互（第二批：先判双击再建锚点 + 单线程解码 + 结果缓存）
        self._ai_pending = None        # 双击判定窗口内的待定点击
        self._ai_click_timer = QtCore.QTimer(self)
        self._ai_click_timer.setSingleShot(True)
        self._ai_click_timer.timeout.connect(self._ai_commit_pending)
        self._ai_results = collections.OrderedDict()  # 提示键 -> 顶点 | None
        self._ai_gen = 0               # 换图/换模型/改参数递增，旧结果作废
        self._ai_busy = False
        self._ai_inflight = None
        self._ai_queue = None
        self._ai_finalise_wanted = False
        self._ai_emb_loading = None
        self._ai_last_commit = None    # (按下时刻, 锚点数)：双击兜底回滚
        self._ai_last_finalised = None
        self._ai_first_ratio = 0.9     # 首锚占比（两锚时，0.5~0.9 可调）
        self._ai_hover_interval = 0.12
        self._ai_largest_only = True
        self.aiDecodeDone.connect(self._ai_on_decode_done)
        self.aiEmbeddingReady.connect(self._ai_on_embedding_ready)

    def fillDrawing(self):
        return self._fill_drawing

    def setFillDrawing(self, value):
        self._fill_drawing = value

    @property
    def createMode(self):
        return self._createMode

    @createMode.setter
    def createMode(self, value):
        if value not in [
            "polygon",
            "rectangle",
            "circle",
            "line",
            "point",
            "linestrip",
            "ai_polygon",
            "ai_mask",
            "brush",
        ]:
            raise ValueError("Unsupported createMode: %s" % value)
        if getattr(self, "_createMode", None) != value:
            # 批注修复（重点 bug）：切换标注方式必须清空 AI 提示/预览/
            # 画笔残留——上一模式的锚点继续作用导致错乱与无法闭合
            self._ai_prompts = []
            self._ai_reset_interaction()
            self._ai_preview_shape = None
            self._ai_last_mask = None
            self._brush_pts = []
            self._brush_erase_pts = []
            self._brush_erasing = False
            if getattr(self, "current", None) is not None:
                self.current = None
                self.drawingPolygon.emit(False)
        self._createMode = value

    def _ai_emb_key(self):
        if self._sam is None or self.pixmap is None or self.pixmap.isNull():
            return None
        return (getattr(self._sam, "name", ""), self.pixmap.cacheKey())

    def _ai_store_embedding(self, key, emb):
        self._sam_embedding[key] = emb
        self._sam_embedding.move_to_end(key)
        while len(self._sam_embedding) > 3:
            self._sam_embedding.popitem(last=False)

    def _ai_embedding(self):
        """当前图嵌入；未就绪返回 None 并触发后台编码（UI 线程不等待）。"""
        key = self._ai_emb_key()
        if key is None:
            return None
        emb = self._sam_embedding.get(key)
        if emb is None:
            self._preload_embedding_async()
        return emb

    def _compute_and_cache_image_embedding(self) -> None:
        if self._sam is None:
            logger.warning("SAM model is not set yet")
            return
        key = self._ai_emb_key()
        if key is None or key in self._sam_embedding:
            return
        logger.debug("Computing image embeddings for model {!r}",
                     self._sam.name)
        image = labelme.utils.img_qt_to_arr(self.pixmap.toImage())
        with self._sam_lock:
            emb = self._sam.encode_image(image=imgviz.asrgb(image))
        self._ai_store_embedding(key, emb)

    def initializeAiModel(self, model_name):
        if self.pixmap is None:
            logger.warning("Pixmap is not set yet")
            return

        if self._sam is None or self._sam.name != model_name:
            logger.debug("Initializing AI model {!r}", model_name)
            self._sam = osam.apis.get_model_type_by_name(model_name)()
            self._sam_embedding.clear()

        self._compute_and_cache_image_embedding()

    def initializeAiModelAsync(self, model_name):
        """批注修复：SAM 模型+嵌入全部后台加载（原同步实现卡死 UI 数秒）。

        完成后经 aiModelReady/aiModelFailed 通知主窗口状态栏；加载期间
        点击会收到 aiStatus 提示而不是卡顿。
        """
        if self._ai_loading:
            self.aiStatus.emit("AI 模型正在后台加载，请稍候…")
            return
        if self._sam is not None and self._sam.name == model_name:
            self._preload_embedding_async()
            return
        self._ai_loading = True
        self.aiStatus.emit(f"AI 模型后台加载中（{model_name}）…")

        def _work():
            try:
                self._sam_pending = \
                    osam.apis.get_model_type_by_name(model_name)()
                _m = np.zeros((32, 32), bool)
                _m[8:24, 8:24] = True
                polygon_from_mask.compute_polygon_from_mask(mask=_m)
                self.aiModelReady.emit(model_name)
            except Exception as e:  # noqa: BLE001
                # 失败必须解锁：_ai_loading 卡 True 会吞掉后续所有点击与
                # 重试（S 键入口首行即被拦），AI 标注永久失效（T5 根因）
                self._ai_loading = False
                self.aiModelFailed.emit(str(e))

        threading.Thread(target=_work, daemon=True).start()

    def _apply_pending_model(self, model_name):
        model = getattr(self, "_sam_pending", None)
        if model is not None:
            self._sam = model
            self._sam_embedding.clear()
            self._sam_pending = None
            self._ai_invalidate()
        self._ai_loading = False
        self._preload_embedding_async()

    def _preload_embedding_async(self):
        """UI 线程快照像素 → 后台编码嵌入（首次点击零等待、换图不阻塞）。"""
        key = self._ai_emb_key()
        if key is None or key in self._sam_embedding \
                or self._ai_emb_loading == key:
            return
        arr = labelme.utils.img_qt_to_arr(self.pixmap.toImage())
        sam = self._sam
        self._ai_emb_loading = key

        def _work():
            emb, err = None, ""
            try:
                with self._sam_lock:
                    emb = sam.encode_image(image=imgviz.asrgb(arr))
            except Exception as e:  # noqa: BLE001
                err = str(e)
            self.aiEmbeddingReady.emit((key, emb, err))

        threading.Thread(target=_work, daemon=True).start()

    def _ai_on_embedding_ready(self, payload):
        key, emb, err = payload
        if self._ai_emb_loading == key:
            self._ai_emb_loading = None
        if emb is None:
            self.aiStatus.emit(f"AI 嵌入失败：{err}")
            return
        self._ai_store_embedding(key, emb)
        self.aiStatus.emit("AI 图像嵌入已就绪")
        self._ai_pump()

    def storeShapes(self):
        shapesBackup = []
        for shape in self.shapes:
            shapesBackup.append(shape.copy())
        if len(self.shapesBackups) > self.num_backups:
            self.shapesBackups = self.shapesBackups[-self.num_backups - 1 :]
        self.shapesBackups.append(shapesBackup)

    @property
    def isShapeRestorable(self):
        # We save the state AFTER each edit (not before) so for an
        # edit to be undoable, we expect the CURRENT and the PREVIOUS state
        # to be in the undo stack.
        if len(self.shapesBackups) < 2:
            return False
        return True

    def restoreShape(self):
        # This does _part_ of the job of restoring shapes.
        # The complete process is also done in app.py::undoShapeEdit
        # and app.py::loadShapes and our own Canvas::loadShapes function.
        if not self.isShapeRestorable:
            return
        self.shapesBackups.pop()  # latest

        # The application will eventually call Canvas.loadShapes which will
        # push this right back onto the stack.
        shapesBackup = self.shapesBackups.pop()
        self.shapes = shapesBackup
        self.selectedShapes = []
        for shape in self.shapes:
            shape.selected = False
        self.update()

    def enterEvent(self, ev):
        self.overrideCursor(self._cursor)

    def leaveEvent(self, ev):
        self.unHighlight()
        self.restoreCursor()
        self._brush_last_cursor = None

    def focusOutEvent(self, ev):
        self.restoreCursor()

    def isVisible(self, shape):
        return self.visible.get(shape, True)

    def drawing(self):
        return self.mode == self.CREATE

    def editing(self):
        return self.mode == self.EDIT

    def setEditing(self, value=True):
        self.mode = self.EDIT if value else self.CREATE
        if self.mode == self.EDIT:
            # CREATE -> EDIT
            self.repaint()  # clear crosshair
        else:
            # EDIT -> CREATE
            self.unHighlight()
            self.deSelectShape()

    def unHighlight(self):
        if self.hShape:
            self.hShape.highlightClear()
            self.update()
        self.prevhShape = self.hShape
        self.prevhVertex = self.hVertex
        self.prevhEdge = self.hEdge
        self.hShape = self.hVertex = self.hEdge = None

    def selectedVertex(self):
        return self.hVertex is not None

    def selectedEdge(self):
        return self.hEdge is not None

    def mouseMoveEvent(self, ev):
        # XCcv：平移中——按住对应按键才移动，按键松开自动结束
        if self._panning:
            if ev.buttons() & self._pan_button:
                self._do_pan(ev)
            else:
                self._end_pan()
            return
        """Update line with last point and current coordinates."""
        try:
            pos = self.transformPos(ev.localPos())
        except AttributeError:
            return

        self.mouseMoved.emit(pos)

        self.prevMovePoint = pos
        self.restoreCursor()

        is_shift_pressed = ev.modifiers() & QtCore.Qt.ShiftModifier

        # Polygon drawing.
        if self.drawing():
            # V0.6 修复循环 4：brush 同 ai 族以 "points" 落地（白名单无 "brush"，
            # 原样赋 createMode 会在移动预览时抛 ValueError——真机堆栈定位）
            if self.createMode in ["ai_polygon", "ai_mask", "brush"]:
                self.line.shape_type = "points"
            else:
                self.line.shape_type = self.createMode

            self.overrideCursor(CURSOR_DRAW)
            if not self.current and not (
                    self.createMode == "brush"
                    and (self._brush_erasing or self._brush_pts)):
                if self.createMode == "brush":
                    # 第三批：笔刷框局部重绘（repaint() 同步整幅重绘是
                    # 延迟根因——大图+多形状时圈远落后于系统十字光标）
                    self._brush_region_update(pos)
                else:
                    self.repaint()  # draw crosshair
                return

            if self.outOfPixmap(pos):
                # Don't allow the user to draw outside the pixmap.
                # Project the point to the pixmap's edges.
                pos = self.intersectionPoint(self.current[-1], pos)
            elif (
                self.snapping
                and self.current is not None
                and len(self.current) > 1
                and self.createMode == "polygon"
                and self.closeEnough(pos, self.current[0])
            ):
                # Attract line to starting point and
                # colorise to alert the user.
                pos = self.current[0]
                self.overrideCursor(CURSOR_POINT)
                self.current.highlightVertex(0, Shape.NEAR_VERTEX)
            if self.createMode in ["polygon", "linestrip"]:
                self.line.points = [self.current[-1], pos]
                self.line.point_labels = [1, 1]
            elif self.createMode in ["ai_polygon", "ai_mask"]:
                self.line.points = [self.current.points[-1], pos]
                self.line.point_labels = [
                    self.current.point_labels[-1],
                    0 if is_shift_pressed else 1,
                ]
                self._ai_hover_maybe(pos, is_shift_pressed)
            elif self.createMode == "brush":
                if self.current is not None and self.current.points:
                    self.line.points = [self.current.points[-1], pos]
                    self.line.point_labels = [1, 1]
                # 批注修复：拖动必须持续追加轨迹点（原实现只记起点——
                # 松手仅一个点，"无法拖动/作用点位与准星相差甚远"根因；
                # 2026-11-05 测试批：0 顶点 current 为假值导致每次 move
                # 都在上面的守卫早退——真机拖动/纯擦除仍只有单点圆，
                # 故落笔点入形状 + 守卫放行画笔手势）
                if self._brush_erasing:
                    if not self._brush_erase_pts or (
                            (pos - self._brush_erase_pts[-1])
                            .manhattanLength() >= 1):
                        self._brush_erase_pts.append(QtCore.QPointF(pos))
                elif not self._brush_pts or (
                        (pos - self._brush_pts[-1]).manhattanLength() >= 1):
                    self._brush_pts.append(QtCore.QPointF(pos))
            elif self.createMode == "rectangle":
                self.line.points = [self.current[0], pos]
                self.line.point_labels = [1, 1]
                self.line.close()
            elif self.createMode == "circle":
                self.line.points = [self.current[0], pos]
                self.line.point_labels = [1, 1]
                self.line.shape_type = "circle"
            elif self.createMode == "line":
                self.line.points = [self.current[0], pos]
                self.line.point_labels = [1, 1]
                self.line.close()
            elif self.createMode == "point":
                self.line.points = [self.current[0]]
                self.line.point_labels = [1]
                self.line.close()
            assert len(self.line.points) == len(self.line.point_labels)
            if self.createMode == "brush":
                self._brush_region_update(pos)
            else:
                self.repaint()
            if self.current is not None:
                self.current.highlightClear()
            return

        # Polygon/Vertex moving.
        if QtCore.Qt.LeftButton & ev.buttons():
            if self.selectedVertex():
                self.boundedMoveVertex(pos)
                self.repaint()
                self.movingShape = True
            elif self.selectedShapes and self.prevPoint:
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShapes(self.selectedShapes, pos)
                self.repaint()
                self.movingShape = True
            return

        # Just hovering over the canvas, 2 possibilities:
        # - Highlight shapes
        # - Highlight vertex
        # Update shape/vertex fill and tooltip value accordingly.
        self.setToolTip(self.tr("Image"))
        for shape in reversed([s for s in self.shapes if self.isVisible(s)]):
            # Look for a nearby vertex to highlight. If that fails,
            # check if we happen to be inside a shape.
            index = shape.nearestVertex(pos, self.epsilon)
            index_edge = shape.nearestEdge(pos, self.epsilon)
            if index is not None:
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex = index
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge
                self.hEdge = None
                shape.highlightVertex(index, shape.MOVE_VERTEX)
                self.overrideCursor(CURSOR_POINT)
                self.setToolTip(
                    self.tr(
                        "Click & Drag to move point\n"
                        "ALT + SHIFT + Click to delete point"
                    )
                )
                self.setStatusTip(self.toolTip())
                self.update()
                break
            elif index_edge is not None and shape.canAddPoint():
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex
                self.hVertex = None
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge = index_edge
                self.overrideCursor(CURSOR_POINT)
                self.setToolTip(self.tr("ALT + Click to create point"))
                self.setStatusTip(self.toolTip())
                self.update()
                break
            elif shape.containsPoint(pos):
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex
                self.hVertex = None
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge
                self.hEdge = None
                self.setToolTip(
                    self.tr("Click & drag to move shape '%s'") % shape.label
                )
                self.setStatusTip(self.toolTip())
                self.overrideCursor(CURSOR_GRAB)
                self.update()
                break
        else:  # Nothing found, clear highlights, reset state.
            self.unHighlight()
        self.vertexSelected.emit(self.hVertex is not None)

    def addPointToEdge(self):
        shape = self.prevhShape
        index = self.prevhEdge
        point = self.prevMovePoint
        if shape is None or index is None or point is None:
            return
        shape.insertPoint(index, point)
        shape.highlightVertex(index, shape.MOVE_VERTEX)
        self.hShape = shape
        self.hVertex = index
        self.hEdge = None
        self.movingShape = True

    def removeSelectedPoint(self):
        shape = self.prevhShape
        index = self.prevhVertex
        if shape is None or index is None:
            return
        shape.removePoint(index)
        shape.highlightClear()
        self.hShape = shape
        self.prevhVertex = None
        self.movingShape = True  # Save changes

    def mousePressEvent(self, ev):
        pos = self.transformPos(ev.localPos())

        is_shift_pressed = ev.modifiers() & QtCore.Qt.ShiftModifier

        # XCcv：平移视图，按住才拖、松开即停（Shift+左键 / 中键 / 编辑模式空白处左键）
        pan_button = None
        if ev.button() == QtCore.Qt.MiddleButton:
            pan_button = ev.button()
        elif ev.button() == QtCore.Qt.LeftButton and (
            self.editing() and not self.drawing()
            and (is_shift_pressed or self._shape_at(pos) is None)
        ):
            pan_button = ev.button()
        if pan_button is not None and self._begin_pan(ev, pan_button):
            return

        if ev.button() == QtCore.Qt.LeftButton:
            if self.drawing():
                if self.current is not None:
                    # Add point to existing shape.
                    if self.createMode == "polygon":
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        if self.current.isClosed():
                            self.finalise()
                    elif self.createMode in ["rectangle", "circle", "line"]:
                        assert len(self.current.points) == 1
                        self.current.points = self.line.points
                        self.finalise()
                    elif self.createMode == "linestrip":
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        if int(ev.modifiers()) == QtCore.Qt.ControlModifier:
                            self.finalise()
                    elif self.createMode in ["ai_polygon", "ai_mask"]:
                        # 先判双击再建锚点（第二批用户要求）：按下处即锚点
                        # 位置（line 端点可能是未移动鼠标时的陈旧值）
                        self._ai_click(self._ai_clamp(pos),
                                       0 if is_shift_pressed else 1,
                                       ev.localPos(), ev.modifiers())
                    elif self.createMode == "brush":
                        self._brush_pts = [self.line.points[1]]
                        self.current.addPoint(self.line.points[1])
                elif not self.outOfPixmap(pos):
                    # Create new shape.
                    # V0.6 修复循环 4：brush 也以 "points" 落地——Shape 白名单
                    # 不含 "brush"，构造即抛 ValueError（真机画笔必现）
                    self.current = Shape(
                        shape_type="points"
                        if self.createMode in ["ai_polygon", "ai_mask", "brush"]
                        else self.createMode
                    )
                    if self.createMode in ["ai_polygon", "ai_mask"] \
                            and self._ai_loading:
                        # 批注修复：模型后台加载中——提示而非卡顿/报错。
                        # 仅拦"加载中"；无模型（未装权重）仍允许提示点入栈
                        #（历史行为：先标记后加载，test_v06_annotation 锁定）
                        self.aiStatus.emit("AI 模型加载中，请稍候")
                        self.current = None
                        self.drawingPolygon.emit(False)
                        return
                    if self.createMode in ["ai_polygon", "ai_mask"]:
                        # 提示点入栈+实时精化（原体——此前被守卫 return 遮蔽成
                        # 不可达死代码，AI 点击从未入栈="锚点不显示"的另一半根因）
                        self.current = Shape(shape_type="points")
                        self._ai_reset_interaction()
                        self._ai_prompts = []
                        self._ai_preview_shape = None
                        # 首锚即时生效（尚无可闭合的形状，无需双击判定）
                        self._ai_add_anchor(QtCore.QPointF(pos),
                                            0 if is_shift_pressed else 1)
                        self.setHiding()
                        self.drawingPolygon.emit(True)
                        self.update()
                        return
                    if self.createMode == "brush":
                        self.current = Shape(shape_type="points")
                        self._brush_pts = [pos]
                        self.current.addPoint(pos)   # 0 顶点为假值——不入形状
                        self.setHiding()
                        self.drawingPolygon.emit(True)
                        self.update()
                        return
                    self.current.addPoint(pos, label=0 if is_shift_pressed else 1)
                    if self.createMode == "point":
                        self.finalise()
                    else:
                        if self.createMode == "circle":
                            self.current.shape_type = "circle"
                        self.line.points = [pos, pos]
                        if (
                            self.createMode in ["ai_polygon", "ai_mask"]
                            and is_shift_pressed
                        ):
                            self.line.point_labels = [0, 0]
                        else:
                            self.line.point_labels = [1, 1]
                        self.setHiding()
                        self.drawingPolygon.emit(True)
                        self.update()
            elif self.editing():
                if self.selectedEdge() and ev.modifiers() == QtCore.Qt.AltModifier:
                    self.addPointToEdge()
                elif self.selectedVertex() and ev.modifiers() == (
                    QtCore.Qt.AltModifier | QtCore.Qt.ShiftModifier
                ):
                    self.removeSelectedPoint()

                group_mode = int(ev.modifiers()) == QtCore.Qt.ControlModifier
                self.selectShapePoint(pos, multiple_selection_mode=group_mode)
                self.prevPoint = pos
                self.repaint()
        elif ev.button() == QtCore.Qt.RightButton:
            # XCcv：右键 = 撤销（绘制中撤销上一个点；点删光后仍留在标注状态，
            # 再次右键才通过标准流程退出标注状态）
            if self.createMode == "brush":
                # 画笔右键拖动=擦除（含未起笔时直接擦既有形状）；
                # 撤销本笔仍用 Esc/E
                self._brush_erasing = True
                self._brush_erase_pts = [QtCore.QPointF(pos)]
                self.update()
                return
            if self.drawing():
                if self.current is not None:
                    self.undoLastPoint()
                else:
                    self._exit_drawing()
            elif self.on_right_click is not None:
                self.on_right_click()

    def mouseReleaseEvent(self, ev):
        # XCcv：松开平移按键即结束拖动
        if self._panning and ev.button() == self._pan_button:
            self._end_pan()
            return
        if ev.button() == QtCore.Qt.RightButton and self._brush_erasing:
            self._brush_erasing = False
            if self.createMode == "brush" and not self._brush_pts:
                self._brush_finalise()   # 纯擦除手势立即提交
            else:
                self.update()
            return
        if ev.button() == QtCore.Qt.LeftButton and self.drawing() \
                and self.createMode == "brush" and self.current is not None:
            if len(getattr(self, "_brush_pts", []) or []) >= 1:
                self.finalise()
            return
        if ev.button() == QtCore.Qt.RightButton:
            # XCcv：右键撤销已在按下时处理
            return
        if ev.button() == QtCore.Qt.LeftButton:
            if self.editing():
                if (
                    self.hShape is not None
                    and self.hShapeIsSelected
                    and not self.movingShape
                ):
                    self.selectionChanged.emit(
                        [x for x in self.selectedShapes if x != self.hShape]
                    )

        if self.movingShape and self.hShape:
            index = self.shapes.index(self.hShape)
            if self.shapesBackups[-1][index].points != self.shapes[index].points:
                self.storeShapes()
                self.shapeMoved.emit()

            self.movingShape = False

    def endMove(self, copy):
        assert self.selectedShapes and self.selectedShapesCopy
        assert len(self.selectedShapesCopy) == len(self.selectedShapes)
        if copy:
            for i, shape in enumerate(self.selectedShapesCopy):
                self.shapes.append(shape)
                self.selectedShapes[i].selected = False
                self.selectedShapes[i] = shape
        else:
            for i, shape in enumerate(self.selectedShapesCopy):
                self.selectedShapes[i].points = shape.points
        self.selectedShapesCopy = []
        self.repaint()
        self.storeShapes()
        return True

    def hideBackroundShapes(self, value):
        self.hideBackround = value
        if self.selectedShapes:
            # Only hide other shapes if there is a current selection.
            # Otherwise the user will not be able to select a shape.
            self.setHiding(True)
            self.update()

    def setHiding(self, enable=True):
        self._hideBackround = self.hideBackround if enable else False

    def canCloseShape(self):
        return self.drawing() and (
            (self.current and len(self.current) > 2)
            or self.createMode in ["ai_polygon", "ai_mask"]
        )

    def mouseDoubleClickEvent(self, ev):
        if self.double_click != "close":
            return
        if self.createMode in ["ai_polygon", "ai_mask"]:
            if self.drawing() and self.current is not None:
                self._ai_double_click_close()
            return
        if self.createMode == "polygon" and self.canCloseShape():
            self.finalise()

    def selectShapes(self, shapes):
        self.setHiding()
        self.selectionChanged.emit(shapes)
        self.update()

    def selectShapePoint(self, point, multiple_selection_mode):
        """Select the first shape created which contains this point."""
        if self.selectedVertex():  # A vertex is marked for selection.
            index, shape = self.hVertex, self.hShape
            shape.highlightVertex(index, shape.MOVE_VERTEX)
        else:
            for shape in reversed(self.shapes):
                if self.isVisible(shape) and shape.containsPoint(point):
                    self.setHiding()
                    if shape not in self.selectedShapes:
                        if multiple_selection_mode:
                            self.selectionChanged.emit(self.selectedShapes + [shape])
                        else:
                            self.selectionChanged.emit([shape])
                        self.hShapeIsSelected = False
                    else:
                        self.hShapeIsSelected = True
                    self.calculateOffsets(point)
                    return
        self.deSelectShape()

    def calculateOffsets(self, point):
        left = self.pixmap.width() - 1
        right = 0
        top = self.pixmap.height() - 1
        bottom = 0
        for s in self.selectedShapes:
            rect = s.boundingRect()
            if rect.left() < left:
                left = rect.left()
            if rect.right() > right:
                right = rect.right()
            if rect.top() < top:
                top = rect.top()
            if rect.bottom() > bottom:
                bottom = rect.bottom()

        x1 = left - point.x()
        y1 = top - point.y()
        x2 = right - point.x()
        y2 = bottom - point.y()
        self.offsets = QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)

    def boundedMoveVertex(self, pos):
        index, shape = self.hVertex, self.hShape
        point = shape[index]
        if self.outOfPixmap(pos):
            pos = self.intersectionPoint(point, pos)
        shape.moveVertexBy(index, pos - point)

    def boundedMoveShapes(self, shapes, pos):
        if self.outOfPixmap(pos):
            return False  # No need to move
        o1 = pos + self.offsets[0]
        if self.outOfPixmap(o1):
            pos -= QtCore.QPointF(min(0, o1.x()), min(0, o1.y()))
        o2 = pos + self.offsets[1]
        if self.outOfPixmap(o2):
            pos += QtCore.QPointF(
                min(0, self.pixmap.width() - o2.x()),
                min(0, self.pixmap.height() - o2.y()),
            )
        # XXX: The next line tracks the new position of the cursor
        # relative to the shape, but also results in making it
        # a bit "shaky" when nearing the border and allows it to
        # go outside of the shape's area for some reason.
        # self.calculateOffsets(self.selectedShapes, pos)
        dp = pos - self.prevPoint
        if dp:
            for shape in shapes:
                shape.moveBy(dp)
            self.prevPoint = pos
            return True
        return False

    def deSelectShape(self):
        if self.selectedShapes:
            self.setHiding(False)
            self.selectionChanged.emit([])
            self.hShapeIsSelected = False
            self.update()

    def deleteSelected(self):
        deleted_shapes = []
        if self.selectedShapes:
            for shape in self.selectedShapes:
                if shape in self.shapes:
                    self.shapes.remove(shape)
                    deleted_shapes.append(shape)
            self.storeShapes()
            self.selectedShapes = []
            self.update()
        return deleted_shapes

    def deleteShape(self, shape):
        if shape in self.selectedShapes:
            self.selectedShapes.remove(shape)
        if shape in self.shapes:
            self.shapes.remove(shape)
        self.storeShapes()
        self.update()

    def paintEvent(self, event: Optional[QtGui.QPaintEvent]) -> None:
        if not self.pixmap:
            return super(Canvas, self).paintEvent(event)

        p = self._painter
        p.begin(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.HighQualityAntialiasing)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        p.scale(self.scale, self.scale)
        p.translate(self.offsetToCenter())

        p.drawPixmap(0, 0, self.pixmap)

        p.scale(1 / self.scale, 1 / self.scale)

        # 准星绘制已移除（用户反馈 2026-11-06）：矩形/智能多边形/画笔
        # 统一用 labelme 原生鼠标光标样式（CURSOR_DRAW 十字光标），
        # 不再自绘任何跟随鼠标的锚框

        Shape.scale = self.scale
        for shape in self.shapes:
            if (shape.selected or not self._hideBackround) and self.isVisible(shape):
                shape.fill = shape.selected or shape == self.hShape
                shape.paint(p)

        # XCcv：类别名称贴着标注目标（质心处，字号不随缩放变化）
        if self._show_labels and not self._hideBackround:
            p.save()
            p.resetTransform()
            offset = self.offsetToCenter()
            font = QtGui.QFont()
            font.setPointSize(10)
            font.setBold(True)
            p.setFont(font)
            metrics = QtGui.QFontMetrics(p.font())
            for shape in self.shapes:
                if not self.isVisible(shape) or not shape.label:
                    continue
                xs = [pt.x() for pt in shape.points]
                ys = [pt.y() for pt in shape.points]
                cxp, cyp = None, None
                if shape.shape_type in ("polygon", "mask") \
                        and len(shape.points) >= 3:
                    ax = np.asarray(xs, dtype=float)
                    ay = np.asarray(ys, dtype=float)
                    bx = np.roll(ax, -1)
                    by = np.roll(ay, -1)
                    cr = ax * by - bx * ay
                    area = cr.sum() / 2.0
                    if abs(area) > 1e-9:
                        cxp = float(((ax + bx) * cr).sum() / (6.0 * area))
                        cyp = float(((ay + by) * cr).sum() / (6.0 * area))
                if cxp is None:
                    cxp = (min(xs) + max(xs)) / 2.0
                    cyp = (min(ys) + max(ys)) / 2.0
                text = shape.label
                w = metrics.horizontalAdvance(text) + 8
                h = metrics.height() + 2
                x = int(cxp * self.scale + offset.x() * self.scale) - w // 2
                y = int(cyp * self.scale + offset.y() * self.scale)
                x = max(0, min(x, self.width() - w))
                y = max(h, min(y, self.height() - 4))
                p.fillRect(x, y - h, w, h, QtGui.QColor(0, 0, 0, 130))
                p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
                p.drawText(x + 4, y - 5, text)
            p.restore()
        if self._ai_prompts and self.drawing() \
                and self.createMode in ["ai_polygon", "ai_mask"]:
            # 批注修复：已点击锚点必须可见（原实现不渲染提示点）
            for q, lab in self._ai_prompts:
                r = 4.0   # 固定设备像素（用户反馈：锚点不应随缩小视图巨大化）
                p.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0), 1))
                p.setBrush(QtGui.QColor(0, 230, 0) if lab
                           else QtGui.QColor(255, 60, 60))
                p.drawEllipse(QtCore.QPointF(
                    q.x() * self.scale, q.y() * self.scale), r, r)
            p.setBrush(QtCore.Qt.NoBrush)
        if self._ai_pending is not None and self.drawing() \
                and self.createMode in ["ai_polygon", "ai_mask"]:
            # 待定点击（双击判定中）：空心圈即时反馈，确认单击后转实心锚点
            q, lab = self._ai_pending[0], self._ai_pending[1]
            p.setPen(QtGui.QPen(QtGui.QColor(0, 230, 0) if lab
                                else QtGui.QColor(255, 60, 60), 2))
            p.setBrush(QtCore.Qt.NoBrush)
            p.drawEllipse(QtCore.QPointF(
                q.x() * self.scale, q.y() * self.scale), 4.0, 4.0)
        if self.current:
            self.current.paint(p)
            if not (self.createMode == "brush" and self.prevMovePoint):
                assert len(self.line.points) == len(self.line.point_labels)
                self.line.paint(p)
        if self.createMode == "brush" and self.drawing() \
                and self.prevMovePoint is not None:
            # 批注修复：画笔圈不再依赖 current（点击前也能瞄准）+
            # 已画轨迹实时半透明预览（原实现拖动全程无预览）
            p.save()
            if self._brush_pts:
                # 宽度必须等比：w×scale（+2 补偿轮廓线）；注意 setPen 后
                # 再 setWidthF 只改局部对象无效（真机"预览不跟随缩放"根因）
                pen = QtGui.QPen(QtGui.QColor(0, 200, 255, 110),
                                 max(2.0, float(self._brush_width)
                                     * self.scale + 2.0))
                pen.setCapStyle(QtCore.Qt.RoundCap)
                pen.setJoinStyle(QtCore.Qt.RoundJoin)
                p.setPen(pen)
                p.drawPolyline([QtCore.QPointF(q.x() * self.scale,
                                               q.y() * self.scale)
                                for q in self._brush_pts])
            if self._brush_erase_pts:
                pen = QtGui.QPen(QtGui.QColor(255, 60, 60, 120),
                                 max(2.0, float(self._brush_width)
                                     * self.scale + 2.0))
                pen.setCapStyle(QtCore.Qt.RoundCap)
                pen.setJoinStyle(QtCore.Qt.RoundJoin)
                p.setPen(pen)
                p.drawPolyline([QtCore.QPointF(q.x() * self.scale,
                                               q.y() * self.scale)
                                for q in self._brush_erase_pts])
            p.setPen(QtGui.QPen(QtGui.QColor(0, 200, 255), 2))
            r = max(2.0, self._brush_width / 2.0 * self.scale + 1.0)
            p.drawEllipse(QtCore.QPointF(
                self.prevMovePoint.x() * self.scale,
                self.prevMovePoint.y() * self.scale), r, r)
            p.restore()
        if self.selectedShapesCopy:
            for s in self.selectedShapesCopy:
                s.paint(p)

        if not self.current:
            p.end()
            return

        # XCcv：绘制预览——当前标注区域蒙上一层淡灰色蒙版（矢量绘制，无额外负载）
        preview = None
        if self.createMode == "polygon" and len(self.current.points) >= 2:
            preview = self.current.copy()
            preview.addPoint(self.line[1])
        elif self.createMode in ("rectangle", "circle") and len(self.line.points) == 2:
            preview = Shape(shape_type=self.createMode)
            for pt in self.line.points:
                preview.addPoint(QtCore.QPointF(pt))
            preview.close()
        if preview is not None:
            preview.fill = True
            preview.fill_color = QtGui.QColor(210, 210, 210, 85)
            preview.paint(p)
        p.end()

    def transformPos(self, point):
        """Convert from widget-logical coordinates to painter-logical ones."""
        return point / self.scale - self.offsetToCenter()

    def offsetToCenter(self):
        s = self.scale
        area = super(Canvas, self).size()
        w, h = self.pixmap.width() * s, self.pixmap.height() * s
        aw, ah = area.width(), area.height()
        x = (aw - w) / (2 * s) if aw > w else 0
        y = (ah - h) / (2 * s) if ah > h else 0
        return QtCore.QPointF(x, y)

    def outOfPixmap(self, p):
        w, h = self.pixmap.width(), self.pixmap.height()
        return not (0 <= p.x() <= w - 1 and 0 <= p.y() <= h - 1)

    # ================= XCcv V0.6 T5-3/T5-5 =================
    def setLabelsVisible(self, value: bool):
        """T5-6：类别文字显隐开关（C）。"""
        self._show_labels = bool(value)
        self.update()

    def _brush_region_update(self, new_pos):
        """画笔圈/轨迹局部重绘（第三批 2026-11-07）。

        原实现每次移动走 repaint() 同步重绘整幅场景（大图+多形状时
        笔刷框明显滞后于原生十字光标）；改为只刷新新旧圆心与移动段
        覆盖的设备区域，异步合并重绘，笔刷框与十字光标贴合无延迟。
        """
        if not self.pixmap:
            self.update()
            return
        new_pos = QtCore.QPointF(new_pos)
        prev = getattr(self, "_brush_last_cursor", None)
        self._brush_last_cursor = QtCore.QPointF(new_pos)
        if prev is None:
            self.update()
            return
        scale = self.scale
        off = self.offsetToCenter()
        r = max(2.0, self._brush_width / 2.0 * scale + 1.0) + 6.0
        a = (prev + off) * scale
        b = (new_pos + off) * scale
        region = QtGui.QRegion(
            QtCore.QRectF(a, b).normalized().adjusted(-r, -r, r, r)
            .toAlignedRect())
        for q in (a, b):
            region += QtGui.QRegion(QtCore.QRectF(
                q.x() - r, q.y() - r, 2 * r, 2 * r).toAlignedRect())
        self.update(region)

    def setBrushWidth(self, w: int):
        import time as _t

        self._brush_width = int(max(2, min(256, w)))
        self._brush_hint_t = _t.monotonic()
        self._brush_last_cursor = None
        self.update()

    # 首锚点簇的确定性微偏移（±2px 内；重复同坐标点过多会破坏 SAM 提示
    # 编码——实测 16 同点返回全图大块，微抖动簇等价"目标上的一簇高置信
    # 正点"，SAM 原生支持良好）
    _AI_W1_OFFSETS = ((0, 0), (2, 1), (-2, 2), (1, -2),
                      (2, -2), (-2, -1), (-1, 2), (0, -2),
                      (3, 0), (-3, 0), (0, 3), (0, -3),
                      (2, 2), (-2, -2), (1, 3), (-1, -3),
                      (3, 1), (-3, 1))

    def _ai_weighted_prompts(self, prompts=None):
        """锚点加权：首锚权重最高且占绝大部分，后续锚点依次降低（2、1、1…）。

        首锚点簇大小由 `_ai_first_ratio`（两锚时首锚占比，默认 0.9 → 18:2）
        换算；挖除点（label=0）同样按位置加权。SAM 点提示无连续权重——用
        锚点邻域的微抖动点簇实现（同坐标过多重复会破坏提示编码）。
        """
        prompts = self._ai_prompts if prompts is None else prompts
        r = min(max(float(getattr(self, "_ai_first_ratio", 0.9)), 0.5), 0.9)
        n1 = max(1, min(len(self._AI_W1_OFFSETS),
                        int(round(2.0 * r / (1.0 - r)))))
        pts, labs = [], []
        for i, (q, lab) in enumerate(prompts):
            w = n1 if i == 0 else (2 if i == 1 else 1)
            for k in range(w):
                dx, dy = (self._AI_W1_OFFSETS[k % len(self._AI_W1_OFFSETS)]
                          if w > 1 else (0, 0))
                pts.append((q.x() + dx, q.y() + dy))
                labs.append(int(lab))
        return pts, labs

    def _ai_decode_prompts(self, extra_pt=None, extra_label=1):
        """当前提示点（+可选悬停点）→ 掩码（原图尺寸 bool）或 None。"""
        if not self._sam or self.pixmap is None:
            return None
        self._compute_and_cache_image_embedding()
        emb = self._sam_embedding.get(self._ai_emb_key())
        if emb is None:
            return None
        pts, labs = self._ai_weighted_prompts()
        if extra_pt is not None:
            pts = pts + [(extra_pt.x(), extra_pt.y())]
            labs = labs + [int(extra_label)]
        if not pts:
            return None
        try:
            with self._sam_lock:
                ann = self._sam.decode(
                    emb, np.asarray(pts, dtype=float), labs)
        except Exception as e:  # noqa: BLE001
            logger.warning("AI 提取失败：{}", e)
            return None
        return None if ann is None else ann.mask

    @staticmethod
    def _largest_component(mask):
        import cv2 as _cv

        n, lab, stats, _c = _cv.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8)
        if n <= 2:
            return mask
        areas = stats[1:, _cv.CC_STAT_AREA]
        keep = 1 + int(np.argmax(areas))
        return lab == keep

    def _ai_refine_current(self):
        """点提示实时提取：把（最大连通域）结果并入当前形状。"""
        self._ai_preview_shape = None
        mask = self._ai_decode_prompts()
        self._ai_refined_n = len(self._ai_prompts)   # 供 finalise 复用判定
        if mask is None or not mask.any():
            self.update()
            return
        self._ai_last_mask = mask
        if self.createMode == "ai_mask":
            ys, xs = np.nonzero(mask)
            y1, x1 = int(ys.min()), int(xs.min())
            y2, x2 = int(ys.max()), int(xs.max())
            self.current.setShapeRefined(
                shape_type="mask",
                points=[QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)],
                point_labels=[1, 1],
                mask=mask[y1:y2 + 1, x1:x2 + 1],
            )
            self.update()
            return
        from labelme._automation import polygon_from_mask

        poly = polygon_from_mask.compute_polygon_from_mask(
            mask=self._largest_component(mask))
        poly = self._simplify_poly(poly, self._ai_max_points,
                                   self._ai_tolerance)
        if len(poly) < 3:
            self.update()
            return
        self.current.setShapeRefined(
            shape_type="polygon",
            points=[QtCore.QPointF(float(x), float(y)) for x, y in poly],
            point_labels=[1] * len(poly),
        )
        self.update()

    @staticmethod
    def _simplify_poly(poly, max_points, tol):
        """顶点上限自适应简化（批注：点数可调）——容差逐步放大至达标。"""
        import cv2 as _cv

        arr = np.asarray(poly, dtype=np.float32)
        if len(arr) == 0:
            return arr
        pts = arr.reshape(-1, 1, 2)
        span = float(np.ptp(arr, axis=0).max()) or 1.0
        eps = max(0.3, span * float(tol))
        out = pts
        for _ in range(8):
            if len(out) <= max_points:
                break
            out = _cv.approxPolyDP(pts, eps, True)
            eps *= 1.5
        return out.reshape(-1, 2)

    def _ai_hover_maybe(self, pos, is_shift_pressed):
        """悬停实时预览：单线程后台解码（节流；解码中/双击判定中跳过），
        结果直接刷新现有多边形，不卡 UI。"""
        if self.current is None or self.pixmap is None \
                or not self._ai_hover_enabled or self._ai_busy \
                or self._ai_pending is not None or not self._ai_prompts:
            return
        now = time.monotonic()
        if now - self._ai_last_hover_t < self._ai_hover_interval:
            return
        self._ai_last_hover_t = now
        self._ai_request(
            self._ai_prompts
            + [(QtCore.QPointF(pos), 0 if is_shift_pressed else 1)],
            "hover")

    def _ai_hover_worker(self, sam, emb, pts, labs, max_pts, tol):
        shape = None
        try:
            with self._sam_lock:
                ann = sam.decode(emb, np.asarray(pts, dtype=float), labs)
            mask = None if ann is None else ann.mask
            if mask is not None and mask.any():
                from labelme._automation import polygon_from_mask

                poly = polygon_from_mask.compute_polygon_from_mask(
                    mask=self._largest_component(mask))
                poly = self._simplify_poly(poly, max_pts, tol)
                if len(poly) >= 3:
                    shape = Shape(shape_type="polygon")
                    for x, y in poly:
                        shape.addPoint(QtCore.QPointF(float(x), float(y)))
        except Exception:  # noqa: BLE001
            shape = None
        self.aiHoverReady.emit(shape)

    def _ai_decode_async(self):
        """已提交锚点异步解码（兼容入口）。"""
        self._ai_request(self._ai_prompts, "refine")

    # ---- 智能多边形：先判双击再建锚点 ----
    def _ai_clamp(self, pos):
        if self.pixmap is None or self.pixmap.isNull():
            return QtCore.QPointF(pos)
        w, h = self.pixmap.width(), self.pixmap.height()
        return QtCore.QPointF(min(max(pos.x(), 0.0), w - 1.0),
                              min(max(pos.y(), 0.0), h - 1.0))

    def _ai_reset_interaction(self):
        """清空智能多边形交互态（换图/换模式/退出标注）。"""
        timer = getattr(self, "_ai_click_timer", None)
        if timer is not None:
            timer.stop()
        self._ai_pending = None
        self._ai_finalise_wanted = False
        self._ai_queue = None
        self._ai_last_commit = None
        self._ai_last_finalised = None

    def _ai_add_anchor(self, pt, lab, t_click=None):
        self._ai_prompts.append((QtCore.QPointF(pt), int(lab)))
        self._ai_last_commit = (time.monotonic() if t_click is None
                                else t_click, len(self._ai_prompts))
        self._ai_refined_n = -1
        if self._ai_embedding() is None and self._sam is not None:
            # 首次点击嵌入未就绪：明确告知（第四批，防用户无反馈重复点击）
            self.aiStatus.emit("正在计算图像嵌入（首次约数秒），请稍候…")
        self._ai_request(self._ai_prompts, "refine")
        self.update()

    def _ai_click(self, pt, lab, local_pos, modifiers):
        """已有锚点时的左键：系统双击间隔内不成锚点（空心圈待定），
        确认不是双击后才入栈；待定期间后台预解，确认后结果即时上屏。"""
        if modifiers & QtCore.Qt.ControlModifier:
            self._ai_commit_pending()
            self._ai_add_anchor(pt, lab)
            self.finalise()
            return
        pend = self._ai_pending
        if pend is not None and self._ai_click_timer.isActive():
            dist = QtGui.QGuiApplication.styleHints() \
                .mouseDoubleClickDistance()
            if (local_pos - pend[2]).manhattanLength() <= max(4, dist) * 2:
                # 判定窗口内同处第二次按下=双击（Qt 未合成双击事件时兜底）
                self._ai_double_click_close()
                return
            self._ai_commit_pending()   # 远处第二击：前一击按单击生效
        self._ai_pending = (QtCore.QPointF(pt), int(lab),
                            QtCore.QPointF(local_pos), time.monotonic())
        # 第三批：双击判定统一 250ms（系统默认 400ms 偏慢，闭合更跟手）
        self._ai_click_timer.start(250)
        self._ai_request(self._ai_prompts + [(pt, lab)], "spec")
        self.update()

    def _ai_commit_pending(self):
        self._ai_click_timer.stop()
        pend, self._ai_pending = self._ai_pending, None
        if pend is None or self.current is None or not self.drawing() \
                or self.createMode not in ("ai_polygon", "ai_mask"):
            return
        self._ai_add_anchor(pend[0], pend[1], t_click=pend[3])

    def _ai_double_click_close(self):
        """双击闭合：双击的第一击（待定点击）丢弃，按已提交锚点完成。"""
        pend = self._ai_pending
        self._ai_click_timer.stop()
        self._ai_pending = None
        if pend is None:
            lc = self._ai_last_commit
            if lc is not None and len(self._ai_prompts) == lc[1] > 1 \
                    and (time.monotonic() - lc[0]) * 1000.0 <= 250.0:
                # 兜底：第一击恰在判定窗口边缘已提交——回滚（结果缓存命中）
                self._ai_prompts.pop()
        self._ai_last_commit = None
        self.finalise()

    # ---- 智能多边形解码流水线（单工作线程 + 提示键缓存）----
    def _ai_key(self, prompts):
        return (self._ai_gen,) + tuple(
            (round(q.x(), 1), round(q.y(), 1), int(lab))
            for q, lab in prompts)

    def _ai_invalidate(self):
        """换图/换模型/改参数：旧解码结果作废。"""
        self._ai_gen += 1
        self._ai_results.clear()
        self._ai_refined_n = -1

    def _ai_request(self, prompts, kind):
        """kind：refine=已提交锚点 / spec=待定点击预解 / hover=悬停预览。"""
        if not prompts or self._sam is None or self.pixmap is None:
            return
        key = self._ai_key(prompts)
        if key in self._ai_results:
            self._ai_results.move_to_end(key)
            self._ai_deliver(key, kind, self._ai_results[key])
            return
        if self._ai_inflight == key:
            return
        job = (kind, [(QtCore.QPointF(q), int(lab)) for q, lab in prompts],
               key)
        if self._ai_busy or self._ai_embedding() is None:
            if kind == "hover" and self._ai_queue is not None \
                    and self._ai_queue[0] != "hover":
                return   # 悬停不顶掉排队中的锚点解码
            self._ai_queue = job
            return
        self._ai_start(job)

    def _ai_start(self, job):
        kind, prompts, key = job
        emb = self._ai_embedding()
        if emb is None:
            self._ai_queue = job
            return
        pts, labs = self._ai_weighted_prompts(prompts)
        self._ai_busy = True
        self._ai_inflight = key
        args = (self._sam, emb, pts, labs, self._ai_max_points,
                self._ai_tolerance, self._ai_largest_only, key, kind)
        threading.Thread(target=self._ai_decode_worker, args=args,
                         daemon=True).start()

    def _ai_decode_worker(self, sam, emb, pts, labs, max_pts, tol, largest,
                          key, kind):
        poly = None
        try:
            with self._sam_lock:
                ann = sam.decode(emb, np.asarray(pts, dtype=float), labs)
            mask = None if ann is None else ann.mask
            if mask is not None and mask.any():
                m = self._largest_component(mask) if largest else mask
                arr = polygon_from_mask.compute_polygon_from_mask(mask=m)
                arr = self._simplify_poly(arr, max_pts, tol)
                if len(arr) >= 3:
                    poly = [(float(x), float(y)) for x, y in arr]
        except Exception as e:  # noqa: BLE001
            logger.warning("AI 解码失败：{}", e)
            poly = None
        self.aiDecodeDone.emit(key, kind, poly)

    def _ai_on_decode_done(self, key, kind, poly):
        self._ai_busy = False
        if self._ai_inflight == key:
            self._ai_inflight = None
        if key and key[0] == self._ai_gen:
            if kind != "hover":
                self._ai_results[key] = poly
                while len(self._ai_results) > 48:
                    self._ai_results.popitem(last=False)
            self._ai_deliver(key, kind, poly)
        self._ai_pump()

    def _ai_deliver(self, key, kind, poly):
        if self.current is None or not self.drawing() \
                or self.createMode not in ("ai_polygon", "ai_mask"):
            return
        committed = self._ai_key(self._ai_prompts)
        if key == committed:
            if poly is not None:
                self._ai_set_current_points(poly)
            self._ai_refined_n = len(self._ai_prompts)
            self.update()
        elif kind == "hover" and poly is not None \
                and self._ai_pending is None and key[:-1] == committed:
            # 悬停预览直接刷新现有多边形（不另建淡色副本）
            self._ai_set_current_points(poly)
            self.update()

    def _ai_pump(self):
        if self._ai_busy:
            return
        if self._ai_finalise_wanted:
            if self._ai_key(self._ai_prompts) in self._ai_results:
                self._ai_finalise_wanted = False
                self.finalise()
                return
            self._ai_queue = None
            self._ai_request(self._ai_prompts, "refine")
            return
        job, self._ai_queue = self._ai_queue, None
        if job is not None:
            self._ai_request(job[1], job[0])
        elif self.current is not None and self._ai_prompts \
                and self.drawing() \
                and self.createMode in ("ai_polygon", "ai_mask") \
                and self._ai_key(self._ai_prompts) not in self._ai_results:
            self._ai_request(self._ai_prompts, "refine")

    def _ai_set_current_points(self, poly):
        self.current.setShapeRefined(
            shape_type="polygon",
            points=[QtCore.QPointF(x, y) for x, y in poly],
            point_labels=[1] * len(poly),
        )

    def _ai_refine_worker(self, sam, emb, pts, labs, mp, tol, n):
        shape = None
        try:
            with self._sam_lock:
                ann = sam.decode(emb, np.asarray(pts, dtype=float), labs)
            mask = None if ann is None else ann.mask
            if mask is not None and mask.any():
                from labelme._automation import polygon_from_mask

                poly = polygon_from_mask.compute_polygon_from_mask(
                    mask=self._largest_component(mask))
                poly = self._simplify_poly(poly, mp, tol)
                if len(poly) >= 3:
                    shape = Shape(shape_type="polygon")
                    for x, y in poly:
                        shape.addPoint(QtCore.QPointF(float(x), float(y)))
        except Exception:  # noqa: BLE001
            shape = None
        self.aiHoverReady.emit(shape)
        self._ai_pending_n = n

    def _ai_hover_apply(self, shape):
        self._ai_hover_busy = False
        self._ai_preview_shape = None   # 淡色副本已移除（用户反馈：
        # 悬停/点击预测直接刷新现有多边形，只显示一个多边形）
        if shape is not None and self.current is not None \
                and self.drawing() \
                and self.createMode in ("ai_polygon", "ai_mask"):
            self.current.setShapeRefined(
                shape_type="polygon",
                points=list(shape.points),
                point_labels=[1] * len(shape.points),
            )
            if getattr(self, "_ai_pending_n", None) is not None:
                self._ai_refined_n = self._ai_pending_n
                self._ai_pending_n = None
        self.update()

    def _brush_raster(self, points, h, w, value=255):
        """笔画轨迹栅格化（图像坐标系）。"""
        import cv2 as _cv

        m = np.zeros((h, w), np.uint8)
        thickness = max(2, int(self._brush_width))
        if len(points) == 1:
            _cv.circle(m, (int(points[0].x()), int(points[0].y())),
                       thickness // 2, value, -1)
        elif len(points) >= 2:
            path = np.array([[int(q.x()), int(q.y())] for q in points],
                            dtype=np.int32).reshape(-1, 1, 2)
            _cv.polylines(m, [path], False, value, thickness)
            _cv.circle(m, (int(points[-1].x()), int(points[-1].y())),
                       thickness // 2, value, -1)
        return m

    def _shape_raster(self, shape, h, w):
        """形状栅格化（不支持的类型返回 None）。"""
        import cv2 as _cv

        st = shape.shape_type
        if st == "mask" and getattr(shape, "mask", None) is not None \
                and len(shape.points) >= 2:
            m = np.zeros((h, w), np.uint8)
            x1 = int(shape.points[0].x())
            y1 = int(shape.points[0].y())
            x2 = int(shape.points[1].x())
            y2 = int(shape.points[1].y())
            m[y1:y2 + 1, x1:x2 + 1] = \
                np.asarray(shape.mask, dtype=np.uint8) * 255
            return m
        if st not in ("polygon", "rectangle"):
            return None
        m = np.zeros((h, w), np.uint8)
        pts = np.array([[int(q.x()), int(q.y())] for q in shape.points],
                       dtype=np.int32)
        if st == "rectangle":
            x1, y1 = int(pts[0][0]), int(pts[0][1])
            x2, y2 = int(pts[1][0]), int(pts[1][1])
            _cv.rectangle(m, (min(x1, x2), min(y1, y2)),
                          (max(x1, x2), max(y1, y2)), 255, -1)
        else:
            _cv.fillPoly(m, [pts.reshape(-1, 1, 2)], 255)
        return m

    def _mask_to_shapes(self, mask):
        """掩膜 -> 多边形形状列表（每个外部连通域一个，含简化）。"""
        import cv2 as _cv

        contours, _ = _cv.findContours(mask, _cv.RETR_EXTERNAL,
                                       _cv.CHAIN_APPROX_SIMPLE)
        shapes = []
        for c in sorted(contours, key=_cv.contourArea, reverse=True):
            if _cv.contourArea(c) < 4.0:
                continue
            eps = max(1.5, 0.004 * _cv.arcLength(c, True))
            poly = _cv.approxPolyDP(c, eps, True).reshape(-1, 2)
            if len(poly) < 3:
                x, y, ww, hh = _cv.boundingRect(c)
                poly = np.array([[x, y], [x + ww, y],
                                 [x + ww, y + hh], [x, y + hh]])
            ns = Shape(shape_type="polygon")
            for x, y in poly:
                ns.addPoint(QtCore.QPointF(float(x), float(y)))
            ns.close()
            shapes.append(ns)
        return shapes

    def _mask_apply_shape(self, mask, shape):
        """掩膜原位更新形状（主连通域），其余连通域返回为新形状；
        空掩膜删除形状。"""
        shapes = self._mask_to_shapes(mask)
        if not shapes:
            if shape in self.shapes:
                self.shapes.remove(shape)
            if shape in self.selectedShapes:
                self.selectedShapes.remove(shape)
            return []
        main, extra = shapes[0], shapes[1:]
        shape.shape_type = "polygon"
        shape.points = list(main.points)
        shape.point_labels = [1] * len(shape.points)
        shape.close()
        return extra

    def _brush_finalise(self):
        """画笔塑形：拖动笔画并入相交形状或新建；擦除笔画从相交形状挖除。"""
        pts = getattr(self, "_brush_pts", []) or []
        eps_pts = getattr(self, "_brush_erase_pts", []) or []
        w, h = self.pixmap.width(), self.pixmap.height()
        stroke = self._brush_raster(pts, h, w) if pts \
            else np.zeros((h, w), np.uint8)
        erase = self._brush_raster(eps_pts, h, w) if eps_pts \
            else np.zeros((h, w), np.uint8)
        self._brush_pts = []
        self._brush_erase_pts = []
        self._brush_erasing = False
        created_all = []
        touched = False
        if erase.any():
            for shape in list(self.shapes):
                sm = self._shape_raster(shape, h, w)
                if sm is None or not (sm & erase).any():
                    continue
                touched = True
                created_all += self._mask_apply_shape(sm & ~erase, shape)
        if stroke.any():
            target = None
            for shape in self.shapes:
                sm = self._shape_raster(shape, h, w)
                if sm is not None and (sm & stroke).any():
                    target = shape
                    break
            if target is not None:
                touched = True
                sm = self._shape_raster(target, h, w) | stroke
                if erase.any():
                    sm = sm & ~erase
                created_all += self._mask_apply_shape(sm, target)
            else:
                new_shapes = self._mask_to_shapes(stroke & ~erase)
                if new_shapes:
                    touched = True
                    created_all += new_shapes
        if touched:
            self.shapes.extend(created_all)
            self.storeShapes()
            if created_all:
                self.newShape.emit()
        if self.current is not None:
            self.current = None
            self.setHiding(False)
            self.drawingPolygon.emit(False)
        self.update()

    def finalise(self):
        if self.current is None:
            # V0.6 修复循环 4：双重 release/模态嵌套下 current 已被上一次
            # finalise 消费——静默返回（原 assert 会弹错误框中断用户操作）
            return
        if self.createMode == "brush":
            self._brush_finalise()
            return
        if self.createMode in ["ai_polygon", "ai_mask"] \
                and (self._ai_prompts or self._ai_pending is not None):
            if self._ai_pending is not None:
                # 回车/Ctrl 完成：待定点击按有效锚点计入（双击路径已先丢弃）
                pend, self._ai_pending = self._ai_pending, None
                self._ai_click_timer.stop()
                self._ai_prompts.append((pend[0], pend[1]))
            if self.createMode == "ai_mask":
                if self._sam and getattr(self, "_ai_refined_n", -1) \
                        != len(self._ai_prompts):
                    self._ai_refine_current()
            else:
                # 闭合=已提交锚点的解码结果（所见即所得；不再以多边形顶点
                # 作提示二次同步重解——双击闭合 1~2s 卡顿根因）
                key = self._ai_key(self._ai_prompts)
                if key not in self._ai_results:
                    if self._sam is None:
                        self.aiStatus.emit("AI 模型未就绪，无法完成提取")
                        return
                    self._ai_finalise_wanted = True   # 结果到达后自动完成
                    self.aiStatus.emit("正在完成提取…")
                    self._ai_pump()
                    return
                poly = self._ai_results[key]
                if poly is None or len(poly) < 3:
                    self._ai_finalise_wanted = False
                    self.aiStatus.emit("未提取到目标：请调整锚点（左键并入 / "
                                       "Shift+左键挖除 / 右键撤销）")
                    return
                self._ai_set_current_points(poly)
            self._ai_last_finalised = list(self._ai_prompts)
            self._ai_prompts = []
            self._ai_finalise_wanted = False
            self._ai_refined_n = -1
        self.current.close()

        self.shapes.append(self.current)
        self.storeShapes()
        self.current = None
        self.setHiding(False)
        self.newShape.emit()
        self.update()

    def closeEnough(self, p1, p2):
        # d = distance(p1 - p2)
        # m = (p1-p2).manhattanLength()
        # print "d %.2f, m %d, %.2f" % (d, m, d - m)
        # divide by scale to allow more precision when zoomed in
        return labelme.utils.distance(p1 - p2) < (self.epsilon / self.scale)

    def intersectionPoint(self, p1, p2):
        # Cycle through each image edge in clockwise fashion,
        # and find the one intersecting the current line segment.
        # http://paulbourke.net/geometry/lineline2d/
        size = self.pixmap.size()
        points = [
            (0, 0),
            (size.width() - 1, 0),
            (size.width() - 1, size.height() - 1),
            (0, size.height() - 1),
        ]
        # x1, y1 should be in the pixmap, x2, y2 should be out of the pixmap
        x1 = min(max(p1.x(), 0), size.width() - 1)
        y1 = min(max(p1.y(), 0), size.height() - 1)
        x2, y2 = p2.x(), p2.y()
        d, i, (x, y) = min(self.intersectingEdges((x1, y1), (x2, y2), points))
        x3, y3 = points[i]
        x4, y4 = points[(i + 1) % 4]
        if (x, y) == (x1, y1):
            # Handle cases where previous point is on one of the edges.
            if x3 == x4:
                return QtCore.QPointF(x3, min(max(0, y2), max(y3, y4)))
            else:  # y3 == y4
                return QtCore.QPointF(min(max(0, x2), max(x3, x4)), y3)
        return QtCore.QPointF(x, y)

    def intersectingEdges(self, point1, point2, points):
        """Find intersecting edges.

        For each edge formed by `points', yield the intersection
        with the line segment `(x1,y1) - (x2,y2)`, if it exists.
        Also return the distance of `(x2,y2)' to the middle of the
        edge along with its index, so that the one closest can be chosen.
        """
        (x1, y1) = point1
        (x2, y2) = point2
        for i in range(4):
            x3, y3 = points[i]
            x4, y4 = points[(i + 1) % 4]
            denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            nua = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
            nub = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
            if denom == 0:
                # This covers two cases:
                #   nua == nub == 0: Coincident
                #   otherwise: Parallel
                continue
            ua, ub = nua / denom, nub / denom
            if 0 <= ua <= 1 and 0 <= ub <= 1:
                x = x1 + ua * (x2 - x1)
                y = y1 + ua * (y2 - y1)
                m = QtCore.QPointF((x3 + x4) / 2, (y3 + y4) / 2)
                d = labelme.utils.distance(m - QtCore.QPointF(x2, y2))
                yield d, i, (x, y)

    # These two, along with a call to adjustSize are required for the
    # scroll area.
    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        if self.pixmap:
            return self.scale * self.pixmap.size()
        return super(Canvas, self).minimumSizeHint()

    def _exit_drawing(self):
        """XCcv：退出标注状态。有 on_exit_drawing 回调时走主窗口标准切换
        （恢复全部绘制动作的可用状态），否则退回编辑模式。"""
        self.current = None
        self._ai_prompts = []
        self._ai_reset_interaction()
        self.drawingPolygon.emit(False)
        if self.on_exit_drawing is not None:
            self.on_exit_drawing()
        else:
            self.setEditing(True)
        self.update()

    def _scroll_area(self):
        p = self.parent()
        while p is not None:
            if isinstance(p, QtWidgets.QAbstractScrollArea):
                return p
            p = p.parent()
        return None

    def _shape_at(self, pos):
        for shape in reversed(self.shapes):
            if self.isVisible(shape) and shape.containsPoint(pos):
                return shape
        return None

    def _begin_pan(self, ev, button):
        area = self._scroll_area()
        if area is None:
            return False
        self._pan_scroll_area = area
        self._pan_start_global = ev.globalPos()
        self._pan_h0 = area.horizontalScrollBar().value()
        self._pan_v0 = area.verticalScrollBar().value()
        self._pan_button = button
        self._panning = True
        self.overrideCursor(QtCore.Qt.ClosedHandCursor)
        return True

    def _do_pan(self, ev):
        dx = ev.globalPos().x() - self._pan_start_global.x()
        dy = ev.globalPos().y() - self._pan_start_global.y()
        self._pan_scroll_area.horizontalScrollBar().setValue(self._pan_h0 - dx)
        self._pan_scroll_area.verticalScrollBar().setValue(self._pan_v0 - dy)

    def _end_pan(self):
        self._panning = False
        self._pan_button = None
        self.restoreCursor()
        self.update()

    def wheelEvent(self, ev):
        # 画笔粗细改由 [ ]（用户反馈 2026-11-06）；滚轮恒为图像缩放
        # XCcv：滚轮直接以鼠标为中心缩放；Shift+滚轮横向滚动
        delta = ev.angleDelta()
        if ev.modifiers() & QtCore.Qt.ShiftModifier:
            self.scrollRequest.emit(delta.y(), QtCore.Qt.Horizontal)
        else:
            self.zoomRequest.emit(delta.y(), ev.pos())
        ev.accept()

    def moveByKeyboard(self, offset):
        if self.selectedShapes:
            self.boundedMoveShapes(self.selectedShapes, self.prevPoint + offset)
            self.repaint()
            self.movingShape = True

    def keyPressEvent(self, ev):
        # XCcv V0.6 T5-5：画笔模式下 [ ] 调节笔刷粗细
        if self.drawing() and self.createMode == "brush":
            if ev.key() == QtCore.Qt.Key_BracketLeft:
                self.setBrushWidth(self._brush_width - 4)
                return
            if ev.key() == QtCore.Qt.Key_BracketRight:
                self.setBrushWidth(self._brush_width + 4)
                return
        modifiers = ev.modifiers()
        key = ev.key()
        if self.drawing():
            if key == QtCore.Qt.Key_Escape:
                # XCcv：ESC 取消当前形状并退出标注状态（走标准切换流程）
                self._exit_drawing()
            elif key == QtCore.Qt.Key_Return and self.canCloseShape():
                self.finalise()
            elif modifiers == QtCore.Qt.AltModifier:
                self.snapping = False
        elif self.editing():
            if key == QtCore.Qt.Key_Up:
                self.moveByKeyboard(QtCore.QPointF(0.0, -MOVE_SPEED))
            elif key == QtCore.Qt.Key_Down:
                self.moveByKeyboard(QtCore.QPointF(0.0, MOVE_SPEED))
            elif key == QtCore.Qt.Key_Left:
                self.moveByKeyboard(QtCore.QPointF(-MOVE_SPEED, 0.0))
            elif key == QtCore.Qt.Key_Right:
                self.moveByKeyboard(QtCore.QPointF(MOVE_SPEED, 0.0))

    def keyReleaseEvent(self, ev):
        modifiers = ev.modifiers()
        if self.drawing():
            if int(modifiers) == 0:
                self.snapping = True
        elif self.editing():
            if self.movingShape and self.selectedShapes:
                index = self.shapes.index(self.selectedShapes[0])
                if self.shapesBackups[-1][index].points != self.shapes[index].points:
                    self.storeShapes()
                    self.shapeMoved.emit()

                self.movingShape = False

    def setLastLabel(self, text, flags):
        assert text
        self.shapes[-1].label = text
        self.shapes[-1].flags = flags
        self.shapesBackups.pop()
        self.storeShapes()
        return self.shapes[-1]

    def undoLastLine(self):
        assert self.shapes
        self.current = self.shapes.pop()
        self.current.setOpen()
        self.current.restoreShapeRaw()
        if self.createMode in ["polygon", "linestrip"]:
            self.line.points = [self.current[-1], self.current[0]]
        elif self.createMode in ["rectangle", "line", "circle"]:
            self.current.points = self.current.points[0:1]
        elif self.createMode == "point":
            self.current = None
        elif self.createMode in ["ai_polygon", "ai_mask"] \
                and self._ai_last_finalised:
            # 取消标签框：恢复锚点与已提交结果，可继续增删锚点
            self._ai_prompts = list(self._ai_last_finalised)
            self._ai_last_finalised = None
            poly = self._ai_results.get(self._ai_key(self._ai_prompts))
            if poly is not None and len(poly) >= 3:
                self._ai_set_current_points(poly)
        self.drawingPolygon.emit(True)

    def undoLastPoint(self):
        # 注意：空 Shape（0 顶点）因 __len__==0 为假值——本方法先按身份判空
        # 再走 ai/brush 分支，最后才用真值判断（历史行为）
        if self.createMode in ["ai_polygon", "ai_mask"] \
                and self.current is not None \
                and (self._ai_prompts or self._ai_pending is not None):
            self._ai_finalise_wanted = False
            if self._ai_pending is not None:
                self._ai_click_timer.stop()
                self._ai_pending = None    # 撤销尚未生效的待定点击
            else:
                self._ai_prompts.pop()
            self._ai_preview_shape = None
            if not self._ai_prompts:
                self.current = None
                self._ai_last_mask = None
                self.drawingPolygon.emit(False)
            else:
                self._ai_refined_n = -1
                self._ai_request(self._ai_prompts, "refine")  # 缓存命中=瞬时
            self.update()
            return
        if self.createMode == "brush":
            self._brush_pts = []
            self.current = None
            self.drawingPolygon.emit(False)
            self.update()
            return
        if not self.current or self.current.isClosed():
            return
        self.current.popPoint()
        if len(self.current) > 0:
            self.line[0] = self.current[-1]
        else:
            self.current = None
            self.drawingPolygon.emit(False)
        self.update()

    def loadPixmap(self, pixmap, clear_shapes=True):
        self.pixmap = pixmap
        self._px_img = pixmap.toImage()
        # 切换图片清 AI 交互残留（用户反馈 2026-11-06：锚点跨图留存）
        self._ai_prompts = []
        self._ai_last_mask = None
        self._ai_refined_n = -1
        self._ai_preview_shape = None
        self._ai_reset_interaction()
        self._ai_invalidate()
        if self.current is not None and \
                self.createMode in ("ai_polygon", "ai_mask"):
            self.current = None
            self.drawingPolygon.emit(False)
        if self._sam:
            self._preload_embedding_async()
        if clear_shapes:
            self.shapes = []
        self.update()

    def loadShapes(self, shapes, replace=True):
        if replace:
            self.shapes = list(shapes)
        else:
            self.shapes.extend(shapes)
        self.storeShapes()
        self.current = None
        self.hShape = None
        self.hVertex = None
        self.hEdge = None
        self.update()

    def setShapeVisible(self, shape, value):
        self.visible[shape] = value
        self.update()

    def overrideCursor(self, cursor):
        self.restoreCursor()
        self._cursor = cursor
        QtWidgets.QApplication.setOverrideCursor(cursor)

    def restoreCursor(self):
        QtWidgets.QApplication.restoreOverrideCursor()

    def resetState(self):
        self.restoreCursor()
        self.pixmap = None
        self.shapesBackups = []
        self.update()


def _update_shape_with_sam(
    shape: Shape,
    createMode: str,
    model_name: str,
    image_embedding: osam.types.ImageEmbedding,
) -> None:
    if createMode not in ["ai_polygon", "ai_mask"]:
        raise ValueError(
            f"createMode must be 'ai_polygon' or 'ai_mask', not {createMode}"
        )

    response: osam.types.GenerateResponse = osam.apis.generate(
        osam.types.GenerateRequest(
            model=model_name,
            image_embedding=image_embedding,
            prompt=osam.types.Prompt(
                points=[[point.x(), point.y()] for point in shape.points],
                point_labels=shape.point_labels,
            ),
        )
    )
    if not response.annotations:
        logger.warning("No annotations returned by model {!r}", model_name)
        return

    if createMode == "ai_mask":
        y1: int
        x1: int
        y2: int
        x2: int
        if response.annotations[0].bounding_box is None:
            y1, x1, y2, x2 = imgviz.instances.mask_to_bbox(
                [response.annotations[0].mask]
            )[0].astype(int)
        else:
            y1 = response.annotations[0].bounding_box.ymin
            x1 = response.annotations[0].bounding_box.xmin
            y2 = response.annotations[0].bounding_box.ymax
            x2 = response.annotations[0].bounding_box.xmax
        shape.setShapeRefined(
            shape_type="mask",
            points=[QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)],
            point_labels=[1, 1],
            mask=response.annotations[0].mask[y1 : y2 + 1, x1 : x2 + 1],
        )
    elif createMode == "ai_polygon":
        points = polygon_from_mask.compute_polygon_from_mask(
            mask=response.annotations[0].mask
        )
        if len(points) < 2:
            return
        # 精化守卫（2026-11-05 测试批）：重解结果须与当前形状显著重叠，
        # 否则视为 SAM 误判、保留点击时已实时精化的多边形——以多顶点
        # 作提示点重解时，邻近大结构会吞掉结果（实测拱形缺陷处
        # 1.7k px² 小目标被换成 142k px² 拱形区域）
        _o_xs = [p.x() for p in shape.points]
        _o_ys = [p.y() for p in shape.points]
        _n_xs = points[:, 0]
        _n_ys = points[:, 1]
        _ix1 = max(min(_o_xs), float(_n_xs.min()))
        _iy1 = max(min(_o_ys), float(_n_ys.min()))
        _ix2 = min(max(_o_xs), float(_n_xs.max()))
        _iy2 = min(max(_o_ys), float(_n_ys.max()))
        _inter = max(0.0, _ix2 - _ix1) * max(0.0, _iy2 - _iy1)
        _union = ((max(_o_xs) - min(_o_xs)) * (max(_o_ys) - min(_o_ys))
                  + (float(_n_xs.max()) - float(_n_xs.min()))
                  * (float(_n_ys.max()) - float(_n_ys.min()))
                  - _inter)
        _iou = _inter / _union if _union > 1e-9 else 0.0
        if _iou < 0.3:
            logger.warning(
                "SAM 精化结果与当前形状偏离（bbox IoU={:.2f}），保留原多边形",
                _iou)
            return
        shape.setShapeRefined(
            shape_type="polygon",
            points=[QtCore.QPointF(point[0], point[1]) for point in points],
            point_labels=[1] * len(points),
        )
