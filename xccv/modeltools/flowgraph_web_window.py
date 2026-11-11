"""流程图编辑器（Web 版，LiteGraph 风格交互）。

- QtWebEngine 承载 web/index.html：左侧节点库拖拽、LiteGraph 画布、
  右侧属性面板（模型文件浏览、运行预览勾选）；
- 桥接（路线 B）：JS ↔ Python 走 QWebChannel（backend.postMessage 槽上行、
  eventArrived 信号下行），无字符串转义与轮询损耗；页面就绪前 Python 事件
  缓冲，通道握手（js_ready）后统一下发。旧 console.log("XCCV:") 入口保留
  兜底（通道未就绪时 JS 自动降级）；
- 新建流程：用户选择文件夹并命名，保存为 .json；不绑定训练平台项目；
- 打包 .xvap：按当前流程图 JSON 把所选模型文件一并封入（单文件）；
- 运行预览：按连线拓扑执行到所选节点，检出结果画在样图上回显。
"""

import json
import logging
import os
import tempfile
from pathlib import Path

# QtWebEngine 必须在 QApplication 创建前导入/初始化；
# 禁用 GPU 进程与沙箱，规避工控机 WDDM 下的初始化崩溃
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--no-sandbox --disable-gpu")
from PyQt5.QtCore import Qt as _Qt
from PyQt5.QtWidgets import QApplication as _QApp

if _QApp.instance() is None:
    _QApp.setAttribute(_Qt.AA_ShareOpenGLContexts, True)
from PyQt5.QtWebEngineWidgets import QWebEnginePage, QWebEngineView  # noqa: F401,E402
from PyQt5.QtWebChannel import QWebChannel  # noqa: E402

from PyQt5.QtCore import QObject, Qt, QUrl, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QMainWindow, QMessageBox,
)

from ..engine.detection_flow import graph_to_chain  # noqa: F401
from ..engine import detection_flow as df
from ..engine import flow_package as fp
from ..ui_style import icon_path

log = logging.getLogger(__name__)
_WEB_DIR = Path(__file__).resolve().parent / "web"


class _Bridge(QObject):
    """JS 消息分发：所有网页→本地操作都经这里。

    postMessage(str)：QWebChannel 上行槽（裸 JSON）；
    message(str)：旧 console.log("XCCV:"+json) 兼容入口（亦供测试直调）；
    eventArrived(str)：Python→JS 下行信号（裸 JSON，JS 侧 JSON.parse）。
    """

    eventArrived = pyqtSignal(str)

    def __init__(self, window: "FlowGraphWebWindow"):
        super().__init__()
        self.win = window

    def send(self, payload: dict):
        if payload.get("event") == "loaded" and isinstance(payload.get("flow"), dict):
            # 画布载入统一入口：旧端口布局连线先迁移到当前布局（修复循环 3），
            # 否则画布按槽位画线会接错端口（port_migrate 幂等，schema:2 跳过）
            from ..engine.flow.port_migrate import migrate_ports

            payload = {**payload, "flow": migrate_ports(payload["flow"])}
        self.win.send_json(payload)

    @pyqtSlot(str)
    def postMessage(self, text: str):
        self._dispatch(text[5:] if text.startswith("XCCV:") else text)

    @pyqtSlot(str)
    def message(self, text: str):
        if not text.startswith("XCCV:"):
            return
        self._dispatch(text[5:])

    def _dispatch(self, raw: str):
        try:
            msg = json.loads(raw)
        except Exception:
            return
        action = msg.get("action")
        try:
            handler = getattr(self, f"do_{action}", None)
            if handler:
                handler(msg)
        except Exception as e:
            log.error("流程图操作失败: %s", e, exc_info=True)
            self.send({"event": "status", "text": f"操作失败: {e}"})

    # ---- 各操作 ----
    def do_js_ready(self, msg):
        """JS 侧 QWebChannel 就绪握手。

        顺序关键（用户实测红色 undefined 节点的根因）：必须**先下发节点库
        （注册 xccv2/ 类型）再冲刷缓冲事件**——外部在通道就绪前推入的
        loaded 事件（主界面编辑流程图/流程图页同步画布）若先于节点库执行，
        graph.configure 会遇到未注册类型，渲染成红色 undefined 占位节点。
        """
        self.win._js_ready = True
        pending, self.win._pending_js = self.win._pending_js, []
        from .flow_session import node_library_payload

        self.send(node_library_payload())
        for data in pending:
            self.bridge_event(data)

    def bridge_event(self, data: str):
        self.eventArrived.emit(data)

    def do_get_node_library(self, msg):
        """JS 侧主动请求节点库（loadFlow 发现类型未注册时的重试防线）。"""
        from .flow_session import node_library_payload

        self.send(node_library_payload())

    def do_new(self, msg):
        path, _ = QFileDialog.getSaveFileName(
            self.win, "新建流程图（选择保存位置并命名）", "flow.json", "流程图 (*.json)")
        if not path:
            return
        empty = {"last_node_id": 0, "last_link_id": 0, "nodes": [], "links": []}
        Path(path).write_text(json.dumps(empty, ensure_ascii=False, indent=1),
                              encoding="utf-8")
        self.win.current_path = Path(path)
        self.send({"event": "loaded", "flow": empty, "name": Path(path).name})

    def do_new_from_template(self, msg):
        """从官方模板创建：选模板 → 载入画布（不绑定保存路径）。"""
        from PyQt5.QtWidgets import QInputDialog

        from .flow_session import list_templates

        templates = list_templates()
        if not templates:
            QMessageBox.information(self.win, "从模板创建", "未找到官方模板文件")
            return
        titles = [t["title"] for t in templates]
        choice, ok = QInputDialog.getItem(
            self.win, "从模板创建", "选择官方模板：", titles, 0, False)
        if not ok:
            return
        t = templates[titles.index(choice)]
        self.win.current_path = None
        self.send({"event": "loaded", "flow": t["flow"],
                   "name": f"{t['title']}（模板）"})
        self.send({"event": "status",
                   "text": f"已载入模板「{t['title']}」：请在节点属性中填写图片/模型路径"
                           f"——{t['description']}"})

    def do_open(self, msg):
        path, _ = QFileDialog.getOpenFileName(
            self.win, "打开流程", "", "流程包/流程图 (*.xvap *.json)")
        if not path:
            return
        path = Path(path)
        if path.suffix == ".xvap":
            # 解包：内嵌模型写「流程模型/」+ 模型引用键改写为绝对路径
            # （新旧键 model_file/model_path 以 flow_package.MODEL_PARAM_KEYS 为准）
            flow, _model_dir = fp.unpack_to_dir(path)
        else:
            flow = json.loads(path.read_text(encoding="utf-8"))
        self.win.current_path = path
        self.send({"event": "loaded", "flow": flow, "name": path.name})

    def do_save_json(self, msg):
        flow = msg.get("flow") or {}
        path = self.win.current_path
        if path is None or path.suffix == ".xvap":
            path, _ = QFileDialog.getSaveFileName(
                self.win, "保存流程图", "flow.json", "流程图 (*.json)")
            if not path:
                return
        Path(path).write_text(json.dumps(flow, ensure_ascii=False, indent=1),
                              encoding="utf-8")
        self.win.current_path = Path(path)
        self.send({"event": "status", "text": f"已保存 {Path(path).name}"})

    def do_save_xvap(self, msg):
        flow = msg.get("flow") or {}
        from .flow_pack import _show_pack_problems, validate_flow_for_pack

        # 画布打包与 Qt 侧打包走同一套打包前校验（此前画布通道绕过校验，
        # 用户实测假权重/非法区域/未连通节点/数量判定均未被拦截）
        problems = validate_flow_for_pack(flow)
        if problems:
            _show_pack_problems(self.win, problems)
            self.send({"event": "status", "text": "打包取消：流程校验未通过（见弹窗清单）"})
            return
        try:
            models = fp.collect_model_files(flow)
        except FileNotFoundError as e:
            QMessageBox.information(self.win, "打包 .xvap", str(e))
            return
        if not models:
            has_legacy = any(str(n.get("type", "")).startswith(
                ("model/", "xccv/model", "post/", "xccv/post_"))
                for n in flow.get("nodes", []))
            if has_legacy:
                QMessageBox.information(
                    self.win, "打包 .xvap",
                    "流程中没有已选择模型文件的模型节点（在节点属性面板中浏览选择）")
                return
            # 纯传统算子流程（用户批注 F2）：允许打包为不含模型的 .xvap
        path, _ = QFileDialog.getSaveFileName(
            self.win, "打包流程", "flow.xvap", "流程包 (*.xvap)")
        if not path:
            return
        fp.save_xvap(path, flow, models)
        tip = f"（含 {len(models)} 个模型）" if models else "（纯算子流程，不含模型）"
        self.send({"event": "status", "text": f"已打包 {Path(path).name}{tip}"})

    def do_pick_model(self, msg):
        path, _ = QFileDialog.getOpenFileName(
            self.win, "选择模型文件", "",
            "模型权重 (*.pt);;所有文件 (*)")
        if path:
            self.send({"event": "model_file", "node_id": msg.get("node_id"), "path": path})

    def do_pick_param_file(self, msg):
        """schema 面板 path 型参数的文件浏览（path 参数复用同一桥接链路）。"""
        path, _ = QFileDialog.getOpenFileName(
            self.win, "选择文件", "",
            "图片/模型/流程 (*.jpg *.jpeg *.png *.bmp *.pt *.json *.xvap);;所有文件 (*)")
        if path:
            self.send({"event": "param_file", "node_id": msg.get("node_id"),
                       "key": msg.get("key"), "path": path})

    def do_pick_param_folder(self, msg):
        """schema 面板 folder 型参数的目录浏览（用户批注 ToC1：输出目录
        应选文件夹而非文件）。"""
        path = QFileDialog.getExistingDirectory(self.win, "选择文件夹")
        if path:
            self.send({"event": "param_file", "node_id": msg.get("node_id"),
                       "key": msg.get("key"), "path": path})

    # ---- 新引擎运行会话（批次十九：运行到此 / 单节点重跑 / 快照 / 日志） ----
    def _session(self):
        if getattr(self.win, "_flow_session", None) is None:
            from .flow_session import FlowSession

            self.win._flow_session = FlowSession()
        return self.win._flow_session

    def do_run_to_node(self, msg):
        flow = msg.get("flow") or {}
        node_id = msg.get("node_id")
        try:
            summary = self._session().run(
                flow, to_node=node_id, label="画布运行",
                flow_dir=(str(self.win.current_path.parent)
                          if self.win.current_path else None))
        except Exception as e:
            log.error("新引擎运行失败: %s", e, exc_info=True)
            self.send({"event": "status", "text": f"运行失败：{e}"})
            return
        summary["event"] = "run_result"
        summary["status"] = (f"运行完成：{summary['ok_count']}/{summary['total']} 节点成功"
                             + (f"，{len(summary['errors'])} 个失败" if summary["errors"] else ""))
        self.send(summary)

    def do_rerun_node(self, msg):
        try:
            summary = self._session().rerun(int(msg.get("node_id")), label="画布")
        except Exception as e:
            log.error("单节点重跑失败: %s", e, exc_info=True)
            self.send({"event": "status", "text": f"重跑失败：{e}"})
            return
        summary["event"] = "run_result"
        summary["status"] = f"节点 #{msg.get('node_id')} 已重跑"
        self.send(summary)

    def do_get_snapshot(self, msg):
        node_id = msg.get("node_id")
        self.send({"event": "snapshot", "node_id": node_id,
                   "images": self._session().snapshot_payload(int(node_id))})

    def do_pick_image(self, msg):
        path, _ = QFileDialog.getOpenFileName(
            self.win, "选择用于框选区域的图片", "",
            "图片 (*.jpg *.jpeg *.png *.bmp)")
        if not path:
            return
        import base64

        suffix = Path(path).suffix.lower().lstrip(".")
        mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png",
                "bmp": "bmp"}.get(suffix, "jpeg")
        try:  # 图片内容经桥接下发（base64），不再依赖页面 file:// 子资源加载
            data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
        except OSError as e:
            self.send({"event": "status", "text": f"读取图片失败: {e}"})
            return
        self.send({"event": "image_for_regions", "node_id": msg.get("node_id"),
                   "path": path, "data_url": f"data:image/{mime};base64,{data}"})

    def do_run_preview(self, msg):
        flow = msg.get("flow") or {}
        # 旧链运行预览：仅服务旧版流程文件（新流程一律走新引擎 run_to_node）
        model_file = next(((n.get("properties") or {}).get("model_file")
                           for n in flow.get("nodes", [])
                           if str(n["type"]).startswith(("model/", "xccv/model"))
                           and (n.get("properties") or {}).get("model_file")), None)
        if not model_file:
            QMessageBox.information(self.win, "运行预览", "请先在模型节点中选择模型文件")
            return
        if self.win._preview_sample is None:
            path, _ = QFileDialog.getOpenFileName(
                self.win, "选择预览样图", "", "图片 (*.jpg *.jpeg *.png *.bmp)")
            if not path:
                return
            from ..engine.io_utils import imread_unicode

            self.win._preview_sample = imread_unicode(path)
            if self.win._preview_sample is None:
                QMessageBox.information(self.win, "运行预览", "无法读取该图片")
                return
        if self.win._preview_model_path != model_file:
            from ..engine.models.registry import load_model

            self.win._preview_model = load_model(model_file)
            self.win._preview_model_path = model_file
        model = self.win._preview_model
        img = self.win._preview_sample
        pred = model.predict(img)
        base_count = len(pred.boxes)
        chain = graph_to_chain(flow)
        selected_id = msg.get("node_id")
        # 只执行到所选节点（含）：按链序截断
        upto = None
        for i, node in enumerate(flow.get("nodes", [])):
            if node.get("id") == selected_id:
                pos_in_chain = _chain_position(chain, node)
                if pos_in_chain is not None:
                    upto = pos_in_chain + 1
                break
        if upto is not None:
            chain = {"nodes": chain["nodes"][:upto]}
        classes = getattr(model, "classes", None)
        from ..engine.flow.legacy_adapter import apply_flow_safe

        filt, info = apply_flow_safe(pred, (img.shape[1], img.shape[0]), chain, classes=classes)
        from ..testing.render import draw_predictions

        task = getattr(model, "task", None)
        task_v = task.value if task is not None else "detection"
        rendered = draw_predictions(img, filt, model.classes or [], task_v,
                                    show_labels=True,
                                    show_masks=(task_v == "segmentation"))
        import cv2

        ok, png = cv2.imencode(".png", rendered)
        import base64

        self.send({"event": "preview", "png": base64.b64encode(png.tobytes()).decode(),
                   "text": f"预览：原始检出 {base_count} → 当前 {len(filt.boxes)}"
                           f"（数量判定 {info.get('count_check') or '未配置'}）"})


def _chain_position(chain: dict, node: dict) -> int | None:
    """节点在执行链中的位置（按 op 与参数粗匹配）。"""
    params = node.get("properties") or node.get("params") or {}
    t = str(node.get("type", ""))
    op = t.split("post/", 1)[-1].split("xccv/post_", 1)[-1]
    for i, c in enumerate(chain.get("nodes", [])):
        if c["op"] == op and c.get("params") == params:
            return i
    return None


class FlowGraphWebWindow(QMainWindow):
    tool_key = "flow"
    tool_title = "流程图编辑器"

    def __init__(self, on_closed=None):
        super().__init__()
        self.on_closed = on_closed
        from PyQt5.QtGui import QIcon

        # v0.5.5追加：流程图编辑器独立图标与平台归属（不再挂模型工具分类）
        self.setWindowTitle(f"{self.tool_title}  -  XCcv 流程图")
        self.setWindowIcon(QIcon(str(icon_path("flow.png"))))
        self.resize(1240, 800)
        # 按要求不给窗口挂任何 Qt 样式表（QSS）：界面全部由 web 页面自绘，
        # Qt 只提供窗口壳与 新建/打开/打包/选模型 的文件对话框，规避 QSS
        # polish 与样式重算带来的卡顿
        self.current_path: Path | None = None
        self._preview_sample = None
        self._preview_model = None
        self._preview_model_path = None
        self._flow_session = None  # 新引擎运行会话（FlowSession，懒建）

        self.bridge = _Bridge(self)
        self._js_ready = False
        self._pending_js: list[str] = []

        self.channel = QWebChannel(self)
        self.channel.registerObject("backend", self.bridge)

        class _Page(QWebEnginePage):
            def javaScriptConsoleMessage(page_self, level, message, line, source):
                if message.startswith("XCCV:"):
                    self.bridge.message(message)
                return QWebEnginePage.javaScriptConsoleMessage(
                    page_self, level, message, line, source)

        self.page = _Page(self)
        view = QWebEngineView(self)
        view.setPage(self.page)
        self.page.setWebChannel(self.channel)
        view.setContextMenuPolicy(_Qt.NoContextMenu)  # 禁用浏览器右键菜单
        url = QUrl.fromLocalFile(str(_WEB_DIR / "index.html"))
        query = [q for q, on in (("nort=1", os.environ.get("XCCV_FLOW_NORT")),
                                 ("nolg=1", os.environ.get("XCCV_FLOW_NOLG"))) if on]
        if query:  # 调试开关：nort=禁用连续渲染循环；nolg=不加载 litegraph.js
            url.setQuery("&".join(query))
        self.page.setUrl(url)
        self.setCentralWidget(view)

    def send_json(self, payload: dict):
        """Python → JS 事件：走 QWebChannel 信号；页面通道就绪前缓冲。"""
        data = json.dumps(payload, ensure_ascii=False)
        if self._js_ready:
            self.bridge.bridge_event(data)
        else:
            self._pending_js.append(data)

    def load_flow_payload(self, flow: dict, name: str):
        """外部（主界面）把流程载入画布：事件在通道就绪后自动下发。"""
        self.send_json({"event": "loaded", "flow": flow, "name": name})

    def closeEvent(self, ev):
        if self.on_closed:
            self.on_closed(self.tool_key)
        # QtWebEngine 页面必须先于应用退出销毁，否则进程结束时段错误
        self.page.deleteLater()
        self.centralWidget().deleteLater()
        self.channel.deleteLater()
        super().closeEvent(ev)


# ---- Cython 编译态槽包装（信号槽参数裁剪修复，见 xccv/qt_slots.py）----
from xccv.qt_slots import apply_auto_slot_module  # noqa: E402
apply_auto_slot_module(globals())
