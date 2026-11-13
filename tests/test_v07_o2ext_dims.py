"""V0.7 O2 扩展（用户追加）：多维度阈值——算子/统计/生成器/面板回归锁。"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from xccv.engine.flow.node_base import RunContext
from xccv.engine.flow.ops.det_filters import (FilterBboxSide,
                                             FilterForeground, FilterGrayDiff,
                                             box_dims, box_image_dims)
from xccv.engine.flow.threshold_flow import (build_threshold_graph,
                                             flow_filename,
                                             save_threshold_flow)
from xccv.engine.io_utils import imwrite_unicode
from xccv.engine.metrics.batch_match import (collect_match_stats,
                                             threshold_metrics_for)
from xccv.engine.models.base import Prediction


def _scene(bright_box=(100, 100, 150, 150), bright=210, bg=30):
    """200×200 暗背景 + 亮块；返回 (BGR 图, 灰度图)。"""
    import cv2

    img = np.full((200, 200, 3), bg, dtype=np.uint8)
    x1, y1, x2, y2 = bright_box
    img[y1:y2, x1:x2] = bright
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def _pred(boxes, scores=None):
    return Prediction(labels=[0] * len(boxes),
                      scores=scores or [0.9] * len(boxes),
                      boxes=[[float(v) for v in b] for b in boxes])


class BoxDimsTest(unittest.TestCase):

    def test_geometry(self):
        d = box_dims((10, 20, 110, 70))
        self.assertAlmostEqual(d["area"], 100 * 50)
        self.assertEqual(d["short"], 50)
        self.assertEqual(d["long"], 100)

    def test_image_dims_bright_box(self):
        _img, gray = _scene()
        d = box_image_dims(gray, (100, 100, 150, 150))
        self.assertIsNotNone(d["fg_ratio"])
        self.assertGreater(d["fg_ratio"], 0.9)      # 亮块全是前景
        self.assertGreater(d["gray_diff"], 100)     # 与暗背景差悬殊

    def test_image_dims_flat_box(self):
        _img, gray = _scene()
        d = box_image_dims(gray, (10, 10, 60, 60))  # 与背景同灰度
        self.assertLess(d["fg_ratio"], 0.05)
        self.assertLess(d["gray_diff"], 3)

    def test_image_dims_box_at_border(self):
        _img, gray = _scene()
        d = box_image_dims(gray, (180, 180, 260, 260))  # 越界框不崩溃
        self.assertIsInstance(d, dict)


class FilterNodeTest(unittest.TestCase):

    def test_side_filter(self):
        pred = _pred([(0, 0, 200, 8), (0, 0, 60, 60), (0, 0, 100, 30)])
        out = FilterBboxSide().run({"result": pred},
                                   {"short_min": 20.0, "long_min": 0.0},
                                   RunContext())
        kept = out["result"].boxes
        self.assertEqual(len(kept), 2)   # 8px 高的横条被滤掉
        self.assertEqual(len(out["removed"].boxes), 1)

    def test_foreground_filter_needs_image(self):
        img, _gray = _scene()
        pred = _pred([(100, 100, 150, 150), (10, 10, 60, 60)])
        out = FilterForeground().run({"image": img, "result": pred},
                                     {"min_ratio": 0.05, "gray_thr": 20},
                                     RunContext())
        self.assertEqual(len(out["result"].boxes), 1)   # 平块被滤
        self.assertEqual(out["removed"].boxes[0], [10.0, 10.0, 60.0, 60.0])

    def test_gray_diff_filter(self):
        img, _gray = _scene()
        pred = _pred([(100, 100, 150, 150), (10, 10, 60, 60)])
        out = FilterGrayDiff().run({"image": img, "result": pred},
                                   {"min_diff": 30.0}, RunContext())
        self.assertEqual(len(out["result"].boxes), 1)

    def test_no_image_passes_through(self):
        pred = _pred([(0, 0, 50, 50)])
        out = FilterForeground().run({"result": pred},
                                     {"min_ratio": 0.99, "gray_thr": 20},
                                     RunContext())
        self.assertEqual(len(out["result"].boxes), 1)   # 缺图放行 + 告警


class DimsCollectTest(unittest.TestCase):

    def test_dims_buckets_and_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            imwrite_unicode(str(d / "a.png"), np.zeros((8, 8, 3), np.uint8))
            (d / "a.json").write_text(json.dumps({
                "shapes": [{"label": "黑点", "points": [[0, 0], [100, 100]],
                            "shape_type": "rectangle"}],
                "imagePath": "a.png", "imageData": None,
                "imageWidth": 200, "imageHeight": 200}), encoding="utf-8")
            results = [{"path": str(d / "a.png"), "detections": [
                {"label": "黑点", "score": 0.9, "bbox": [0, 0, 100, 100],
                 "_dims": {"fg_ratio": 0.8, "gray_diff": 120.0}},
                {"label": "黑点", "score": 0.4, "bbox": [150, 150, 170, 170],
                 "_dims": {"fg_ratio": 0.01, "gray_diff": 3.0}}]}]
            stats = collect_match_stats(results)
            dims = stats["dims"]
            self.assertEqual(len(dims["area"]["green"]), 1)
            self.assertEqual(dims["area"]["green"][0], 10000.0)
            self.assertEqual(dims["short"]["red_defect"], [20.0])
            self.assertEqual(dims["fg_ratio"]["green"], [0.8])
            self.assertEqual(dims["gray_diff"]["red_defect"], [3.0])
            # GT 共 1 实例（大框 TP、小框 FP）：短边阈值 25 → 只留 TP
            m = threshold_metrics_for(stats, "short", 25.0)
            self.assertEqual(m["precision"], 1.0)
            self.assertEqual(m["recall"], 1.0)
            # 阈值降到 15 → FP 也保留：精确率掉到 0.5
            m = threshold_metrics_for(stats, "short", 15.0)
            self.assertEqual(m["precision"], 0.5)

    def test_ok_image_dims_go_red_ok(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "OK"
            d.mkdir()
            imwrite_unicode(str(d / "ok0.png"), np.zeros((8, 8, 3), np.uint8))
            (d / "ok0.json").write_text(json.dumps({
                "shapes": [], "imagePath": "ok0.png", "imageData": None,
                "imageWidth": 200, "imageHeight": 200}), encoding="utf-8")
            results = [{"path": str(d / "ok0.png"), "detections": [
                {"label": "黑点", "score": 0.8, "bbox": [10, 10, 60, 60]}]}]
            stats = collect_match_stats(results)
            self.assertEqual(stats["n_ok_images"], 1)
            self.assertEqual(stats["dims"]["area"]["red_ok"], [2500.0])


class ThresholdGraphTest(unittest.TestCase):

    def test_registry_no_duplicate_with_legacy(self):
        """重名注册锁：ops 与 legacy_adapter 双模块导入不得冲突（f7g 抓到
        filter_bbox_area 重名——面积过滤沿用 legacy 薄委托节点）。"""
        from xccv.engine.flow import legacy_adapter  # noqa: F401  触发薄委托注册
        from xccv.engine.flow.ops import load_all
        from xccv.engine.flow.registry import get_node_class

        load_all()
        for n in ("filter_bbox_area", "filter_bbox_side",
                  "filter_foreground", "filter_gray_diff"):
            self.assertIsNotNone(get_node_class(n), n)

    def test_graph_structure_and_registry(self):
        from xccv.engine.flow import legacy_adapter  # noqa: F401  filter_bbox_area 在薄委托模块
        from xccv.engine.flow.executor import Executor
        from xccv.engine.flow.ops import load_all

        load_all()   # 注册全部算子（含新增 3 个）
        g = build_threshold_graph(
            "C:/m/best.xva", "斑点检测",
            {"conf": 0.35, "area": 500, "short": 12, "fg_ratio": 0.1})
        types = [n["type"] for n in g["nodes"]]
        self.assertEqual(types, ["xccv2/image_loader", "xccv2/model_detection",
                                 "xccv2/filter_bbox_area", "xccv2/filter_bbox_side",
                                 "xccv2/filter_foreground", "xccv2/result_output"])
        model = g["nodes"][1]["properties"]
        self.assertEqual(model["conf"], 0.35)
        self.assertEqual(model["model_path"], "C:/m/best.xva")
        # 链路双端一致 + 静态校验全过（结构/类型/环）
        for n in g["nodes"]:
            for inp in n["inputs"]:
                lid = inp.get("link")
                if lid is not None:
                    lk = g["links"][lid - 1]
                    self.assertEqual(lk[3], n["id"])
            for slot, out in enumerate(n["outputs"]):
                for lid in out["links"]:
                    lk = g["links"][lid - 1]
                    self.assertEqual((lk[1], lk[2]), (n["id"], slot))
        # Executor 校验用注册表键（与 flow_session._normalize_types 同语义：
        # "xccv2/<注册名>" → "<注册名>"）
        bare = {**g, "nodes": [{**n, "type": n["type"].split("/", 1)[1]}
                               for n in g["nodes"]]}
        Executor({"nodes": bare["nodes"], "links": g["links"]}).validate()

    def test_long_merges_into_side_node(self):
        g = build_threshold_graph("w.xva", "m", {"short": 8, "long": 200})
        types = [n["type"] for n in g["nodes"]]
        self.assertEqual(types.count("xccv2/filter_bbox_side"), 1)
        props = next(n["properties"] for n in g["nodes"]
                     if n["type"] == "xccv2/filter_bbox_side")
        self.assertEqual(props, {"short_min": 8.0, "long_min": 200.0})

    def test_filename_and_save_location(self):
        name = flow_filename("斑点/检测*模型")
        self.assertNotIn("/", name)
        self.assertTrue(name.split("_", 2)[2].endswith("模型.json"))
        with tempfile.TemporaryDirectory() as td:
            path = save_threshold_flow(td, "斑点检测",
                                       {"conf": 0.4, "area": 300}, "w.xva")
            self.assertEqual(path.parent.name, "flow")
            self.assertIn("斑点检测", path.name)
            g = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(g["schema"], 3)
            self.assertEqual(g["nodes"][1]["properties"]["conf"], 0.4)


class PanelSmokeTest(unittest.TestCase):
    """offscreen 面板冒烟：维度切换/拖线启用/生成信号。"""

    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def _stats(self):
        return {"gt_available": True, "gt_total": 2, "n_gt_images": 1,
                "n_ok_images": 1, "n_unmatched": 1,
                "green": [0.9], "red_defect": [0.4], "red_ok": [0.6],
                "dims": {"area": {"green": [10000.0], "red_defect": [400.0],
                                  "red_ok": [2500.0]},
                         "short": {"green": [100.0], "red_defect": [20.0],
                                   "red_ok": [50.0]},
                         "fg_ratio": {"green": [0.8], "red_defect": [0.01],
                                      "red_ok": []},
                         "gray_diff": {"green": [120.0], "red_defect": [3.0],
                                       "red_ok": []}},
                "per_image": []}

    def test_dims_flow(self):
        from xccv.testing.threshold_panel import ThresholdPanel

        p = ThresholdPanel()
        p.set_stats(self._stats(), 0.25)
        # 维度下拉：score + 4 个有数据的维度（long 无数据不出现）
        keys = [p.cmb_dim.itemData(i) for i in range(p.cmb_dim.count())]
        self.assertEqual(keys, ["score", "area", "short", "fg_ratio", "gray_diff"])
        # 切到短边：默认 0 未启用
        idx = p.cmb_dim.findData("short")
        p.cmb_dim.setCurrentIndex(idx)
        self.assertFalse(p._dim_state["short"]["enabled"])
        # 拖线（模拟 setValue）→ 自动启用
        p._line.setValue(25.0)
        self.assertTrue(p._dim_state["short"]["enabled"])
        self.assertEqual(p._dim_state["short"]["thr"], 25.0)
        m = threshold_metrics_for(self._stats(), "short", 25.0)
        self.assertEqual(p.lbl_precision.text(), f"{m['precision']:.3f}")
        # 生成信号携带 conf+short
        got = {}
        p.generateFlowRequested.connect(lambda d: got.update(d))
        p._emit_generate_flow()
        self.assertEqual(got, {"score": 0.25, "short": 25.0})
        # 应用按钮恒用 conf 维（即使当前在 short 维）
        vals = []
        p.applyRequested.connect(lambda v: vals.append(v))
        p._emit_apply_conf()
        self.assertEqual(vals, [0.25])


if __name__ == "__main__":
    unittest.main()
