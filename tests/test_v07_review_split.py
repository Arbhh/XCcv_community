"""V0.7 勘误重构（2026-11-13）：测试平台 标准/项目复盘 拆分回归锁。

- review_logic：三分组（过检>漏检>正确）、阈值复用现算、OK 图口径、未评估；
- batch_match：per_image 增逐检出 dets 记录（列序与 review_logic 对齐）；
- 阈值面板：thresholdsChanged 信号（拖线/开关 → enabled_dims）；
- 体检：__ignore__ 不再被当类别统计（"样本过少：__ignore__=1" 瑕疵修复）；
- 测试窗口：双页结构存在、底部页签存在、标准页不再含复判/AB/阈值面板。
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from xccv.engine.io_utils import imwrite_unicode
from xccv.engine.metrics.batch_match import collect_match_stats
from xccv.testing.review_logic import (DET_FG, DET_MATCHED, DET_SCORE,
                                       classify_images, det_passes, stats_line)


def _write_png(path: Path):
    imwrite_unicode(str(path), np.full((8, 8, 3), 200, dtype=np.uint8))


def _write_json(path: Path, shapes: list):
    path.write_text(json.dumps({
        "version": "5.8.1", "flags": {}, "shapes": shapes,
        "imagePath": path.with_suffix(".png").name, "imageData": None,
        "imageHeight": 200, "imageWidth": 200,
    }), encoding="utf-8")


def _rect(label: str, box) -> dict:
    return {"label": label, "points": [[box[0], box[1]], [box[2], box[3]]],
            "shape_type": "rectangle"}


def _det(label, score, box, dims=None):
    d = {"label": label, "score": score, "bbox": [float(v) for v in box]}
    if dims:
        d["_dims"] = dims
    return d


class DetPassesTest(unittest.TestCase):

    def test_all_dims_must_pass(self):
        rec = [0.8, True, 5000.0, 50.0, 100.0, 0.6, 120.0]
        self.assertTrue(det_passes(rec, {"score": 0.5, "area": 1000}))
        self.assertFalse(det_passes(rec, {"area": 6000}))
        self.assertFalse(det_passes(rec, {"score": 0.9}))

    def test_none_value_fails(self):
        rec = [0.8, True, 5000.0, 50.0, 100.0, None, None]
        self.assertFalse(det_passes(rec, {"fg_ratio": 0.05}))  # 缺图维度不过
        self.assertTrue(det_passes(rec, {}))


class ClassifyTest(unittest.TestCase):

    def _stats(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _write_png(d / "a.png")
            _write_json(d / "a.json", [_rect("黑点", (0, 0, 100, 100))])
            _write_png(d / "b.png")
            _write_json(d / "b.json", [_rect("黑点", (0, 0, 100, 100))])
            _write_png(d / "ok.png")
            _write_json(d / "ok.json", [])
            _write_png(d / "nogt.png")
            results = [
                {"path": str(d / "a.png"), "detections": [
                    _det("黑点", 0.9, (0, 0, 100, 100), {"fg_ratio": 0.8})]},
                {"path": str(d / "b.png"), "detections": [
                    _det("黑点", 0.9, (0, 0, 100, 100)),
                    _det("黑点", 0.4, (150, 150, 180, 180), {"fg_ratio": 0.01})]},
                {"path": str(d / "ok.png"), "detections": [
                    _det("黑点", 0.6, (10, 10, 50, 50))]},
                {"path": str(d / "nogt.png"), "detections": []},
            ]
            return collect_match_stats(results, iou_thr=0.5)

    def test_groups_and_rethreshold_without_rerun(self):
        stats = self._stats()
        # conf=0.25：a 正确；b 过检（多一个 0.4 FP）；ok 过检（OK 误检）；nogt 未评估
        g = classify_images(stats, {"score": 0.25})
        self.assertEqual([e["file"] for e in g["correct"]], ["a.png"])
        self.assertEqual([e["file"] for e in g["over"]], ["b.png", "ok.png"])
        self.assertEqual(len(g["unjudged_list"]), 1)
        self.assertEqual(g["unjudged"], 1)
        self.assertEqual(g["total"], 4)
        # 再勘误④：未评估图必须带条目进组（生产图肉眼可确认）
        self.assertEqual(g["unjudged_list"][0]["file"], "nogt.png")
        # conf=0.5：b 的低分 FP 被滤掉 → b 转正确；ok 仍过检（0.6≥0.5）
        g = classify_images(stats, {"score": 0.5})
        self.assertEqual([e["file"] for e in g["correct"]], ["a.png", "b.png"])
        self.assertEqual([e["file"] for e in g["over"]], ["ok.png"])
        # 前景占比维度：把 b 的主检出也滤掉 → b 变漏检
        g = classify_images(stats, {"score": 0.25, "fg_ratio": 0.5})
        self.assertEqual([e["file"] for e in g["missed"]], ["b.png"])

    def test_ok_dir_same_as_training_rule(self):
        """再勘误③：OK 图判定与训练平台目录名规则（NEGATIVE_DIR_NAMES）同源——
        OK 目录内无 json 的图按 OK 评估；非 OK 目录无 json 图 = 未评估。"""
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            okd = d / "良品"
            okd.mkdir()
            _write_png(okd / "k.png")            # OK 目录无 json → OK 图
            _write_png(d / "prod.png")           # 无 json 生产图 → 未评估
            results = [{"path": str(okd / "k.png"), "detections": []},
                       {"path": str(d / "prod.png"), "detections": []}]
            stats = collect_match_stats(results, iou_thr=0.5)
            by = {Path(e["path"]).name: e for e in stats["per_image"]}
            self.assertTrue(by["k.png"]["gt_available"])       # OK 图参与评估
            self.assertTrue(by["k.png"]["ok_dir"])
            self.assertFalse(by["prod.png"]["gt_available"])   # 生产图未评估
            g = classify_images(stats, {})
            self.assertEqual(len(g["correct"]), 1)             # k.png 零检出=正确
            self.assertEqual(len(g["unjudged_list"]), 1)       # prod.png 可见

    def test_prod_only_folder_still_groups(self):
        """再勘误④：纯生产图（无任何 GT json）也必须产出完整 per_image，
        复盘「未评估」组要能展示每张图（修复 gt_available 短路）。"""
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _write_png(d / "p0.png")
            _write_png(d / "p1.png")
            results = [{"path": str(d / "p0.png"), "detections": [
                            _det("黑点", 0.5, (10, 10, 60, 60))]},
                       {"path": str(d / "p1.png"), "detections": []}]
            stats = collect_match_stats(results, iou_thr=0.5)
            self.assertFalse(stats["gt_available"])
            self.assertEqual(len(stats["per_image"]), 2)     # 不再被短路吞掉
            g = classify_images(stats, {})
            self.assertEqual(len(g["unjudged_list"]), 2)
            self.assertEqual(g["unjudged"], 2)
            self.assertEqual(g["total"], 2)

    def test_stats_line(self):
        g = {"correct": [1, 2], "missed": [1], "over": [], "unjudged": 3}
        line = stats_line(g)
        self.assertIn("正确 2", line)
        self.assertIn("漏检 1", line)
        self.assertIn("未评估 3", line)


class DetsRecordTest(unittest.TestCase):

    def test_per_image_dets_columns(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _write_png(d / "a.png")
            _write_json(d / "a.json", [_rect("黑点", (0, 0, 100, 100))])
            results = [{"path": str(d / "a.png"), "detections": [
                _det("黑点", 0.9, (10, 10, 110, 110), {"fg_ratio": 0.7,
                                                       "gray_diff": 88.0})]}]
            stats = collect_match_stats(results, iou_thr=0.5)
            rec = stats["per_image"][0]["dets"][0]
            self.assertEqual(rec[DET_SCORE], 0.9)
            self.assertTrue(rec[DET_MATCHED])
            self.assertEqual(rec[DET_FG], 0.7)

    def test_ok_image_dets_all_unmatched(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "OK"
            d.mkdir()
            _write_png(d / "k.png")
            _write_json(d / "k.json", [])
            results = [{"path": str(d / "k.png"), "detections": [
                _det("黑点", 0.7, (5, 5, 45, 45))]}]
            stats = collect_match_stats(results, iou_thr=0.5)
            rec = stats["per_image"][0]["dets"][0]
            self.assertFalse(rec[DET_MATCHED])
            self.assertTrue(stats["per_image"][0]["gt_available"])


class PanelSignalTest(unittest.TestCase):
    """thresholdsChanged：拖线即启用并广播 enabled_dims。"""

    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_thresholds_changed_signal(self):
        from xccv.testing.threshold_panel import ThresholdPanel

        p = ThresholdPanel()
        stats = {"gt_available": True, "gt_total": 1, "n_gt_images": 1,
                 "n_ok_images": 0, "n_unmatched": 0, "green": [0.9],
                 "red_defect": [0.4], "red_ok": [],
                 "dims": {"area": {"green": [10000.0], "red_defect": [400.0],
                                   "red_ok": []}},
                 "per_image": []}
        p.set_stats(stats, 0.25)
        got = {}
        p.thresholdsChanged.connect(
            lambda d: (got.clear(), got.update(d)))
        idx = p.cmb_dim.findData("area")
        p.cmb_dim.setCurrentIndex(idx)
        p._line.setValue(500.0)   # 拖线 → 启用 area
        self.assertEqual(got, {"score": 0.25, "area": 500.0})
        p.cb_enable.setChecked(False)   # 取消参与 → 只剩 score
        self.assertEqual(got, {"score": 0.25})


class DataHealthIgnoreClassTest(unittest.TestCase):

    def test_ignore_not_counted_as_class(self):
        from xccv.training.data_health import run_health_check

        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _write_png(d / "a.png")
            _write_json(d / "a.json", [_rect("黑点", (10, 10, 60, 60)),
                                       _rect("__ignore__", (100, 100, 180, 180))])
            rep = run_health_check("detection", d)
            texts = [i.detail for i in rep.issues]
            self.assertFalse(any("__ignore__=" in t for t in texts),
                             texts)


class TestingWindowStructureTest(unittest.TestCase):
    """双页结构：底部页签 + 标准页不含本轮新增控件 + 复盘页核心件齐备。"""

    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])
        cls.win = None

    @classmethod
    def tearDownClass(cls):
        if cls.win is not None:
            cls.win.close()

    def test_two_pages_and_mode_buttons(self):
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from xccv.testing.testing_window import TestingWindow

        w = TestingWindow()
        type(self).win = w
        # 底部页签（状态栏）+ 双页 stack
        self.assertTrue(w.btn_mode_std.isChecked())
        self.assertFalse(w.btn_mode_review.isChecked())
        self.assertEqual(w._detail_stack.count(), 2)
        self.assertIs(w._detail_stack.currentWidget(),
                      w._detail_stack.widget(0))
        w._switch_mode(1)
        self.assertIs(w._detail_stack.currentWidget(), w.review_page)
        self.assertTrue(w.btn_mode_review.isChecked())
        # 标准页已剥离本轮新增控件
        self.assertFalse(hasattr(w, "btn_review"))
        self.assertFalse(hasattr(w, "btn_ab"))
        self.assertFalse(hasattr(w, "btn_dist"))
        self.assertFalse(hasattr(w, "threshold_panel"))
        # 复盘页核心件
        self.assertTrue(hasattr(w.review_page, "threshold_panel"))
        self.assertTrue(hasattr(w.review_page, "cmb_run"))
        # 再勘误①④：数据集可浏览 + 四分组（未评估组在列）
        self.assertTrue(hasattr(w.review_page, "btn_browse"))
        self.assertTrue(hasattr(w.review_page, "fld_dataset"))
        self.assertEqual(len(w.review_page.grids), 4)
        self.assertIn("未评估", w.review_page.grp_unjudged.title())
        # 再勘误②：面板为 compact 上下布局（直方图不被表单挤成窄条）
        from xccv.testing.threshold_panel import ThresholdPanel
        self.assertIsInstance(w.review_page.threshold_panel, ThresholdPanel)
        # 勘误：关闭/启动兜底清缓存——close 后 review_cache 目录必须消失
        cache = Path("platform_data/review_cache")
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "复盘缓存.json").write_text("{}", encoding="utf-8")
        w.close()
        self.assertFalse((cache / "复盘缓存.json").is_file(),
                         "closeEvent 未清理复盘缓存")
        w._switch_mode(0)


if __name__ == "__main__":
    unittest.main()
