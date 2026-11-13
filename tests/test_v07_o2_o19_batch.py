"""V0.7 O2/O19/O20：批量 GT 匹配统计、阈值面板、复判回流回归锁。

纯逻辑全覆盖（batch_match / batch_review），UI 接线用源码锁；
对话框模态交互（回流确认/目录选择）属真机验证项（P196 纪律）。
"""

import json
import tempfile
import unittest
from pathlib import Path

from xccv.engine.metrics.batch_match import (
    collect_match_stats,
    match_detections,
    read_gt_shapes,
    threshold_metrics,
)
from xccv.testing.batch_review import (
    execute_reflow,
    load_review,
    plan_reflow,
    review_stats,
    save_review,
    suggest_status,
    STATUS_FP, STATUS_MISSED, STATUS_OK, STATUS_UNJUDGED,
)

LABELME_DEFECT = {
    "version": "5.8.1", "flags": {},
    "shapes": [
        {"label": "黑点", "points": [[10, 10], [50, 50]],
         "shape_type": "rectangle"},
        {"label": "划痕", "points": [[100, 100], [150, 100], [150, 130]],
         "shape_type": "polygon"},
    ],
    "imagePath": "a.png", "imageData": None, "imageHeight": 200,
    "imageWidth": 200,
}


class ReadGtShapesTest(unittest.TestCase):

    def test_rectangle_and_polygon(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.json"
            p.write_text(json.dumps(LABELME_DEFECT), encoding="utf-8")
            boxes, labels = read_gt_shapes(p)
            self.assertEqual(labels, ["黑点", "划痕"])
            self.assertEqual(boxes[0], [10.0, 10.0, 50.0, 50.0])
            self.assertEqual(boxes[1], [100.0, 100.0, 150.0, 130.0])

    def test_missing_file_returns_empty(self):
        boxes, labels = read_gt_shapes(Path("no/such.json"))
        self.assertEqual((boxes, labels), ([], []))


class MatchDetectionsTest(unittest.TestCase):

    def test_greedy_one_to_one_by_score(self):
        # 两个检出都压住同一个 GT：只有分数更高者命中（GT 一对一）
        det_boxes = [[10, 10, 50, 50], [12, 12, 52, 52]]
        det_labels = ["黑点", "黑点"]
        det_scores = [0.4, 0.9]
        matched = match_detections(det_boxes, det_labels, det_scores,
                                   [[10, 10, 50, 50]], ["黑点"], 0.5)
        self.assertEqual(matched, [False, True])
        # 两个 GT 空闲时各得其所：分数降序贪心后都命中
        matched2 = match_detections(det_boxes, det_labels, det_scores,
                                    [[10, 10, 50, 50], [11, 11, 51, 51]],
                                    ["黑点", "黑点"], 0.5)
        self.assertEqual(matched2, [True, True])

    def test_cross_label_no_match(self):
        matched = match_detections([[10, 10, 50, 50]], ["A"], [0.9],
                                   [[10, 10, 50, 50]], ["B"], 0.5)
        self.assertEqual(matched, [False])

    def test_empty_sides(self):
        self.assertEqual(match_detections([], [], [], [[0, 0, 1, 1]], ["A"]), [])
        self.assertEqual(match_detections([[0, 0, 1, 1]], ["A"], [0.9], [], []), [False])


class CollectMatchStatsTest(unittest.TestCase):

    def _make_dir(self, td: Path):
        (td / "a.png").write_bytes(b"img")
        (td / "a.json").write_text(json.dumps(LABELME_DEFECT), encoding="utf-8")
        (td / "ok.png").write_bytes(b"img")
        (td / "ok.json").write_text(json.dumps(
            {"shapes": [], "imagePath": "ok.png"}), encoding="utf-8")
        (td / "noGT.png").write_bytes(b"img")

    def test_stats_over_defect_ok_and_ungt(self):
        results = [
            {"path": "td/a.png", "file": "a.png", "detections": [
                {"label": "黑点", "score": 0.9, "bbox": [10, 10, 50, 50]},
                {"label": "划痕", "score": 0.3, "bbox": [200, 200, 240, 240]},
            ]},
            {"path": "td/ok.png", "file": "ok.png", "detections": [
                {"label": "黑点", "score": 0.6, "bbox": [10, 10, 50, 50]},
            ]},
            {"path": "td/noGT.png", "file": "noGT.png", "detections": []},
        ]
        with tempfile.TemporaryDirectory() as td:
            self._make_dir(Path(td))
            results = [{**r, "path": str(Path(td) / Path(r["path"]).name)}
                       for r in results]
            stats = collect_match_stats(results, iou_thr=0.5)
        self.assertTrue(stats["gt_available"])
        self.assertEqual(stats["gt_total"], 2)
        self.assertEqual((stats["n_gt_images"], stats["n_ok_images"]), (1, 1))
        self.assertEqual(stats["green"], [0.9])       # 命中 黑点@0.9
        self.assertEqual(stats["red_defect"], [0.3])  # 划痕偏移未命中
        self.assertEqual(stats["red_ok"], [0.6])      # OK 图误检
        self.assertEqual(stats["n_unmatched"], 1)     # GT 划痕漏检
        by_name = {Path(e["path"]).name: e for e in stats["per_image"]}
        self.assertEqual(by_name["a.png"]["missed"], 1)
        self.assertEqual(by_name["ok.png"]["fp_ok"], 1)
        self.assertIsNone(by_name["noGT.png"]["gt"])

    def test_no_gt_at_all(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.png"
            p.write_bytes(b"img")
            stats = collect_match_stats(
                [{"path": str(p), "file": "x.png", "detections": []}])
        self.assertFalse(stats["gt_available"])


class ThresholdMetricsTest(unittest.TestCase):

    _stats = {"gt_available": True, "gt_total": 4, "n_ok_images": 2,
              "green": [0.9, 0.7, 0.5], "red_defect": [0.6, 0.2],
              "red_ok": [0.8, 0.1]}

    def test_metrics_at_threshold(self):
        m = threshold_metrics(self._stats, 0.5)
        self.assertEqual(m["tp"], 3)
        self.assertEqual(m["fp"], 2)                      # 0.6 + 0.8
        self.assertAlmostEqual(m["precision"], 3 / 5, places=4)
        self.assertAlmostEqual(m["recall"], 3 / 4, places=4)
        self.assertAlmostEqual(m["ok_fp_per_img"], 0.5, places=4)

    def test_high_threshold(self):
        m = threshold_metrics(self._stats, 0.85)
        self.assertEqual(m["tp"], 1)                       # 仅 0.9
        self.assertEqual(m["fp"], 0)                       # 0.8 < 0.85
        self.assertAlmostEqual(m["precision"], 1.0, places=4)
        self.assertAlmostEqual(m["recall"], 0.25, places=4)
        self.assertAlmostEqual(m["ok_fp_per_img"], 0.0, places=4)


class ReviewLogicTest(unittest.TestCase):

    def test_review_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "复判.json"
            save_review(p, {"a.png": {"status": "漏检", "time": "t"}})
            self.assertEqual(load_review(p)["a.png"]["status"], "漏检")
            self.assertEqual(load_review(p.with_name("无.json")), {})

    def test_suggest_status(self):
        self.assertEqual(suggest_status({"gt": None}), STATUS_UNJUDGED)
        self.assertEqual(suggest_status({"gt": 2, "missed": 1, "fp": 1}),
                         STATUS_MISSED)
        self.assertEqual(suggest_status({"gt": 2, "missed": 0, "fp": 1}),
                         STATUS_FP)
        self.assertEqual(suggest_status({"gt": 2, "missed": 0, "fp": 0}),
                         STATUS_OK)

    def test_review_stats_counts(self):
        entries = [{"path": "a", "gt": 1, "missed": 0, "fp": 0},
                   {"path": "b", "gt": 1, "missed": 1, "fp": 0},
                   {"path": "c", "gt": None}]
        stats = review_stats({}, entries)
        self.assertEqual(stats[STATUS_OK]["count"], 1)
        self.assertEqual(stats[STATUS_MISSED]["count"], 1)
        self.assertEqual(stats[STATUS_UNJUDGED]["count"], 1)
        self.assertEqual(stats["_total"], 3)
        self.assertAlmostEqual(stats[STATUS_MISSED]["rate"], 1 / 3, places=4)

    def test_reflow_with_gt_and_prediction(self):
        with tempfile.TemporaryDirectory() as td, \
                tempfile.TemporaryDirectory() as ds:
            td, ds = Path(td), Path(ds)
            img = td / "a.png"
            img.write_bytes(b"imgdata")
            (td / "a.json").write_text(json.dumps(LABELME_DEFECT),
                                       encoding="utf-8")
            entry = {"path": str(img), "gt": 2, "missed": 1, "fp": 0}
            plan = plan_reflow(entry, {}, ds)
            self.assertFalse(plan["used_prediction"])
            copied, errors = execute_reflow([plan])
            self.assertEqual((copied, errors), (1, []))
            dst = ds / "待标注" / "a.json"
            self.assertTrue((ds / "待标注" / "a.png").is_file())
            # 原 GT 标注原样复制（shapes 保留，imagePath 不改写）
            self.assertEqual(
                len(json.loads(dst.read_text(encoding="utf-8"))["shapes"]), 2)

            # 无 GT：预测框转 labelme
            img2 = td / "b.png"
            img2.write_bytes(b"img2")
            result = {"path": str(img2), "width": 200, "height": 200,
                      "detections": [{"label": "黑点", "score": 0.9,
                                      "bbox": [10, 10, 50, 50]}]}
            plan2 = plan_reflow({"path": str(img2), "gt": None}, result, ds)
            self.assertTrue(plan2["used_prediction"])
            copied, errors = execute_reflow([plan2])
            self.assertEqual((copied, errors), (1, []))
            lm = json.loads((ds / "待标注" / "b.json").read_text(
                encoding="utf-8"))
            self.assertEqual(lm["shapes"][0]["label"], "黑点")
            self.assertEqual(lm["imagePath"], str(img2))


class WindowWiringTest(unittest.TestCase):
    """testing_window / review_page 接线源码锁（勘误：面板与复判入口移入复盘页）。"""

    def test_wiring_present(self):
        src = Path("xccv/testing/testing_window.py").read_text(encoding="utf-8")
        review = Path("xccv/testing/review_page.py").read_text(
            encoding="utf-8")
        self.assertIn("ThresholdPanel(compact=True)", review)
        self.assertIn("self.btn_review.setEnabled(True)", review)
        self.assertIn("collect_match_stats(", src)     # 标准批量线程仍出统计
        self.assertIn("collect_match_stats(", review)  # 复盘线程也出统计
        self.assertIn('node["params"]["note"] = (', src)      # O18 留痕
        self.assertIn("def _open_batch_review", src)
        self.assertIn("BatchReviewDialog(", src)
        self.assertIn('_detail_stack.addWidget(self.review_page)', src)
        self.assertIn('"min_ms"', Path("xccv/engine/result_json.py")
                      .read_text(encoding="utf-8"))           # O13


if __name__ == "__main__":
    unittest.main()
