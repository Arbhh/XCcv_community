"""V0.7 O8：`__ignore__` 忽略区域近似语义回归锁（训练剔除 + 评估同口径）。"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from xccv.engine.dataset.base import Shape, TaskType
from xccv.engine.dataset.ignore import (IGNORE_LABEL, filter_ignore_shapes,
                                        suppress_boxes)
from xccv.engine.dataset.prepare import prepare
from xccv.engine.dataset.reader import scan_labelme_dataset
from xccv.engine.io_utils import imwrite_unicode
from xccv.engine.metrics.batch_match import collect_match_stats, read_gt_shapes
from xccv.engine.models.yolo_data import build_labels_from_shapes
from xccv.training.data_health import LEVEL_INFO, run_health_check
from xccv.training.dataset_stats import collect_stats


def _write_png(path: Path):
    imwrite_unicode(str(path), np.full((8, 8, 3), 200, dtype=np.uint8))


def _write_json(path: Path, shapes: list, w: int = 200, h: int = 200):
    path.write_text(json.dumps({
        "version": "5.8.1", "flags": {}, "shapes": shapes,
        "imagePath": path.with_suffix(".png").name, "imageData": None,
        "imageHeight": h, "imageWidth": w,
    }), encoding="utf-8")


def _rect(label: str, box) -> dict:
    return {"label": label, "points": [[box[0], box[1]], [box[2], box[3]]],
            "shape_type": "rectangle"}


def _mk_sample_dataset(td: Path):
    """三张图：img1 正常 GT+ignore 区（互不相交）；img2 GT 全在 ignore 区内；
    img3 GT 与 ignore 区部分重叠（30%，保留）。"""
    d = td / "缺陷"
    d.mkdir()
    cases = {
        "img1": [_rect("黑点", (10, 10, 60, 60)), _rect(IGNORE_LABEL, (100, 100, 180, 180))],
        "img2": [_rect("黑点", (110, 110, 160, 160)), _rect(IGNORE_LABEL, (100, 100, 180, 180))],
        "img3": [_rect("黑点", (20, 20, 80, 80)), _rect(IGNORE_LABEL, (70, 70, 150, 150))],
    }
    for name, shapes in cases.items():
        _write_png(d / f"{name}.png")
        _write_json(d / f"{name}.json", shapes)
    return d, cases


class IgnoreHelperTest(unittest.TestCase):

    def test_fully_inside_dropped(self):
        shapes = [Shape("黑点", "rectangle", [[110, 110], [160, 160]]),
                  Shape(IGNORE_LABEL, "rectangle", [[100, 100], [180, 180]])]
        kept, n_reg, n_drop = filter_ignore_shapes(shapes)
        self.assertEqual((n_reg, n_drop), (1, 1))
        self.assertEqual(kept, [])   # GT 全包被剔除 + ignore 形状移除 → 空

    def test_outside_kept(self):
        shapes = [Shape("黑点", "rectangle", [[10, 10], [60, 60]]),
                  Shape(IGNORE_LABEL, "rectangle", [[100, 100], [180, 180]])]
        kept, n_reg, n_drop = filter_ignore_shapes(shapes)
        self.assertEqual((n_reg, n_drop), (1, 0))
        self.assertEqual(len(kept), 1)

    def test_partial_overlap_below_ratio_kept(self):
        # GT (20,20)-(80,80)=3600px²，与区域 (70,70)-(150,150) 交 10×10=100 → 2.8%
        shapes = [Shape("黑点", "rectangle", [[20, 20], [80, 80]]),
                  Shape(IGNORE_LABEL, "rectangle", [[70, 70], [150, 150]])]
        _, _, n_drop = filter_ignore_shapes(shapes)
        self.assertEqual(n_drop, 0)

    def test_partial_overlap_above_ratio_dropped(self):
        # GT (20,20)-(100,100)=6400px²，与区域 (40,40)-(100,100) 交 60×60=3600
        # → 56.25% > 50% → 剔除
        shapes = [Shape("黑点", "rectangle", [[20, 20], [100, 100]]),
                  Shape(IGNORE_LABEL, "rectangle", [[40, 40], [100, 100]])]
        _, _, n_drop = filter_ignore_shapes(shapes)
        self.assertEqual(n_drop, 1)

    def test_polygon_region_by_bbox(self):
        # 多边形区域按外接框近似（文档声明的口径）：GT 全在外接框内 → 剔除
        shapes = [Shape("黑点", "rectangle", [[110, 110], [160, 160]]),
                  Shape(IGNORE_LABEL, "polygon",
                        [[100, 100], [180, 100], [180, 180], [140, 130]])]
        kept, n_reg, n_drop = filter_ignore_shapes(shapes)
        self.assertEqual((n_reg, n_drop), (1, 1))
        self.assertEqual(kept, [])

    def test_no_regions_noop(self):
        shapes = [Shape("黑点", "rectangle", [[10, 10], [60, 60]])]
        kept, n_reg, n_drop = filter_ignore_shapes(shapes)
        self.assertEqual((n_reg, n_drop, len(kept)), (0, 0, 1))

    def test_suppress_boxes(self):
        regions = [(100, 100, 180, 180)]
        boxes = [[110, 110, 160, 160], [10, 10, 60, 60],
                 [60, 60, 110, 110]]   # 与区域仅 10×10 交（4%），保留
        labels = ["a", "b", "c"]
        kb, kl = suppress_boxes(boxes, labels, regions)
        self.assertEqual(kl, ["b", "c"])


class ReaderClassesTest(unittest.TestCase):

    def test_ignore_label_not_a_class(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _write_png(d / "a.png")
            _write_json(d / "a.json", [_rect("黑点", (10, 10, 60, 60)),
                                       _rect(IGNORE_LABEL, (100, 100, 180, 180))])
            samples, classes, _ = scan_labelme_dataset(d)
            self.assertEqual(classes, ["黑点"])
            # ignore 形状保留在样本上（评估层/网格角标还要用）
            labels = [s.label for s in samples[0].shapes]
            self.assertIn(IGNORE_LABEL, labels)

    def test_classes_txt_ignore_filtered(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _write_png(d / "a.png")
            _write_json(d / "a.json", [_rect("黑点", (10, 10, 60, 60))])
            (d / "classes.txt").write_text("黑点\n__ignore__\n", encoding="utf-8")
            _, classes, _ = scan_labelme_dataset(d)
            self.assertEqual(classes, ["黑点"])


class PrepareFilterTest(unittest.TestCase):

    def test_prepare_drops_overlapping_gt(self):
        with tempfile.TemporaryDirectory() as td:
            d, cases = _mk_sample_dataset(Path(td))
            data = prepare(TaskType.DETECTION, "yolo", d, val_ratio=0.0)
            self.assertEqual(data.classes, ["黑点"])
            self.assertEqual(data.ignore_regions, 3)
            self.assertEqual(data.ignore_dropped, 1)   # 仅 img2 全包被剔除
            # val_ratio=0 时稀有类兜底仍会把一张图移入验证集 → 按 train+val 全集断言
            by_stem = {Path(s.image_path).stem: s
                       for s in data.train_samples + data.val_samples}
            self.assertEqual(len(by_stem), 3)
            self.assertEqual([s.label for s in by_stem["img1"].shapes], ["黑点"])
            self.assertEqual(by_stem["img2"].shapes, [])   # 全包 → 剔光
            self.assertEqual([s.label for s in by_stem["img3"].shapes], ["黑点"])

    def test_yolo_labels_skip_ignore(self):
        shapes = [Shape("黑点", "rectangle", [[10, 10], [60, 60]]),
                  Shape(IGNORE_LABEL, "rectangle", [[100, 100], [180, 180]])]
        lb = build_labels_from_shapes(shapes, ["黑点"], "detection", 200, 200)
        self.assertEqual(lb["cls"].shape[0], 1)   # ignore 不进标签


class BatchMatchIgnoreTest(unittest.TestCase):

    def test_gt_exclude_ignore_and_overlap(self):
        with tempfile.TemporaryDirectory() as td:
            d, _ = _mk_sample_dataset(Path(td))
            boxes, labels = read_gt_shapes(d / "img2.json")
            self.assertEqual((boxes, labels), ([], []))     # 全包 GT 剔光
            boxes, labels = read_gt_shapes(d / "img1.json")
            self.assertEqual(labels, ["黑点"])               # ignore 标记不是 GT

    def test_det_in_ignore_region_not_fp(self):
        with tempfile.TemporaryDirectory() as td:
            d, _ = _mk_sample_dataset(Path(td))
            results = [{"path": str(d / "img1.png"),
                        "detections": [
                            {"label": "黑点", "score": 0.9, "bbox": [10, 10, 60, 60]},
                            {"label": "黑点", "score": 0.8, "bbox": [120, 120, 170, 170]}]}]
            stats = collect_match_stats(results)
            self.assertEqual(stats["gt_total"], 1)
            self.assertEqual(len(stats["green"]), 1)
            self.assertEqual(stats["red_defect"], [])   # 区域内检出不算 FP

    def test_ok_image_ignore_zone_dets_not_counted(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "OK"
            d.mkdir()
            _write_png(d / "ok0.png")
            _write_json(d / "ok0.json", [_rect(IGNORE_LABEL, (100, 100, 180, 180))])
            results = [{"path": str(d / "ok0.png"),
                        "detections": [{"label": "黑点", "score": 0.9,
                                        "bbox": [120, 120, 170, 170]},
                                       {"label": "黑点", "score": 0.7,
                                        "bbox": [10, 10, 60, 60]}]}]
            stats = collect_match_stats(results)
            self.assertEqual(stats["n_ok_images"], 1)
            self.assertEqual(stats["red_ok"], [0.7])    # 区域内检出不算 OK 误检


class VisibilityTest(unittest.TestCase):

    def test_health_check_reports_ignore(self):
        with tempfile.TemporaryDirectory() as td:
            _mk_sample_dataset(Path(td))
            rep = run_health_check("detection", Path(td))
            issues = [i for i in rep.issues if i.code == "ignore_regions"]
            self.assertEqual(len(issues), 1)
            self.assertEqual(issues[0].level, LEVEL_INFO)
            self.assertIn("3 个", issues[0].detail)

    def test_stats_count_ignore(self):
        with tempfile.TemporaryDirectory() as td:
            d, _ = _mk_sample_dataset(Path(td))
            stats = collect_stats(TaskType.DETECTION, d)
            self.assertEqual(stats.n_ignore, 3)
            self.assertEqual(stats.class_counts.get(IGNORE_LABEL), None)


if __name__ == "__main__":
    unittest.main()
