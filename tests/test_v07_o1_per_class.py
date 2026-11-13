"""V0.7 O1：逐类精确率/召回率表回归锁。

聚合逻辑（engine.metrics.per_class）+ 两家族评估器的逐类 AP@50 提取 +
训练窗口三处调用点接线（源码锁）。不跑真训练——AP 提取用假数组/桩。
"""

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from xccv.engine.metrics.per_class import (
    coco_per_class_ap50,
    per_class_rows,
    ultralytics_per_class_ap50,
)


def _cm_a_b() -> list:
    """三类手工矩阵（[gt, pred] 约定，末行/末列=背景）：
    A: TP8/FN2/FP3；B: TP5/FN0/FP1；C 全零。"""
    return [[8, 0, 0, 2],
            [0, 5, 0, 0],
            [0, 0, 0, 0],
            [3, 1, 0, 0]]


class PerClassRowsTest(unittest.TestCase):

    def test_aggregation_matches_manual_counts(self):
        rows = per_class_rows(_cm_a_b(), ["A", "B", "C"])
        a, b, c = rows
        self.assertEqual((a["tp"], a["fp"], a["fn"]), (8, 3, 2))
        self.assertEqual((a["gt"], a["pred"]), (10, 11))
        self.assertAlmostEqual(a["precision"], 8 / 11, places=4)
        self.assertAlmostEqual(a["recall"], 0.8, places=4)
        self.assertEqual((b["tp"], b["fp"], b["fn"]), (5, 1, 0))
        self.assertAlmostEqual(b["recall"], 1.0, places=4)
        self.assertEqual((c["gt"], c["pred"], c["tp"]), (0, 0, 0))

    def test_undefined_ratios_are_none(self):
        rows = per_class_rows(_cm_a_b(), ["A", "B", "C"])
        self.assertIsNone(rows[2]["precision"])   # 零检出分母
        self.assertIsNone(rows[2]["recall"])      # 零实例分母
        self.assertIsNotNone(rows[0]["precision"])

    def test_ap50_column_alignment_and_none(self):
        rows = per_class_rows(_cm_a_b(), ["A", "B", "C"], ap50_by_class=[0.9, 0.5])
        self.assertAlmostEqual(rows[0]["ap50"], 0.9, places=4)
        self.assertAlmostEqual(rows[1]["ap50"], 0.5, places=4)
        self.assertIsNone(rows[2]["ap50"])        # 缺项 → None（UI 显示 -）
        rows2 = per_class_rows(_cm_a_b(), ["A", "B", "C"])
        self.assertTrue(all(r["ap50"] is None for r in rows2))

    def test_shape_mismatch_raises_valueerror(self):
        with self.assertRaises(ValueError):
            per_class_rows([[1]], ["A", "B"])


class CocoPerClassAp50Test(unittest.TestCase):

    def test_valid_entries_averaged_at_iou_050(self):
        # [T=3 IoU(0.1,0.25,0.5), R=5, K=2 类, A=1, M=1]
        prec = np.full((3, 5, 2, 1, 1), -1.0)
        prec[2, :, 0, 0, 0] = [0.8, 0.7, -1.0, 0.9, 0.6]  # 类 0：哨兵剔除后均值 0.75
        self.assertAlmostEqual(coco_per_class_ap50(prec)[0], 0.75, places=4)

    def test_class_without_valid_entries_is_zero(self):
        prec = np.full((3, 5, 2, 1, 1), -1.0)
        out = coco_per_class_ap50(prec)
        self.assertEqual(out[1], 0.0)

    def test_shape_guard(self):
        with self.assertRaises(ValueError):
            coco_per_class_ap50(np.zeros((2, 5, 2, 1, 1)))


class UltralyticsPerClassAp50Test(unittest.TestCase):

    def test_missing_classes_filled_zero(self):
        ns = SimpleNamespace(ap50=[0.9, 0.5], ap_class_index=[0, 2])
        out = ultralytics_per_class_ap50(ns)
        self.assertEqual(out, [0.9, 0.0, 0.5])    # 类 1 未出现在验证集 → 0.0

    def test_absent_or_empty_metric_returns_none(self):
        self.assertIsNone(ultralytics_per_class_ap50(SimpleNamespace()))
        self.assertIsNone(ultralytics_per_class_ap50(
            SimpleNamespace(ap50=[], ap_class_index=[])))

    def test_broken_metric_returns_none(self):
        class Boom:
            @property
            def ap50(self):
                raise RuntimeError("boom")
        self.assertIsNone(ultralytics_per_class_ap50(Boom()))


class TrainingWindowWiringTest(unittest.TestCase):
    """UI 接线源码锁：表格构建 + 三处调用点（开始清空/历史详情/训练完成）。"""

    def test_call_sites_present(self):
        src = Path("xccv/training/training_window.py").read_text(encoding="utf-8")
        self.assertIn("self.per_class_table = QTableWidget(0, 9)", src)
        self.assertEqual(src.count("self._update_per_class_table("), 5,
                         "1 处定义 + 4 处调用（开始/进行中/历史详情/完成）")
        self.assertIn("from ..engine.metrics.per_class import per_class_rows", src)

    def test_engine_per_class_key_emitted(self):
        # R-CNN/YOLO 评估器的 additive 键（源码锁，不跑真训练）
        rcnn_src = Path("xccv/engine/models/rcnn.py").read_text(encoding="utf-8")
        self.assertIn('out["per_class_ap50"] = coco_per_class_ap50(prec)', rcnn_src)
        yolo_src = Path("xccv/engine/models/yolo.py").read_text(encoding="utf-8")
        self.assertIn('out["per_class_ap50"] = per_ap', yolo_src)


if __name__ == "__main__":
    unittest.main()
