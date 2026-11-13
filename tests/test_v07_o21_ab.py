"""V0.7 O21：模型 A/B 批量比对回归锁（纯逻辑 + 对话框冒烟）。"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from xccv.engine.io_utils import imwrite_unicode
from xccv.testing.model_compare import (build_compare_payload,
                                        compare_batch, compare_image,
                                        compare_summary, score_dets)


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


def _det(label, score, box):
    return {"label": label, "score": score, "bbox": [float(v) for v in box]}


class ScoreTest(unittest.TestCase):

    def test_score_dets(self):
        gt_b = [[0, 0, 100, 100]]
        gt_l = ["黑点"]
        dets = [_det("黑点", 0.9, (0, 0, 100, 100)), _det("黑点", 0.5, (150, 150, 180, 180))]
        s = score_dets(dets, gt_b, gt_l)
        self.assertEqual((s["n"], s["hit"], s["fp"]), (2, 1, 1))


class CompareImageTest(unittest.TestCase):

    def test_a_better_by_hit(self):
        gt_b = [[0, 0, 100, 100]]
        gt_l = ["黑点"]
        rec = compare_image([_det("黑点", 0.9, (0, 0, 100, 100))],
                            [], gt_b, gt_l)
        self.assertEqual(rec["verdict"], "A优")

    def test_b_better_by_fp(self):
        gt_b = [[0, 0, 100, 100]]
        gt_l = ["黑点"]
        rec = compare_image(
            [_det("黑点", 0.9, (0, 0, 100, 100)), _det("黑点", 0.4, (150, 150, 180, 180))],
            [_det("黑点", 0.9, (0, 0, 100, 100))], gt_b, gt_l)
        self.assertEqual(rec["verdict"], "B优")

    def test_same_is_tie(self):
        gt_b = [[0, 0, 100, 100]]
        gt_l = ["黑点"]
        d = [_det("黑点", 0.9, (0, 0, 100, 100))]
        self.assertEqual(compare_image(d, list(d), gt_b, gt_l)["verdict"], "一致")

    def test_ok_image_compares_fp(self):
        # OK 图（GT 空）：B 有误检 A 无 → A 优
        rec = compare_image([], [_det("黑点", 0.6, (10, 10, 50, 50))], [], [])
        self.assertEqual(rec["verdict"], "A优")
        rec = compare_image([_det("黑点", 0.6, (10, 10, 50, 50))],
                            [_det("黑点", 0.6, (10, 10, 50, 50))], [], [])
        self.assertEqual(rec["verdict"], "一致")

    def test_no_gt_is_unjudged(self):
        rec = compare_image([_det("黑点", 0.9, (0, 0, 10, 10))], [], [], [],
                            gt_available=False)
        self.assertEqual(rec["verdict"], "未评估")


class CompareBatchTest(unittest.TestCase):

    def test_batch_records_and_summary(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _write_png(d / "a.png")
            _write_json(d / "a.json", [_rect("黑点", (0, 0, 100, 100))])
            _write_png(d / "ok.png")
            _write_json(d / "ok.json", [])
            _write_png(d / "nogt.png")
            results_a = [
                {"file": "a.png", "path": str(d / "a.png"),
                 "detections": [_det("黑点", 0.9, (0, 0, 100, 100))]},
                {"file": "ok.png", "path": str(d / "ok.png"),
                 "detections": [_det("黑点", 0.7, (5, 5, 40, 40))]},   # OK 误检
                {"file": "nogt.png", "path": str(d / "nogt.png"),
                 "detections": []},
            ]
            preds_b = {str(d / "a.png"): [],                    # A 命中 B 全漏
                       str(d / "ok.png"): []}                   # B 无误检 → B 优
            records = compare_batch(results_a, preds_b, 0.5)
            by = {r["file"]: r for r in records}
            self.assertEqual(by["a.png"]["verdict"], "A优")
            self.assertEqual(by["ok.png"]["verdict"], "B优")
            self.assertEqual(by["nogt.png"]["verdict"], "未评估")
            s = compare_summary(records)
            self.assertEqual(s["counts"], {"一致": 0, "A优": 1, "B优": 1,
                                           "未评估": 1})

    def test_payload_shape(self):
        records = [{"gt": 1, "a": {"n": 1, "hit": 1, "fp": 0},
                    "b": {"n": 0, "hit": 0, "fp": 0}, "verdict": "A优",
                    "gt_available": True, "path": "x", "file": "x.png"}]
        payload = build_compare_payload("a.xva", "b.xva", 0.25, 0.5, records)
        self.assertEqual(payload["schema"], 1)
        self.assertEqual(payload["model_a"], "a.xva")
        self.assertIn("records", payload)
        with tempfile.TemporaryDirectory() as td:
            from xccv.testing.model_compare import save_compare_json

            p = Path(td) / "比对.json"
            save_compare_json(p, payload)
            data = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(data["summary"]["counts"]["A优"], 1)


class DialogSmokeTest(unittest.TestCase):
    """offscreen 对话框冒烟：填充表格与统计，不真跑模型。"""

    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_fill_and_payload(self):
        from xccv.testing.model_compare_dialog import ModelCompareDialog

        results_a = [{"file": "a.png", "path": "Z:/no/a.png",
                      "detections": [_det("黑点", 0.9, (0, 0, 100, 100))]}]
        dlg = ModelCompareDialog(results_a, None, 0.25, 0.5, None, ["黑点"],
                                 model_a_name="a.xva")
        records = [{"gt": 1, "a": {"n": 1, "hit": 1, "fp": 0},
                    "b": {"n": 0, "hit": 0, "fp": 0}, "verdict": "A优",
                    "gt_available": True, "path": "Z:/no/a.png", "file": "a.png"}]
        dlg._fill_table(records)
        self.assertEqual(dlg.table.rowCount(), 1)
        self.assertEqual(dlg.table.item(0, 6).text(), "A优")
        self.assertIsNone(dlg.payload())


if __name__ == "__main__":
    unittest.main()
