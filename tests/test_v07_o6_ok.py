"""V0.7 O6/O9：OK 负样本可见性回归锁（体检第 7 项 + 三处呈现）。"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from xccv.engine.dataset.reader import NEGATIVE_DIR_NAMES
from xccv.engine.io_utils import imwrite_unicode
from xccv.training.data_health import run_health_check, LEVEL_INFO
from xccv.training.dataset_stats import collect_stats


def _write_png(path: Path):
    """1×1 灰图（image_size 解析需要真图片，坑：假字节全被静默跳过）。"""
    imwrite_unicode(str(path), np.full((8, 8, 3), 200, dtype=np.uint8))


def _write_labelme_json(path: Path, label: str, box=(10, 10, 60, 60)):
    path.write_text(json.dumps({
        "version": "5.8.1", "flags": {},
        "shapes": [{"label": label, "points": [[box[0], box[1]],
                                               [box[2], box[3]]],
                    "shape_type": "rectangle"}],
        "imagePath": path.with_suffix(".png").name, "imageData": None,
        "imageHeight": 200, "imageWidth": 200,
    }), encoding="utf-8")


def _make_dataset(td: Path, n_defect: int = 2, n_ok: int = 2):
    d = td / "缺陷"
    d.mkdir()
    for i in range(n_defect):
        _write_png(d / f"def{i}.png")
        _write_labelme_json(d / f"def{i}.json", "黑点")
    if n_ok:
        ok = td / "OK"
        ok.mkdir()
        for i in range(n_ok):
            _write_png(ok / f"ok{i}.png")


class OkRatioHealthTest(unittest.TestCase):

    def test_ok_in_band_no_issue(self):
        with tempfile.TemporaryDirectory() as td:
            _make_dataset(Path(td))  # 2 缺陷 / 2 OK = 1:1
            rep = run_health_check("detection", Path(td))
            self.assertFalse([i for i in rep.issues
                              if i.code in ("ok_missing", "ok_ratio")])

    def test_no_ok_gives_hint(self):
        with tempfile.TemporaryDirectory() as td:
            _make_dataset(Path(td), n_ok=0)
            rep = run_health_check("detection", Path(td))
            issues = [i for i in rep.issues if i.code == "ok_missing"]
            self.assertEqual(len(issues), 1)
            self.assertEqual(issues[0].level, LEVEL_INFO)
            self.assertIn("OK", issues[0].suggestion)

    def test_ok_too_many_hinted(self):
        with tempfile.TemporaryDirectory() as td:
            _make_dataset(Path(td), n_defect=1, n_ok=10)  # 10:1
            rep = run_health_check("detection", Path(td))
            codes = [i.code for i in rep.issues]
            self.assertIn("ok_ratio", codes)

    def test_classification_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            _make_dataset(Path(td), n_ok=0)
            rep = run_health_check("classification", Path(td))
            self.assertFalse([i for i in rep.issues
                              if i.code.startswith("ok_")])


class OkVisibilityTest(unittest.TestCase):

    def test_collect_stats_counts_ok(self):
        with tempfile.TemporaryDirectory() as td:
            _make_dataset(Path(td), n_defect=2, n_ok=3)
            st = collect_stats("detection", Path(td))
            self.assertEqual(st.n_ok, 3)
            self.assertEqual(st.n_images, 2)

    def test_summarize_metrics_ok_line(self):
        from xccv.training.training_window import ProjectWindow

        text = ProjectWindow._summarize_metrics({
            "final": {"map_loose": 0.8, "neg_val_images": 5,
                      "neg_fp_per_img": 0.12, "neg_fp_img_rate": 0.1},
            "neg_train_images": 41,
        })
        self.assertIn("OK 负样本：训练 41 张 / 验证 5 张", text)
        self.assertIn("误检 0.12 框/张", text)

    def test_labelme_badge_and_menu_sources(self):
        src = Path("labelme/app.py").read_text(encoding="utf-8")
        self.assertIn("is_negative_dir_name", src)
        self.assertIn("[OK]", src)
        self.assertIn("_mark_dir_as_ok", src)
        self.assertIn("把所在文件夹设为 OK/良品目录", src)


if __name__ == "__main__":
    unittest.main()
