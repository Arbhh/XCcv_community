"""V0.7 O11：训练平台数据网格回归锁（扫描/筛选/角标/缩略/组件冒烟）。"""

import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from xccv.engine.io_utils import imwrite_unicode
from xccv.training.data_grid import (GridEntry, badge_text, filter_grid_entries,
                                     make_thumbnail, scan_grid_entries)


def _write_png(path: Path, w: int = 40, h: int = 30):
    rng = np.random.default_rng(w * h)
    imwrite_unicode(str(path),
                    rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


def _write_json(path: Path, labels: list):
    import json

    path.write_text(json.dumps({
        "version": "5.8.1", "flags": {},
        "shapes": [{"label": lab, "points": [[1, 1], [20, 20]],
                    "shape_type": "rectangle"} for lab in labels],
        "imagePath": path.with_suffix(".png").name, "imageData": None,
        "imageHeight": 30, "imageWidth": 40,
    }), encoding="utf-8")


def _make_dataset(td: Path):
    det = td / "缺陷"
    det.mkdir()
    _write_png(det / "a.png")
    _write_json(det / "a.json", ["黑点"])
    _write_png(det / "b.png")
    _write_json(det / "b.json", ["__ignore__", "__ignore__", "黑点"])
    _write_png(det / "c.png")                       # 未标注（缺陷目录）
    ok = td / "OK"
    ok.mkdir()
    _write_png(ok / "ok0.png")                      # OK 目录未标注
    _write_png(ok / "ok1.png")
    _write_json(ok / "ok1.json", [])                # 已审无缺陷


class ScanTest(unittest.TestCase):

    def test_scan_states(self):
        with tempfile.TemporaryDirectory() as td:
            _make_dataset(Path(td))
            es = scan_grid_entries(Path(td))
            by = {Path(e.path).name: e for e in es}
            self.assertEqual(len(by), 5)
            self.assertEqual(by["a.png"].state, "labeled")
            self.assertEqual(by["a.png"].labels, ("黑点",))
            self.assertEqual(by["b.png"].n_ignore, 2)
            self.assertEqual(by["b.png"].labels, ("黑点",))
            self.assertEqual(by["c.png"].state, "unlabeled")
            self.assertFalse(by["c.png"].is_ok_dir)
            self.assertEqual(by["ok0.png"].state, "unlabeled")
            self.assertTrue(by["ok0.png"].is_ok_dir)
            self.assertEqual(by["ok1.png"].state, "clean")

    def test_bad_json_degrades(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _write_png(d / "x.png")
            (d / "x.json").write_text("{broken", encoding="utf-8")
            es = scan_grid_entries(d)
            self.assertEqual(es[0].state, "unlabeled")

    def test_hidden_dir_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            hid = Path(td) / ".hidden"
            hid.mkdir()
            _write_png(hid / "h.png")
            self.assertEqual(scan_grid_entries(Path(td)), [])


class FilterBadgeTest(unittest.TestCase):

    def _entries(self, td):
        _make_dataset(Path(td))
        return scan_grid_entries(Path(td))

    def test_filter_ok(self):
        with tempfile.TemporaryDirectory() as td:
            es = self._entries(Path(td))
            got = filter_grid_entries(es, only_ok=True)
            names = {Path(e.path).name for e in got}
            self.assertEqual(names, {"ok0.png", "ok1.png"})

    def test_filter_unlabeled_excludes_clean(self):
        with tempfile.TemporaryDirectory() as td:
            es = self._entries(Path(td))
            got = filter_grid_entries(es, only_unlabeled=True)
            names = {Path(e.path).name for e in got}
            self.assertIn("ok0.png", names)
            self.assertIn("c.png", names)
            self.assertNotIn("ok1.png", names)   # clean ≠ 未标注

    def test_filter_label(self):
        with tempfile.TemporaryDirectory() as td:
            es = self._entries(Path(td))
            got = filter_grid_entries(es, label="黑点")
            names = {Path(e.path).name for e in got}
            self.assertEqual(names, {"a.png", "b.png"})

    def test_filter_ignore(self):
        with tempfile.TemporaryDirectory() as td:
            es = self._entries(Path(td))
            got = filter_grid_entries(es, only_ignore=True)
            self.assertEqual([Path(e.path).name for e in got], ["b.png"])

    def test_badges(self):
        e_ok = GridEntry("x", is_ok_dir=True)
        e_ign = GridEntry("y", n_ignore=2, labels=("a",))
        self.assertEqual(badge_text(e_ok), "[OK]")
        self.assertEqual(badge_text(e_ign), "⊘2")


class ThumbTest(unittest.TestCase):

    def test_letterbox_keeps_aspect(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.png"
            _write_png(p, w=80, h=40)
            arr = make_thumbnail(str(p), size=64)
            self.assertEqual(arr.shape, (64, 64, 3))
            # 非灰底区域存在（图片内容贴入）
            self.assertTrue((arr != 235).any())

    def test_missing_returns_none(self):
        self.assertIsNone(make_thumbnail("Z:/no/such/file.png"))


class WidgetSmokeTest(unittest.TestCase):
    """offscreen 组件冒烟：扫描→填充→筛选→缩略投递全链。"""

    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])
        cls._tmp = tempfile.TemporaryDirectory()
        cls._td = Path(cls._tmp.name)
        _make_dataset(cls._td)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _pump(self, widget, seconds=5.0):
        """事件循环等待后台线程产物（offscreen 测试先例模式）。"""
        end = time.time() + seconds
        while time.time() < end:
            self.app.processEvents()
            if widget._scan_worker is None and widget._thumb_worker is None:
                break
            time.sleep(0.02)
        self.app.processEvents()

    def test_full_chain(self):
        from xccv.training.data_grid import get_data_grid_class

        w = get_data_grid_class()()
        w.set_dataset(self._td)
        self._pump(w)
        self.assertEqual(w.list.count(), 5)
        self.assertIn("显示 5 / 共 5 张", w.lbl_count.text())
        # 角标：OK 目录未标注 + 已审无缺陷（缺陷目录的未标注 c.png 无角标）
        badges = [w.list.item(i).text() for i in range(w.list.count())]
        self.assertEqual(sum("[OK]" in t for t in badges), 2)
        self.assertEqual(sum("⊘2" in t for t in badges), 1)
        # 缩略图全部解码入缓存（懒加载完成）
        self.assertEqual(len(w._cache), 5)
        # 只看 OK 筛选
        w.cb_ok.setChecked(True)
        self.assertEqual(w.list.count(), 2)
        w.cb_ok.setChecked(False)
        # 标签筛选
        idx = w.cmb_label.findText("黑点")
        w.cmb_label.setCurrentIndex(idx)
        self.assertEqual(w.list.count(), 2)
        w.cmb_label.setCurrentIndex(0)
        # 二次 set_dataset 重建（旧线程停干净）
        w.set_dataset(self._td)
        self._pump(w)
        self.assertEqual(w.list.count(), 5)
        w.shutdown()


if __name__ == "__main__":
    unittest.main()
