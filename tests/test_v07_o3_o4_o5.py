"""V0.7 O3/O4/O5/O12：Loss 判读、缩放 8×8 校验、灰度模式、划分环形图回归锁。"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from xccv.engine.io_utils import imread_unicode, imwrite_unicode
from xccv.engine.models.base import TrainConfig
from xccv.training.loss_diagnosis import diagnose
from xccv.training.scale_guard import scale_issue, scale_issue_text


def _hist(losses, metric=None, metric_key="map_loose"):
    out = []
    for i, l in enumerate(losses):
        e = {"epoch": i + 1, "train_loss": l}
        if metric and i < len(metric):
            e[metric_key] = metric[i]
        out.append(e)
    return out


class LossDiagnosisTest(unittest.TestCase):

    def test_underfit_detected(self):
        # 末段损失仍在明显线性下降（斜率/均值 < -0.05）
        h = _hist([2.0, 1.7, 1.45, 1.25, 1.05, 0.9, 0.78, 0.66, 0.56, 0.48])
        d = diagnose(h)
        self.assertEqual(d["kind"], "underfit")
        self.assertIn("欠拟合", d["text"])

    def test_overfit_detected(self):
        # 验证指标先升后明显回落（>10%），训练损失继续降
        losses = [1.0 - 0.05 * i for i in range(10)]     # 持续下降
        metric = [0.2, 0.4, 0.6, 0.75, 0.82, 0.85, 0.86, 0.85, 0.82, 0.74]
        d = diagnose(_hist(losses, metric))
        self.assertEqual(d["kind"], "overfit")
        self.assertIn("过拟合", d["text"])
        self.assertIn("best", d["text"])

    def test_nan_reported(self):
        d = diagnose(_hist([1.0, 0.8, 0.6, float("nan"), 0.4, 0.3, 0.2]))
        self.assertEqual(d["kind"], "nan")
        self.assertIn("NaN", d["text"])

    def test_insufficient_epochs(self):
        d = diagnose(_hist([1.0, 0.9, 0.8]))
        self.assertEqual(d["kind"], "insufficient")

    def test_normal_convergence(self):
        losses = [1.0, 0.6, 0.45, 0.4, 0.38, 0.37, 0.365, 0.36]
        metric = [0.5, 0.8, 0.85, 0.86, 0.865, 0.866, 0.866, 0.867]
        d = diagnose(_hist(losses, metric))
        self.assertEqual(d["kind"], "ok")
        self.assertIn("正常", d["text"])

    def test_empty_history(self):
        self.assertEqual(diagnose([])["kind"], "empty")
        self.assertEqual(diagnose(None)["kind"], "empty")


class ScaleGuardTest(unittest.TestCase):

    class _Sh:
        def __init__(self, label, box, w=1000, h=800):
            from xccv.engine.dataset.base import Shape

            self.shape = Shape(label=label, shape_type="rectangle",
                               points=[[box[0], box[1]], [box[2], box[3]]])
            self.width, self.height = w, h
            self.shapes = [self.shape]

    def test_stretch_scaling(self):
        # 1000×800 → 500×400（等比一半）：20px 短边 → 10px
        samples = [self._Sh("d", (0, 0, 100, 20))]
        stat = scale_issue(samples, "stretch", 500, 400)
        self.assertAlmostEqual(stat["median"], 10.0, places=1)

    def test_letterbox_scaling(self):
        # 1000×800 长边 1000 → imgsz 640：k=0.64，20px → 12.8
        samples = [self._Sh("d", (0, 0, 100, 20))]
        stat = scale_issue(samples, "letterbox", 640, 640)
        self.assertAlmostEqual(stat["median"], 12.8, places=1)

    def test_issue_text_threshold(self):
        # 95% 实例缩放后 <8px（1px 短边 ×100 张）
        samples = [self._Sh("d", (0, 0, 200, 1)) for _ in range(95)]
        samples += [self._Sh("d", (0, 0, 200, 200)) for _ in range(5)]
        stat = scale_issue(samples, "stretch", 640, 640)
        self.assertGreater(stat["ratio"], 0.5)
        detail, suggestion = scale_issue_text(stat)
        self.assertIn("<8px", detail)
        self.assertIn("切图训练", suggestion)

    def test_no_issue_below_threshold(self):
        samples = [self._Sh("d", (0, 0, 200, 100))]
        self.assertIsNone(scale_issue_text(scale_issue(samples, "stretch", 640, 640)))


class GrayModeTest(unittest.TestCase):

    def test_imread_gray_returns_3ch(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.png"
            img = np.zeros((16, 16, 3), np.uint8)
            img[:, :, 0] = 200          # 蓝通道有内容：灰度化应保留亮度
            imwrite_unicode(str(p), img)
            color = imread_unicode(str(p))
            gray = imread_unicode(str(p), color=False)
            self.assertEqual(color.shape, (16, 16, 3))
            self.assertEqual(gray.shape, (16, 16, 3))     # 复制回 3 通道
            # 灰度图三通道一致（亮度化后）
            self.assertTrue(np.array_equal(gray[:, :, 0], gray[:, :, 1]))
            self.assertTrue(np.array_equal(gray[:, :, 1], gray[:, :, 2]))

    def test_config_snapshot_carries_color_mode(self):
        from xccv.engine.models.base import _with_run_meta
        from datetime import datetime

        cfg = TrainConfig(task="detection", dataset_dir=Path("."),
                          output_dir=Path("."), model_name="yolo11n",
                          color_mode="gray")
        m = {}
        _with_run_meta(m, cfg, datetime.now(), datetime.now(), "completed")
        self.assertEqual(m["config"]["color_mode"], "gray")

    def test_rcnn_dataset_grayscale_flag(self):
        # RcnnDataset 接受 grayscale 并以同口径读图（行为锁：构造不崩+读图 3 通道）
        from xccv.engine.models.rcnn import RcnnDataset

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.png"
            imwrite_unicode(str(p), np.full((32, 32, 3), 128, np.uint8))
            from xccv.engine.dataset.base import Sample

            s = Sample(image_path=p, width=32, height=32)
            ds = RcnnDataset([s], ["d"], "detection", train=False, grayscale=True)
            img, target = ds[0]
            self.assertEqual(tuple(img.shape), (3, 32, 32))


class ManifestTest(unittest.TestCase):
    """E2：run_manifest 序列化回归（f7e 实锤坑：asdict 含 Path → TypeError
    静默写入失败）。"""

    def test_manifest_serializes_with_path_fields(self):
        import json
        import tempfile
        from pathlib import Path as _P

        from xccv.engine.models.manifest import build_run_manifest

        cfg = TrainConfig(task="detection", dataset_dir=_P("."),
                          output_dir=_P("."), model_name="yolo11n")
        with tempfile.TemporaryDirectory() as td:
            fp = _P(td) / "ds"        # dataset_dir 为 Path：指纹函数可读目录
            fp.mkdir()
            cfg.dataset_dir = fp
            m = build_run_manifest(cfg, {"final": {"map50": 0.5}})
            raw = json.dumps(m, ensure_ascii=False)   # 不抛 TypeError 即过
            self.assertIn("xccv-run-manifest", raw)
            self.assertEqual(m["config"]["dataset_dir"], str(fp))


class SplitDonutTest(unittest.TestCase):
    """O12：环形图随统计刷新（offscreen 冒烟——构造已在 ProjectWindow 内）。"""

    def test_update_split_smoke(self):
        from xccv.training.charts import SplitDonutCanvas

        c = SplitDonutCanvas()
        c.update_split(80, 20, 0.2)       # 四段
        c.update_split(80, 0, 0.2)        # 仅已标
        c.update_split(0, 0, 0.2)         # 空数据占位
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
