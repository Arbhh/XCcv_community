"""V0.7 A1：无监督异常检测（PatchCore-lite）回归锁。

合成数据（纯噪声纹理 OK vs 同纹理+亮块 NG）+ 真实 resnet18 离线权重：
训练收敛 → OK 分低 NG 分高 → 异常区命中。CPU 跑得动（image_size 128）。
"""

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from xccv.engine.dataset.base import TaskType
from xccv.engine.dataset.prepare import prepare
from xccv.engine.metrics.batch_match import collect_match_stats  # noqa: F401
from xccv.engine.models.anomaly import (
    AnomalyModel,
    _z_from_conf,
    image_ap,
    image_auroc,
)
from xccv.engine.models.registry import create_model, list_models

_IMAGE_SIZE = 128
_N_OK = 6
_N_NG = 4


def _texture(rng) -> np.ndarray:
    """固定纹理（平滑噪声，模拟均质工业表面，BGR 3 通道）。"""
    base = rng.normal(0, 1, (_IMAGE_SIZE // 8, _IMAGE_SIZE // 8)) * 40 + 120
    img = cv2.resize(base.astype(np.float32), (_IMAGE_SIZE, _IMAGE_SIZE),
                     interpolation=cv2.INTER_CUBIC)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _make_dataset(root: Path, seed: int = 7):
    rng = np.random.default_rng(seed)
    ok_dir = root / "OK"
    ng_dir = root / "NG"
    ok_dir.mkdir(parents=True)
    ng_dir.mkdir(parents=True)
    for i in range(_N_OK):
        img = _texture(rng)
        cv2.imwrite(str(ok_dir / f"ok_{i}.png"), img)
    for i in range(_N_NG):
        img = _texture(rng)
        x, y = 40 + i * 8, 40
        img[y:y + 28, x:x + 28] = 245          # 亮块异常
        cv2.imwrite(str(ng_dir / f"ng_{i}.png"), img)
    return ok_dir, ng_dir


class AnomalyEngineTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        _make_dataset(root)
        cls.model = create_model("patchcore", TaskType.ANOMALY, 2,
                                 device="cpu", use_pretrained=True)
        cfg = cls.model._make_test_cfg(root) if hasattr(
            cls.model, "_make_test_cfg") else None
        from xccv.engine.models.base import TrainConfig

        cfg = TrainConfig(
            task=TaskType.ANOMALY, dataset_dir=root,
            output_dir=root / "run", model_name="patchcore",
            epochs=1, image_size=_IMAGE_SIZE, device="cpu", workers=0,
            seed=7, use_pretrained=True)
        cls.cfg = cfg
        cls.metrics = cls.model.train(cfg, reporter=None)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_registry_and_listing(self):
        self.assertIsInstance(self.model, AnomalyModel)
        names = [m["name"] for m in list_models(TaskType.ANOMALY)]
        self.assertIn("patchcore", names)

    def test_weights_saved_and_reloaded(self):
        xva = Path(self.cfg.output_dir) / "best.pt"
        self.assertTrue(xva.is_file(), "训练应收尾产出加密 best.xva")
        from xccv.engine.models import load_model

        m2 = load_model(xva, device="cpu")
        self.assertIsInstance(m2, AnomalyModel)
        self.assertEqual(m2.image_size, _IMAGE_SIZE)
        self.assertGreater(m2.memory.shape[0], 0)

    def test_ok_scores_lower_than_ng(self):
        """验证：NG 图像级 z 分数显著高于 OK 图（异常检出能力）。"""
        rng = np.random.default_rng(99)
        ok_scores, ng_scores = [], []
        for i in range(3):
            ok_scores.append(self.model.predict(_texture(rng)).probs[0])
            img = _texture(rng)
            img[40:70, 60:90] = 245
            ng_scores.append(self.model.predict(img).probs[0])
        self.assertGreater(min(ng_scores), max(ok_scores),
                           f"NG 分数应全面高于 OK：ok={ok_scores} ng={ng_scores}")

    def test_predict_boxes_on_anomaly_region(self):
        img = _texture(np.random.default_rng(1))
        img[40:80, 40:80] = 250
        pred = self.model.predict(img, conf=0.25)
        self.assertTrue(len(pred.boxes) > 0, "异常亮块应产出检出框")
        x1, y1, x2, y2 = pred.boxes[0]
        self.assertTrue(x1 < 100 and y1 < 100 and x2 > 30 and y2 > 30,
                        f"检出框应命中亮块区域: {pred.boxes[0]}")
        self.assertTrue(pred.masks is not None and len(pred.masks) == len(pred.boxes))
        self.assertEqual(pred.labels, [0] * len(pred.boxes))

    def test_metrics_have_auroc(self):
        self.assertGreaterEqual(self.metrics["final"]["auroc"], 0.999,
                                "合成数据 AUROC 应接近 1")
        self.assertIn("history", self.metrics)

    def test_prepare_anomaly_family(self):
        data = prepare(TaskType.ANOMALY, "folder",
                       Path(self.cfg.dataset_dir), val_ratio=0.25, seed=7)
        self.assertTrue(data.classes)            # OK/NG 目录名成类
        self.assertTrue(data.train_samples)
        self.assertTrue(data.val_samples)

    def test_z_from_conf_monotonic(self):
        self.assertAlmostEqual(_z_from_conf(0.25), 3.0, places=6)
        self.assertGreater(_z_from_conf(0.1), _z_from_conf(0.5))


class ScoreUtilsTest(unittest.TestCase):

    def test_auroc_perfect_and_degenerate(self):
        self.assertEqual(image_auroc([1, 2, 3, 4], [0, 0, 1, 1]), 1.0)
        self.assertEqual(image_auroc([1, 2, 3, 4], [1, 1, 1, 1]), 0.0)
        self.assertEqual(image_auroc([1.0, 1.0], [0, 1]), 0.5)   # 并列平均秩

    def test_ap_perfect(self):
        self.assertEqual(image_ap([9, 8, 1, 0.5], [1, 1, 0, 0]), 1.0)
        self.assertEqual(image_ap([1, 2], []), 0.0)


if __name__ == "__main__":
    unittest.main()
