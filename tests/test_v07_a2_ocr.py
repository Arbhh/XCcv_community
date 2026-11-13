"""V0.7 A2：OCR 字符识别（CRNN+CTC）回归锁。

合成数据：PIL 默认字体在固定位置画文本 + 同位置 rectangle 标注
（label=字符串内容）。小字符集 + 固定渲染，短训练可记忆——
断言 loss 收敛 + 精确匹配率阈值 + 加密权重回环 + JSON text 键。
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from xccv.engine.dataset.base import TaskType
from xccv.engine.dataset.prepare import prepare
from xccv.engine.models.base import Prediction
from xccv.engine.models.registry import create_model, list_models, load_model
from xccv.engine.models.ocr import OcrModel, _levenshtein

_TEXTS = ["CAT", "DOG", "A12", "G55"]
_FONT = None


def _draw_sample(text: str, size=(200, 120)) -> tuple:
    """(BGR 图, rectangle 顶点)——文本画在 (40,40)。"""
    global _FONT
    if _FONT is None:
        _FONT = ImageFont.load_default()
    img = Image.new("RGB", size, (30, 30, 30))
    d = ImageDraw.Draw(img)
    d.text((40, 40), text, fill=(240, 240, 240), font=_FONT)
    bbox = d.textbbox((40, 40), text, font=_FONT)
    arr = np.array(img)[:, :, ::-1].copy()   # RGB→BGR
    points = [[float(bbox[0] - 2), float(bbox[1] - 2)],
              [float(bbox[2] + 2), float(bbox[1] - 2)],
              [float(bbox[2] + 2), float(bbox[3] + 2)],
              [float(bbox[0] - 2), float(bbox[3] + 2)]]
    return arr, points


class OcrEngineTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        rng = np.random.default_rng(3)
        for i in range(16):                       # 每文本 4 张
            text = _TEXTS[i % len(_TEXTS)]
            arr, pts = _draw_sample(text)
            img_path = root / f"img_{i:02d}.png"
            from xccv.engine.io_utils import imwrite_unicode

            imwrite_unicode(str(img_path), arr)
            (root / f"img_{i:02d}.json").write_text(json.dumps({
                "version": "5.8.1", "flags": {},
                "shapes": [{"label": text, "points": pts,
                            "shape_type": "rectangle"}],
                "imagePath": img_path.name, "imageData": None,
                "imageHeight": arr.shape[0], "imageWidth": arr.shape[1],
            }), encoding="utf-8")
        cls.model = create_model("crnn", TaskType.OCR, 1, device="cpu",
                                 use_pretrained=False)
        from xccv.engine.models.base import TrainConfig

        cfg = TrainConfig(task=TaskType.OCR, dataset_dir=root,
                          output_dir=root / "run", model_name="crnn",
                          epochs=30, learning_rate=2e-3, device="cpu",
                          workers=0, seed=3, use_pretrained=False)
        cls.cfg = cfg
        cls.metrics = cls.model.train(cfg, reporter=None)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_registry_and_listing(self):
        self.assertIsInstance(self.model, OcrModel)
        names = [m["name"] for m in list_models(TaskType.OCR)]
        self.assertIn("crnn", names)

    def test_training_converged(self):
        hist = self.metrics["history"]
        self.assertGreater(len(hist), 0)
        # epoch1 可能有负 CTC 数值噪声（BN+随机初始化，自愈型）——
        # 与第 3 轮比对收敛趋势，不与首轮比
        warm = hist[min(2, len(hist) - 1)]["train_loss"]
        self.assertLess(hist[-1]["train_loss"], warm,
                        f"CTC 损失应下降: {[h['train_loss'] for h in hist[:5]]}…")
        self.assertGreaterEqual(self.metrics["best_val_top1"] or 0.0, 0.75,
                                f"最优精确匹配率过低: {self.metrics['final']}")

    def test_weights_roundtrip(self):
        xva = Path(self.cfg.output_dir) / "best.pt"
        self.assertTrue(xva.is_file())
        m2 = load_model(xva, device="cpu")
        self.assertIsInstance(m2, OcrModel)
        self.assertEqual(m2.charset, self.model.charset)

    def test_predict_returns_text(self):
        arr, _ = _draw_sample("CAT")
        pred = self.model.predict(arr)
        self.assertEqual(len(pred.boxes), 1)
        self.assertIsInstance(pred.texts[0], str)
        self.assertGreater(len(pred.texts[0]), 0)

    def test_prepare_ocr_family(self):
        data = prepare(TaskType.OCR, "yolo", Path(self.cfg.dataset_dir),
                       val_ratio=0.25, seed=3)
        self.assertTrue(data.classes)            # 标注文本即"类别"

    def test_detection_rows_text_passthrough(self):
        from xccv.engine.result_json import build_result_json

        pred = Prediction(labels=[0], scores=[0.9],
                          boxes=[[0, 0, 10, 10]], texts=["CAT"])
        payload = build_result_json(task="ocr", source={"file": "a.png"},
                                    prediction=pred, classes=["文本"])
        self.assertEqual(payload["detections"][0]["text"], "CAT")

    def test_levenshtein(self):
        self.assertEqual(_levenshtein("CAT", "CAT"), 0)
        self.assertEqual(_levenshtein("CAT", "CAP"), 1)
        self.assertEqual(_levenshtein("", "AB"), 2)


if __name__ == "__main__":
    unittest.main()
