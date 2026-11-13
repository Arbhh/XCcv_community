"""A2（V0.7）：轻量 OCR 字符识别（CRNN + CTC，自增优化，用户指定优先）。

对照 VIMO"字符识别"的首期落地：标注平台 rectangle 标注、label=字符串
内容（标注工具零改动）→ 行裁剪训练 → 识别文本输出。

- 模型：CRNN（CNN 特征 → BiLSTM → CTC logits），纯 torch；
- 输入：行图统一缩到 高 32 × 宽 160（工业编码字号相近，等比差异通过
  拉伸容忍；多行/环形字符不在本期）；
- 字符集：从全部标注文本聚合（index 0 = CTC blank）；
- 推理：predict(image) 把整图当**单行文本**处理——定位由检测模型/流程
  ROI 完成（级联：检测框 → 裁图 → OCR），统一 JSON detections 增 text 键；
- 指标：逐行精确匹配率（exact_match）/ 字符错误率（CER）/ 字符准确率；
- TRT 导出：不做（RNN 算子限制），.pt 交付，文档注明。
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from ..dataset.base import TaskType
from ..io_utils import imread_unicode
from .base import BaseModel, Prediction, TrainConfig, TrainReporter

log = logging.getLogger(__name__)

_LINE_H = 32          # 行图统一高度
_LINE_W = 160         # 行图统一宽度（CTC 定长输入）
_HIDDEN = 128         # BiLSTM 隐层（双向拼接后 256）


def _crop_line(img_bgr: np.ndarray, points: list, pad: int = 2) -> np.ndarray:
    """shape 顶点 → 行图（外接框 + pad，缩放到 定长）。"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    h, w = img_bgr.shape[:2]
    x1 = max(0, int(min(xs)) - pad)
    y1 = max(0, int(min(ys)) - pad)
    x2 = min(w, int(max(xs)) + pad)
    y2 = min(h, int(max(ys)) + pad)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return np.zeros((_LINE_H, _LINE_W, 3), np.uint8)
    crop = img_bgr[y1:y2, x1:x2]
    return cv2.resize(crop, (_LINE_W, _LINE_H), interpolation=cv2.INTER_AREA)


def _to_tensor(line_bgr: np.ndarray) -> torch.Tensor:
    x = line_bgr.astype(np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return torch.from_numpy(x.astype(np.float32)).permute(2, 0, 1)


class CRNN(nn.Module):
    """CNN 特征（高度池化到 1）→ BiLSTM → CTC 字符 logits。"""

    def __init__(self, n_classes: int):
        super().__init__()

        def block(cin, cout, pool=True):
            layers = [nn.Conv2d(cin, cout, 3, padding=1),
                      nn.ReLU(inplace=True),
                      nn.BatchNorm2d(cout)]
            if pool:
                layers.append(nn.MaxPool2d(2, 2))
            return layers

        self.cnn = nn.Sequential(
            *block(3, 64, pool=False),
            nn.MaxPool2d((2, 2), (2, 2)),          # H 32→16, W /2
            *block(64, 128),
            nn.MaxPool2d((2, 2), (2, 2)),          # H 16→8, W /2
            *block(128, 256),
            *block(256, 256, pool=False),
            nn.MaxPool2d((2, 1), (2, 1)),          # H 8→4, W 不变
            *block(256, 256, pool=False),
            nn.AdaptiveAvgPool2d((1, None)),       # 高度压到 1
        )
        self.rnn = nn.LSTM(256, _HIDDEN, num_layers=1, bidirectional=True,
                           batch_first=True)
        self.fc = nn.Linear(_HIDDEN * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N,3,32,W) → (W', N, n_classes)（CTC 的 T×N×C 约定）。"""
        f = self.cnn(x)                            # (N,256,1,W')
        f = f.squeeze(2).permute(2, 0, 1)          # (W',N,256)
        r, _ = self.rnn(f)
        return self.fc(r)                          # (W',N,C)


def _ctc_decode(logits: torch.Tensor) -> tuple[str, float]:
    """greedy CTC 解码：(T,1,C) → (text, 平均字符置信度)。"""
    probs = logits.softmax(-1)                     # (T,1,C)
    ids = probs.argmax(-1).squeeze(1).tolist()     # T
    conf = probs.max(-1).values.squeeze(1).tolist()
    out_ids: list = []
    out_conf: list = []
    prev = -1
    for t, i in enumerate(ids):
        if i != prev and i != 0:                   # 0=blank；折叠重复
            out_ids.append(i)
            out_conf.append(conf[t])
        prev = i
    return out_ids, out_conf


class OcrModel(BaseModel):
    """CRNN+CTC 行文本识别（family=yolo：labelme 扫描，label=文本内容）。"""

    name = "crnn"
    family = "yolo"
    supported_tasks = (TaskType.OCR,)
    default_image_size = 0   # 行图定长，与整图训练输入无关

    def __init__(self, num_classes: int = 1, device: str = "auto",
                 use_pretrained: bool = True):
        super().__init__(num_classes, device)
        self.charset: list[str] = []               # index 0 保留给 blank
        self.net: CRNN | None = None
        self._text_class = ["文本"]

    # ---- 字符集与编解码 ----
    def _build_charset(self, texts: list) -> None:
        chars: set = set()
        for t in texts:
            chars.update(str(t))
        self.charset = sorted(chars)
        if not self.charset:
            raise ValueError("标注文本为空，无法构建字符集"
                             "（OCR 需要 rectangle 标注、label=字符串内容）")
        self.net = CRNN(len(self.charset) + 1).to(self.device)
        self._ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    def _encode(self, text: str) -> list:
        idx = {c: i + 1 for i, c in enumerate(self.charset)}
        ids = [idx[c] for c in str(text) if c in idx]
        return ids or [1]                          # 全未知字符：给 1 占位避免 0 长目标

    def _decode_ids(self, ids: list) -> str:
        return "".join(self.charset[i - 1] for i in ids if 1 <= i <= len(self.charset))

    # ---- 训练 ----
    def fit(self, data, cfg: TrainConfig, reporter: TrainReporter | None = None) -> dict:
        reporter = reporter or TrainReporter()
        torch.manual_seed(cfg.seed)
        texts = [sh.label for s in data.train_samples for sh in s.shapes]
        val_texts = [(sh.label, s.image_path, sh.points)
                     for s in data.val_samples for sh in s.shapes]
        self._build_charset(texts)
        log.info("OCR 字符集: %d 字符（%s）", len(self.charset),
                 "".join(self.charset[:30]))
        # 训练行样本（图路径, 顶点, 目标 ids）
        train_lines = [(s.image_path, sh.points, self._encode(sh.label))
                       for s in data.train_samples for sh in s.shapes]
        if not train_lines:
            raise ValueError("OCR 训练需要 rectangle 标注（label=文本内容）")
        opt = torch.optim.AdamW(self.net.parameters(),
                                lr=cfg.learning_rate or 1e-3,
                                weight_decay=cfg.weight_decay)
        # 余弦退火：CTC 后期高学习率易发散（探针实测 ep35 loss 反弹 1.8）
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, cfg.epochs))
        history = []
        best_exact = -1.0
        rng = np.random.default_rng(cfg.seed)
        started = datetime.now()
        cache: dict = {}

        def _line_img(path, points):
            key = (str(path), tuple(tuple(p) for p in points))
            if key not in cache:
                img = imread_unicode(path)
                cache.clear() if len(cache) > 64 else None
                cache[key] = (img if img is not None
                              else np.zeros((_LINE_H, _LINE_W, 3), np.uint8))
            img = cache[key]
            return _crop_line(img, list(points))

        n = len(train_lines)
        for epoch in range(1, max(1, cfg.epochs) + 1):
            order = rng.permutation(n)
            total_loss, n_batches = 0.0, 0
            for bi, i in enumerate(order, start=1):
                path, points, target = train_lines[int(i)]
                x = _to_tensor(_line_img(path, points)).unsqueeze(0).to(self.device)
                logits = self.net(x)                     # (T,1,C)
                t_len = len(target)
                in_len = logits.shape[0]
                if t_len > in_len:                       # 行太短放不下目标：截断目标
                    target = target[:in_len]
                    t_len = in_len
                t = torch.tensor([target], dtype=torch.long)             # (1,L)
                loss = self._ctc(logits, t, torch.full((1,), in_len, dtype=torch.long),
                                 torch.full((1,), t_len, dtype=torch.long))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += float(loss)
                n_batches += 1
                if reporter is not None:
                    reporter.on_train_progress(epoch, bi, n)
            exact, cer, char_acc = self._eval_lines(
                [(p, pt, self._encode(t)) for t, p, pt in val_texts]
                or [(p, pt, tg) for p, pt, tg in train_lines[:min(32, n)]])
            entry = {"epoch": epoch, "train_loss": round(total_loss / max(1, n_batches), 4),
                     "val_top1": round(exact, 4), "exact_match": round(exact, 4)}
            history.append(entry)
            log.info("OCR 轮次 %d/%d: loss=%.4f exact=%.3f cer=%.3f",
                     epoch, cfg.epochs, entry["train_loss"], exact, cer)
            if reporter is not None:
                reporter.on_epoch(epoch, entry)
            if exact > best_exact:
                best_exact = exact
                self._best_state = {k: v.detach().cpu().clone()
                                    for k, v in self.net.state_dict().items()}
            sched.step()
            if reporter is not None and reporter.stop_requested:
                break
        if getattr(self, "_best_state", None) is not None:
            self.net.load_state_dict(self._best_state)

        # 收尾：最优权重评估值 + 权重落盘（平台自有明文 .pt 容器）
        best_pt = Path(cfg.output_dir) / "best.pt"
        torch.save(self._ckpt(), best_pt)
        metrics = {
            "final_val_top1": round(best_exact, 4) if best_exact >= 0 else None,
            "best_val_top1": round(best_exact, 4) if best_exact >= 0 else None,
            "final": {"exact_match": round(exact, 4), "cer": round(cer, 4),
                      "char_acc": round(char_acc, 4)},
            "charset": "".join(self.charset),
            "classes": self._text_class,
            "history": history,
            "start_time": started.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "completed",
        }
        reporter.on_finish(True)
        return metrics

    def _eval_lines(self, lines: list) -> tuple:
        """(精确匹配率, CER, 字符准确率)——val 行优先，val 空回退训练行。"""
        n_ok, dist_sum, n_chars, n_ref = 0, 0, 0, 0
        for path, points, target in lines:
            gt = self._decode_ids(target)
            img = imread_unicode(path)           # ndarray 判空禁真值（P150）
            if img is None:
                img = np.zeros((_LINE_H, _LINE_W, 3), np.uint8)
            x = _to_tensor(_crop_line(img, list(points))).unsqueeze(0).to(self.device)
            with torch.no_grad():
                ids, _ = _ctc_decode(self.net(x))
            pred = self._decode_ids(ids)
            if pred == gt:
                n_ok += 1
            dist_sum += _levenshtein(pred, gt)
            n_chars += len(gt)
        total = max(1, len(lines))
        return (n_ok / total,
                dist_sum / max(1, n_chars),
                1.0 - dist_sum / max(1, n_chars))

    def _ckpt(self) -> dict:
        return {"format": "xccv-ocr", "charset": self.charset,
                "classes": self._text_class,
                "state": self.net.state_dict()}

    # ---- 推理 ----
    def predict(self, image_bgr: np.ndarray, conf: float = 0.25,
                iou: float = 0.45) -> Prediction:
        """整图当单行文本识别（定位交给检测/流程 ROI 级联）。"""
        if self.net is None:
            raise RuntimeError("OCR 模型未加载")
        x = _to_tensor(cv2.resize(image_bgr, (_LINE_W, _LINE_H),
                                  interpolation=cv2.INTER_AREA)
                       ).unsqueeze(0).to(self.device)
        with torch.no_grad():
            ids, confs = _ctc_decode(self.net(x))
        text = self._decode_ids(ids)
        score = float(np.mean(confs)) if confs else 0.0
        h, w = image_bgr.shape[:2]
        return Prediction(labels=[0], scores=[round(score, 4)],
                          boxes=[[0.0, 0.0, float(w), float(h)]],
                          texts=[text])

    # ---- 权重 ----
    def load_weights(self, source) -> None:
        ckpt = source if isinstance(source, dict) else torch.load(
            source, map_location="cpu", weights_only=False)
        if ckpt.get("format") != "xccv-ocr":
            raise ValueError("不是 xccv-ocr 权重文件")
        self.charset = list(ckpt.get("charset") or [])
        if not self.charset:
            raise ValueError("权重缺少字符集")
        self.net = CRNN(len(self.charset) + 1).to(self.device)
        self.net.load_state_dict(ckpt["state"])
        self.net.eval()
        self._ctc = nn.CTCLoss(blank=0, zero_infinity=True)


def _levenshtein(a: str, b: str) -> int:
    """编辑距离（CER 用）。"""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]
