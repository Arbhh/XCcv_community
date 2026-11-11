"""分类指标：Top-K 准确率与混淆矩阵。"""

import numpy as np


def topk_accuracy(probs: np.ndarray, targets: np.ndarray, k: int = 1) -> float:
    """probs: [N, C] 概率或 logits；targets: [N]。返回 Top-K 命中率。"""
    if len(targets) == 0:
        return 0.0
    k = min(k, probs.shape[1])
    topk = np.argsort(probs, axis=1)[:, -k:]
    return float(np.any(topk == np.asarray(targets)[:, None], axis=1).mean())


def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    """混淆矩阵 m[真实类别, 预测类别]。"""
    m = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            m[t, p] += 1
    return m
