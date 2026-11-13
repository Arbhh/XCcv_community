"""O3（V0.7）：Loss 曲线自动判读（VIMO"指标差如何调优"第 5 步对照）。

训练结束/历史详情自动生成一句话判读，纯规则实现、零训练改动：
- NaN/Inf → 直接给 FAQ 文案（VIMO"Loss became infinite"处置口径）；
- 末段训练损失仍在明显下降 → 疑似欠拟合，建议增加轮次；
- 验证指标较最佳回落明显（且训练损失继续降）→ 疑似过拟合，
  建议采用 best 轮次权重或减少轮次；
- 其余 → 收敛正常。

history 条目与各家族落盘一致（epoch/train_loss + 验证指标键之一）。
"""

from __future__ import annotations

import math

_METRIC_KEYS = ("map_loose", "map50", "val_top1", "auroc", "exact_match")
_MIN_EPOCHS = 6          # 少于该轮数不下结论
_TAIL_RATIO = 0.3        # 末段占比
_TAIL_SLOPE = -0.05      # 末段线性斜率 / 均值 < 该值 → 仍在明显下降
_OVERFIT_DROP = 0.10     # 验证指标较最佳回落超 10% → 疑似过拟合


def _finite(values) -> list:
    return [float(v) for v in values
            if isinstance(v, (int, float)) and math.isfinite(v)]


def _slope(values: list) -> float:
    """最小二乘斜率（按样本序归一：每轮变化量）。"""
    n = len(values)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(values) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, values)) / denom


def diagnose(history: list) -> dict:
    """逐轮 history → {"kind": ..., "text": ...}。

    kind ∈ nan | underfit | overfit | ok | insufficient | empty。
    """
    history = [h for h in (history or []) if isinstance(h, dict)]
    if not history:
        return {"kind": "empty", "text": ""}
    losses = [h.get("train_loss") for h in history]
    if any(isinstance(v, (int, float)) and not math.isfinite(v)
           for v in losses):
        return {"kind": "nan",
                "text": "训练损失出现 NaN/Inf——数值不稳定：建议降低学习率、"
                        "检查标注数据（空/损坏样本），或在开发者选项开启"
                        "「确定性训练」复现问题"}
    if len(history) < _MIN_EPOCHS:
        return {"kind": "insufficient",
                "text": f"轮次不足 {_MIN_EPOCHS} 轮，损失形态判读意义有限"}
    fin_losses = _finite(losses)
    metric = _metric_series(history)
    metric_vals = [v for _, v in metric if math.isfinite(v)] if metric else []
    # 过拟合先判（val 回落 + train 继续降是过拟合特征；欠拟合是"只剩
    # loss 在降"）——顺序错了会把过拟合误报成欠拟合
    if metric_vals and len(metric_vals) >= 3:
        best = max(metric_vals)
        last = metric_vals[-1]
        best_epoch = next(e for e, v in metric if math.isfinite(v) and v == best)
        if best > 0 and last < best * (1 - _OVERFIT_DROP):
            return {"kind": "overfit",
                    "text": f"疑似过拟合：验证指标较最佳回落 "
                            f"{(best - last) / best:.0%}（最佳在第 "
                            f"{best_epoch} 轮，当前第 {metric[-1][0]} 轮），"
                            "建议采用 best 轮次权重（默认已选）或减少轮次"}
    if len(fin_losses) >= _MIN_EPOCHS:
        tail_n = max(3, int(len(fin_losses) * _TAIL_RATIO))
        tail = fin_losses[-tail_n:]
        mean = sum(tail) / len(tail)
        if mean > 0 and _slope(tail) / mean < _TAIL_SLOPE:
            return {"kind": "underfit",
                    "text": f"疑似欠拟合：训练损失末段仍在明显下降"
                            f"（近 {tail_n} 轮斜率为负），建议增加训练轮次"}
    return {"kind": "ok", "text": "训练收敛正常（损失与验证指标形态无异常）"}


def _metric_series(history: list) -> list:
    """第一个可用的验证指标序列 [(epoch, value)]（家族自适应）。"""
    for key in _METRIC_KEYS:
        series = [(h.get("epoch", i + 1), h.get(key))
                  for i, h in enumerate(history)
                  if isinstance(h.get(key), (int, float))]
        if series:
            return series
    return []
