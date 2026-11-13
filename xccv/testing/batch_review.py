"""O19+O20（V0.7）：批量结果人工复判（四状态）与一键回流训练集。

对照 VIMO 推理页"复判"：逐图打 正常/漏检/过检/存疑 并统计占比；问题图
可回流到训练集目录（图 + 标注 json）形成"推理→挑错→回流→再训练"闭环。

- 状态持久化：批量输出目录「检出结果/复判.json」（键=图片绝对路径）；
- 建议列：由 GT 匹配统计（batch_match.per_image）自动给出，人工可改；
- 回流规则：有同名 GT json 的图复制原图+原标注（人工修正漏检）；无 GT
  的图复制原图+当前模型预测框（to_labelme 转 labelme 格式）——统一落
  数据集目录的「待标注」子目录，由人工确认后再入训；
- 纯逻辑函数（load_review/save_review/suggest_status/plan_reflow）与
  对话框分离，回归测试不依赖 Qt。
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

from ..engine.result_json import to_labelme

log = logging.getLogger(__name__)

REVIEW_FILENAME = "复判.json"

STATUS_UNJUDGED = "未判"
STATUS_OK = "正常"
STATUS_MISSED = "漏检"
STATUS_FP = "过检"
STATUS_UNCERTAIN = "存疑"
STATUSES = [STATUS_UNJUDGED, STATUS_OK, STATUS_MISSED, STATUS_FP,
            STATUS_UNCERTAIN]


def load_review(path: Path) -> dict:
    """复判.json → {图片路径: {"status": ..., "time": ...}}；缺失/损坏返回 {}。"""
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        log.warning("复判文件读取失败（按空处理）: %s", p, exc_info=True)
        return {}


def save_review(path: Path, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def suggest_status(entry: dict) -> str:
    """per_image 条目 → 建议状态。gt=None（无 GT json）→ 未判。"""
    if entry.get("gt") is None:
        return STATUS_UNJUDGED
    missed = entry.get("missed") or 0
    fp = entry.get("fp") or 0
    if missed and fp:
        return STATUS_MISSED      # 漏检优先（最危险）
    if missed:
        return STATUS_MISSED
    if fp:
        return STATUS_FP
    return STATUS_OK


def review_stats(review: dict, entries: list) -> dict:
    """四状态计数与占比（占比按参与统计的图片总数）。"""
    counts = {s: 0 for s in STATUSES}
    for entry in entries:
        rec = review.get(str(entry.get("path") or "")) or {}
        status = rec.get("status") or suggest_status(entry)
        counts[status if status in counts else STATUS_UNJUDGED] += 1
    total = sum(counts.values())
    out = {}
    for s, n in counts.items():
        out[s] = {"count": n,
                  "rate": round(n / total, 4) if total else 0.0}
    out["_total"] = total
    return out


def plan_reflow(entry: dict, result: dict, dataset_dir: Path,
                subdir: str = "待标注") -> dict:
    """计算一张图的回流动作（纯函数，返回说明；实际复制由对话框执行）。

    返回 {"image": 源图, "json": 源标注(None=用预测), "dst_image",
    "dst_json", "used_prediction": bool}；无法回流（图缺失）返回 {}。
    """
    src = Path(str(entry.get("path") or ""))
    if not src.is_file():
        return {}
    dst_dir = Path(dataset_dir) / subdir
    stem = src.stem
    # 同名冲突：追加时间戳后缀（同图重复回流也可区分）
    dst_image = dst_dir / src.name
    if dst_image.exists():
        dst_image = dst_dir / f"{stem}_{datetime.now().strftime('%H%M%S')}{src.suffix}"
    gt_json = src.with_suffix(".json")
    if gt_json.exists():
        dst_json = dst_dir / (dst_image.stem + ".json")
        return {"image": src, "json": gt_json, "dst_image": dst_image,
                "dst_json": dst_json, "used_prediction": False}
    dst_json = dst_dir / (dst_image.stem + ".json")
    return {"image": src, "json": None, "dst_image": dst_image,
            "dst_json": dst_json, "used_prediction": True,
            "payload": result}


def execute_reflow(actions: list[dict]) -> tuple[int, list[str]]:
    """执行 plan_reflow 产出的动作列表 → (成功数, 错误文案列表)。

    - 图与 GT 标注：原样复制（shutil.copy2，字节保真）；
    - 无 GT：当前预测框经 to_labelme 转 labelme json 落「待标注」。
    """
    copied, errors = 0, []
    for a in actions:
        try:
            a["dst_image"].parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(a["image"], a["dst_image"])
            if a.get("json") is not None:
                shutil.copy2(a["json"], a["dst_json"])
            else:
                payload = a.get("payload") or {}
                src = a["image"]
                payload = {"source": {
                    "file": src.name, "path": str(src),
                    "width": payload.get("width") or 0,
                    "height": payload.get("height") or 0},
                    "detections": payload.get("detections") or []}
                a["dst_json"].write_text(
                    json.dumps(to_labelme(payload), ensure_ascii=False,
                               indent=1),
                    encoding="utf-8")
            copied += 1
            log.info("回流: %s → %s（%s）", a["image"], a["dst_image"],
                     "原标注" if a.get("json") is not None else "预测框")
        except Exception as e:  # noqa: BLE001  单张失败不阻塞其余回流
            errors.append(f"{Path(str(a.get('image'))).name}: {e}")
            log.warning("回流失败: %s", a.get("image"), exc_info=True)
    return copied, errors
