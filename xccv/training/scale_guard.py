"""O4（V0.7）：标注实例缩放后像素尺寸校验（VIMO"缩放后缺陷应 ≥8×8"对照）。

训练输入会把原图缩放到网络尺寸：缩放后缺陷过小（短边 <8px，极限 4×4
有风险）时模型基本学不到。本模块扫描全部标注实例的最小外接框短边，
按当前训练输入口径换算"缩放后尺寸"，统计过小实例占比并给一键建议。

口径（与两家族实际几何一致）：
- stretch（R-CNN 家族 + 平台增强 resize）：非等比拉伸到 宽×高；
- letterbox（YOLO 家族）：长边缩放到 imgsz 保持长宽比（官方口径，P209）；
- OCR/异常检测：无实例几何语义，跳过。
"""

from __future__ import annotations

import numpy as np

MIN_PX = 8          # 缩放后短边下限（VIMO：极限 4×4 有风险）
WARN_RATIO = 0.05   # 过小实例占比超 5% → 警告


def instance_short_sides(samples: list) -> list:
    """每实例最小外接框短边（原图像素）。"""
    sides = []
    for s in samples:
        for sh in s.shapes:
            x1, y1, x2, y2 = sh.bbox()
            sides.append(min(max(0.0, x2 - x1), max(0.0, y2 - y1)))
    return sides


def scaled_short_sides(samples: list, mode: str, size_w: int,
                       size_h: int) -> list:
    """按训练输入口径换算后的实例短边列表。

    mode="stretch"：sx=size_w/imgW、sy=size_h/imgH（非等比）；
    mode="letterbox"：s=imgsz/max(imgW,imgH)（等比，imgsz=长边目标）。
    """
    out = []
    for s in samples:
        if not s.shapes or s.width <= 0 or s.height <= 0:
            continue
        if mode == "letterbox":
            imgsz = max(size_w, size_h)
            k = imgsz / max(s.width, s.height)
            sx = sy = k
        else:
            sx = (size_w or s.width) / s.width
            sy = (size_h or s.height) / s.height
        for sh in s.shapes:
            x1, y1, x2, y2 = sh.bbox()
            out.append(min(max(0.0, (x2 - x1)) * sx, max(0.0, (y2 - y1)) * sy))
    return out


def scale_issue(samples: list, mode: str, size_w: int, size_h: int,
                min_px: float = MIN_PX,
                warn_ratio: float = WARN_RATIO) -> dict | None:
    """返回统计或 None（无实例/无可判数据）。

    {"n": 实例数, "n_small": 过小数, "ratio": 占比, "p10": 短边 P10,
     "min_px": 下限, "median": 中位数}
    """
    sides = scaled_short_sides(samples, mode, size_w, size_h)
    if not sides:
        return None
    arr = np.asarray(sides, dtype=float)
    n_small = int((arr < min_px).sum())
    return {"n": int(arr.size), "n_small": n_small,
            "ratio": round(n_small / arr.size, 4),
            "p10": round(float(np.percentile(arr, 10)), 1),
            "median": round(float(np.median(arr)), 1),
            "min_px": min_px}


def scale_issue_text(stat: dict) -> tuple | None:
    """stat → (详情, 建议) 二元组（超阈值才返回）。"""
    if stat is None or stat["ratio"] <= WARN_RATIO:
        return None
    detail = (f"{stat['n_small']}/{stat['n']} 个标注实例缩放后短边 "
              f"<{stat['min_px']}px（占 {stat['ratio']:.0%}；短边 P10="
              f"{stat['p10']}px，中位数 {stat['median']}px）")
    suggestion = ("缩放过小的缺陷模型基本学不到（经验：缩放后应 ≥8×8，极限 4×4 "
                  "有风险）。两个方向：①增大训练输入尺寸（增强策略「调整图像"
                  "大小」/训练输入宽高）；②启用切图训练（tile 640 + 推理侧 "
                  "4K 分块推理配对）。")
    return detail, suggestion
