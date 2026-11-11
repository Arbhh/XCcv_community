# -*- coding: utf-8 -*-
"""AI 批量预标注（V0.6 T5-4）：对文件夹逐图跑检测/分割模型 → labelme JSON。

- 模型走 engine.models.registry.load_model 唯一入口（.pt；GPU 可用自动用）；
- 形状映射：检出框→rectangle；分割掩码→多边形（cv2.findContours + approxPolyDP，
  eps 控制简化程度）；每图一个 JSON（不内嵌 imageData，与 T5-1 瘦身一致）；
- 跳过已有同名 JSON 的图（可关）；进度/取消由调用方（窗口 QThread）驱动；
- 输出可被 labelme.label_file.load 直接打开（旧 JSON 兼容加载已验）。
"""

import json
from pathlib import Path

import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def list_images(folder: str) -> list:
    p = Path(folder)
    if not p.is_dir():
        raise ValueError(f"文件夹不存在：{folder}")
    return sorted(q for q in p.iterdir()
                  if q.is_file() and q.suffix.lower() in IMG_EXTS)


def mask_to_polygons(mask: np.ndarray, eps_ratio: float = 0.002) -> list:
    """bool 掩码 → 多边形顶点列表（[[x,y],...]；取最大轮廓 + 简化）。"""
    import cv2

    m = np.ascontiguousarray(mask.astype(np.uint8))
    contours, _hier = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    c = max(contours, key=cv2.contourArea)
    eps = max(1.0, eps_ratio * cv2.arcLength(c, True))
    poly = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    if len(poly) < 3:
        x, y, w, h = cv2.boundingRect(c)
        poly = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    return [[float(x), float(y)] for x, y in poly]


def prediction_to_shapes(pred, classes: list, eps_ratio: float = 0.002) -> list:
    """Prediction（boxes/masks/labels/scores）→ labelme shapes 列表。"""
    shapes = []
    labels = [] if pred.labels is None else list(pred.labels)
    scores = [] if pred.scores is None else list(pred.scores)
    masks = getattr(pred, "masks", None)
    masks = [] if masks is None else list(masks)   # P150：不用真值判空
    for i, box in enumerate([] if pred.boxes is None else pred.boxes):
        x1, y1, x2, y2 = (float(v) for v in box)
        lid = int(labels[i]) if i < len(labels) else 0
        label = classes[lid] if 0 <= lid < len(classes) else str(lid)
        m = masks[i] if i < len(masks) else None
        if m is not None and np.asarray(m).size and np.asarray(m).any():
            pts = mask_to_polygons(np.asarray(m).astype(bool), eps_ratio)
            if len(pts) >= 3:
                shapes.append({"label": label, "points": pts, "group_id": None,
                               "shape_type": "polygon",
                               "flags": {}, "description": "",
                               "score": round(float(scores[i]), 4)
                               if i < len(scores) else None})
                continue
        shapes.append({"label": label,
                       "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                       "group_id": None, "shape_type": "rectangle",
                       "flags": {}, "description": "",
                       "score": round(float(scores[i]), 4)
                       if i < len(scores) else None})
    return shapes


def write_labelme_json(path: Path, image_path: Path, shapes: list) -> None:
    """labelme JSON（不内嵌 imageData；label_file.load 按 imagePath 回读）。"""
    from PIL import Image

    with Image.open(image_path) as im:
        w, h = im.size
    data = {"version": "5.8.1", "flags": {}, "shapes": shapes,
            "imagePath": image_path.name, "imageData": None,
            "imageHeight": int(h), "imageWidth": int(w)}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def annotate_folder(model_path: str, folder: str, conf: float = 0.25,
                    iou: float = 0.45, device: str = "auto",
                    skip_annotated: bool = True, eps_ratio: float = 0.002,
                    progress_cb=None, cancel=None) -> dict:
    """批量预标注主流程（窗口线程调用）。progress_cb(done, total, name)；
    cancel: callable() 返回 True 时中止。返回摘要 dict。"""
    images = list_images(folder)
    total = len(images)
    if total == 0:
        raise ValueError("文件夹里没有可标注的图片")
    from ..engine.models.registry import load_model

    model = load_model(model_path, device=device)
    classes = list(getattr(model, "classes", None) or [])
    task = getattr(getattr(model, "task", None), "value", "")
    if task not in ("detection", "segmentation"):
        raise ValueError(f"批量预标注需要检测/分割模型，所选模型任务为「{task}」")
    done = skipped = annotated = 0
    per_image: list = []
    for img_path in images:
        if cancel is not None and cancel():
            break
        out_json = img_path.with_suffix(".json")
        if skip_annotated and out_json.is_file():
            skipped += 1
            done += 1
            if progress_cb is not None:
                progress_cb(done, total, img_path.name)
            continue
        try:
            from ..engine.io_utils import imread_unicode

            mat = imread_unicode(str(img_path))
            if mat is None:
                raise ValueError("无法读取图片")
            pred = model.predict(mat, conf=float(conf), iou=float(iou))
            shapes = prediction_to_shapes(pred, classes, eps_ratio)
            write_labelme_json(out_json, img_path, shapes)
            annotated += 1
            per_image.append({"image": img_path.name, "shapes": len(shapes),
                              "error": ""})
        except Exception as e:  # noqa: BLE001  单图失败不拖垮整批
            per_image.append({"image": img_path.name, "shapes": 0,
                              "error": str(e)})
        done += 1
        if progress_cb is not None:
            progress_cb(done, total, img_path.name)
    return {"total": total, "annotated": annotated, "skipped": skipped,
            "cancelled": bool(cancel is not None and cancel()),
            "failed": sum(1 for r in per_image if r["error"]),
            "images": per_image}
