"""Defect Augment Studio 后端 — FastAPI。

接口一览：
  GET  /                     前端页面（web/ 静态托管）
  GET  /api/health           健康检查
  GET  /api/sam_status       SAM 引擎可用性（GrabCut 恒可用）
  POST /api/extract          抠图：GrabCut（默认）/ FastSAM（吞吐）/ MobileSAM（精度）
  POST /api/poisson_preview  编辑器内 Poisson 融合预览（代理分辨率）
  POST /api/composite        单张全分辨率合成导出（PNG/JPEG/TIFF）
  POST /api/batch/start      背景池批量合成作业（服务器目录或上传文件）
  GET  /api/batch/status/{jid}  批量进度与摘要报告
  POST /api/batch/cancel/{jid}   取消批量作业
"""

from __future__ import annotations

import base64
import glob
import importlib.util
import json
import math
import mimetypes
import os
import threading
import time
import traceback
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi import Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .composite import (b64_to_img, encode_image, img_to_b64, jitter_instances,
                        materialize_instances, render_scene, to_bgra)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# MIME 兜底：Windows 注册表可能把 .js 的 Content Type 劫持为 text/plain
#（Python mimetypes 读 HKCR\.js），Chromium 严格校验会拒载 module script，
# 表现为"界面正常但所有按钮无反应"（前端 JS 完全没执行）。显式注册覆盖之。
for _ext, _mt in ((".js", "text/javascript"), (".mjs", "text/javascript"),
                  (".css", "text/css"), (".wasm", "application/wasm"),
                  (".json", "application/json"), (".svg", "image/svg+xml")):
    mimetypes.add_type(_mt, _ext)
# XCcv：权重与主项目共用 weights/sam（标注平台同目录）
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(ROOT)),
                           "weights", "sam")
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

app = FastAPI(title="Defect Augment Studio", version="1.1")


@app.middleware("http")
async def no_cache_ui(request, call_next):
    """HTML/JS/CSS 强制协商缓存（no-cache）：改版后浏览器必须回源验证，
    杜绝"旧 index.html + 新 js"的版本错配（曾导致 boot 找不到新按钮直接中断）。
    静态资源带 ETag，未变化时回 304，无带宽代价。"""
    resp = await call_next(request)
    p = request.url.path
    if p == "/" or p.endswith((".html", ".js", ".mjs", ".css")):
        resp.headers["Cache-Control"] = "no-cache, must-revalidate"
    return resp

_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()


# ---------------------------------------------------------------- SAM 三档引擎

def _engine_available(kind: str) -> bool:
    if kind == "fastsam":
        if importlib.util.find_spec("ultralytics") is None:
            return False
        return os.path.exists(os.path.join(WEIGHTS_DIR, "FastSAM-s.pt"))
    if kind == "mobilesam":
        if importlib.util.find_spec("ultralytics") is None:
            return False
        return os.path.exists(os.path.join(WEIGHTS_DIR, "mobile_sam.pt"))
    return True


_SAM_CACHE: dict = {}


def _segment_with_sam(kind: str, img_bgr: np.ndarray, box_xyxy) -> np.ndarray:
    """返回 0/255 uint8 掩码。box prompt 来自用户选区外接框。"""
    if kind == "fastsam":
        from ultralytics import FASTSAM
        model_path = os.path.join(WEIGHTS_DIR, "FastSAM-s.pt")
        model = _SAM_CACHE.get(model_path)
        if model is None:
            model = FASTSAM(model_path)
            _SAM_CACHE[model_path] = model
        results = model(img_bgr, device="cpu", retina_masks=True,
                        bboxes=[float(v) for v in box_xyxy], verbose=False)
        masks = results[0].masks
        if masks is None or len(masks.data) == 0:
            raise RuntimeError("FastSAM 未在选区内产生掩码")
        m = masks.data.cpu().numpy().max(axis=0)
        return (m > 0.5).astype(np.uint8) * 255
    if kind == "mobilesam":
        # XCcv：环境无 mobile_sam 包，经 ultralytics 加载同一权重
        from ultralytics import SAM
        ckpt = os.path.join(WEIGHTS_DIR, "mobile_sam.pt")
        model = _SAM_CACHE.get("mobilesam")
        if model is None:
            model = _SAM_CACHE["mobilesam"] = SAM(ckpt)
        results = model(img_bgr, bboxes=[[float(v) for v in box_xyxy]],
                        verbose=False)
        masks = results[0].masks
        if masks is None or len(masks.data) == 0:
            raise RuntimeError("MobileSAM 未在选区内产生掩码")
        m = masks.data.cpu().numpy().max(axis=0)
        return (m > 0.5).astype(np.uint8) * 255
    raise ValueError("未知引擎: " + kind)


# ---------------------------------------------------------------- 抠图

def _shape_mask(shape: str, geometry, w: int, h: int) -> np.ndarray:
    """用户选区（多边形/矩形/椭圆，图像像素坐标）→ PR_FGD 区域。"""
    m = np.zeros((h, w), np.uint8)
    pts = np.asarray(json.loads(geometry), np.float32)
    if shape in ("polygon", "poly"):
        cv2.fillPoly(m, [pts.astype(np.int32)], 1)
    elif shape == "rect":
        x, y, x2, y2 = [int(v) for v in pts.flatten()[:4]]
        cv2.rectangle(m, (min(x, x2), min(y, y2)), (max(x, x2), max(y, y2)), 1, -1)
    elif shape == "ellipse":
        x, y, x2, y2 = [float(v) for v in pts.flatten()[:4]]
        cv2.ellipse(m, (int((x + x2) / 2), int((y + y2) / 2)),
                    (int(abs(x2 - x) / 2), int(abs(y2 - y) / 2)), 0, 0, 360, 1, -1)
    else:
        raise ValueError("未知选区类型: " + shape)
    return m


def _grabcut(img_bgr: np.ndarray, sel: np.ndarray, iters: int,
             shrink: int, feather: float, notes: list) -> np.ndarray:
    """GrabCut：选区内部 PR_FGD（核心 FGD），外扩环带 PR_BGD，更外侧 BGD。"""
    h, w = sel.shape[:2]
    mask = np.full((h, w), cv2.GC_BGD, np.uint8)
    ring = cv2.dilate(sel, np.ones((15, 15), np.uint8)) > 0
    mask[ring & (sel == 0)] = cv2.GC_PR_BGD
    mask[sel > 0] = cv2.GC_PR_FGD
    core = cv2.erode(sel, np.ones((9, 9), np.uint8)) > 0
    if int(core.sum()) >= 50:
        mask[core] = cv2.GC_FGD
    if int((sel > 0).sum()) < 30:
        notes.append("选区过小（<30px），请圈更大的范围")
        return np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img_bgr, mask, None, bgd, fgd, max(1, min(8, iters)),
                    cv2.GC_INIT_WITH_MASK)
    except cv2.error as e:
        notes.append("GrabCut 失败(%s)" % str(e).splitlines()[0][:80])
        return np.zeros((h, w), np.uint8)
    out = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8) * 255

    # GrabCut 退化检测：结果≈整个选区（前景/背景颜色模型接近时发生）
    sel_area = int((sel > 0).sum())
    out_area = int((out > 0).sum())
    if sel_area and out_area >= 0.95 * sel_area:
        selm = sel > 0
        # 兜底1：彩色鲁棒Z离群点（逐通道中位数+MAD 距离）。对拉丝纹理噪声免疫，
        # 适合细高对比缺陷（划痕等）；呈细长状（PCA 伸长比≥5）时才采纳。
        f = img_bgr.astype(np.float32)
        med = np.median(f[selm], axis=0)
        mad = np.median(np.abs(f[selm] - med), axis=0) * 1.4826 + 1e-6
        dist = np.sqrt((((f - med) / mad) ** 2).sum(-1))
        zmap = ((dist > 3.5) & selm).astype(np.uint8) * 255
        zmap = cv2.morphologyEx(zmap, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, labels, stats, cents = cv2.connectedComponentsWithStats(zmap, 8)
        if n > 1:
            big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            pts_ij = np.column_stack(np.nonzero(labels == big)).astype(np.float32)  # (y,x)
            mu = pts_ij - pts_ij.mean(0)
            cov = (mu.T @ mu) / len(mu)
            eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
            elong = float(np.sqrt(max(eig[0], 1e-6) / max(eig[1], 1e-6)))
            frac = float((zmap > 0).sum()) / sel_area
            if 30 < (zmap > 0).sum() < 0.4 * sel_area and elong >= 5.0:
                cand = cv2.dilate(zmap, np.ones((3, 3), np.uint8))
                cand = _clean_mask(cand, img_bgr, shrink, feather)
                notes.append("GrabCut 未分离，已按细线缺陷用彩色离群点分割"
                             "（松散选区建议贴合缺陷，或部署 FastSAM/MobileSAM）")
                return cand
        # 兜底2：大缺陷（凹坑/锈斑等填满选区的形态）→ 腐蚀核心，配合笔刷修正
        k = max(9, (int(np.sqrt(sel_area)) // 6) | 1)
        core = cv2.erode(sel, np.ones((k, k), np.uint8))
        if int((core > 0).sum()) >= 50:
            notes.append("GrabCut 未分离，已回退为选区腐蚀核心；建议用笔刷修正边缘或改用 SAM 引擎")
            return _clean_mask(core, img_bgr, shrink, feather)
        # 兜底3：保留选区
        notes.append("GrabCut 未分离，已保留选区作为掩码；请用笔刷修正")
    return _clean_mask(out, img_bgr, shrink, feather)


def _clean_mask(m: np.ndarray, img_bgr: np.ndarray, shrink: int, feather: float) -> np.ndarray:
    """形态学清理 + 去微小斑点 + 可选边缘收缩/羽化。"""
    if m.max() == 0:
        return m
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 30:
            m[labels == i] = 0
    if shrink > 0:
        m = cv2.erode(m, np.ones((shrink * 2 + 1, shrink * 2 + 1), np.uint8))
    elif shrink < 0:
        m = cv2.dilate(m, np.ones((-shrink * 2 + 1, -shrink * 2 + 1), np.uint8))
    if feather > 0:
        m = cv2.GaussianBlur(m, (0, 0), feather)
    return m


def _mask_payload_b64(m: np.ndarray) -> str:
    """掩码以 RGBA PNG 返回：RGB=白，alpha=掩码值。前端 drawImage 后天然带透明度语义。"""
    rgba = np.zeros((m.shape[0], m.shape[1], 4), np.uint8)
    rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = 255
    rgba[..., 3] = m
    return img_to_b64(rgba)


@app.post("/api/extract")
async def api_extract(file: UploadFile = File(...),
                      engine: str = Form("grabcut"),
                      iters: int = Form(4),
                      shape: str = Form("polygon"),
                      geometry: str = Form(...),
                      shrink: int = Form(0),
                      feather: float = Form(0.0)):
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "图像解码失败")
    notes: list = []
    sel = _shape_mask(shape, geometry, img.shape[1], img.shape[0])
    engine_used, note = engine, ""
    if engine != "grabcut":
        if not _engine_available(engine):
            hint = ("缺少 %s" % os.path.join(
                WEIGHTS_DIR, "FastSAM-s.pt" if engine == "fastsam"
                else "mobile_sam.pt"))
            note = "%s 未部署（%s），已回退 GrabCut" % (engine.upper(), hint)
            engine_used = "grabcut"
        else:
            try:
                ys, xs = np.where(sel > 0)
                box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                m = _segment_with_sam(engine, img, box)
                m = _clean_mask(m, img, shrink, feather)
                ys, xs = np.where(m > 0)
                bbox = ([int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1),
                         int(ys.max() - ys.min() + 1)] if len(xs) else [0, 0, 0, 0])
                return JSONResponse({"mask_b64": _mask_payload_b64(m), "engine_used": engine,
                                     "bbox": bbox, "note": note, "notes": notes})
            except Exception as e:  # 权重加载/推理失败 → 降级
                note = "%s 推理失败（%s），已回退 GrabCut" % (engine.upper(), str(e)[:120])
                engine_used = "grabcut"
    m = _grabcut(img, sel, iters, shrink, feather, notes)
    ys, xs = np.where(m > 0)
    bbox = ([int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1),
             int(ys.max() - ys.min() + 1)] if len(xs) else [0, 0, 0, 0])
    return JSONResponse({"mask_b64": _mask_payload_b64(m), "engine_used": engine_used,
                         "bbox": bbox, "note": note, "notes": notes})


@app.post("/api/decode_image")
async def api_decode_image(request: Request):
    """浏览器解不了的格式（TIFF/16 位等）转 8 位 PNG。"""
    data = await request.body()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(400, "图像解码失败")
    if img.dtype != np.uint8:
        lo, hi = float(img.min()), float(img.max())
        img = np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-6) * 255.0,
                      0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(500, "PNG 编码失败")
    return Response(content=buf.tobytes(), media_type="image/png")


@app.get("/api/sam_status")
async def api_sam_status():
    return {
        "grabcut": True,
        "fastsam": _engine_available("fastsam"),
        "mobilesam": _engine_available("mobilesam"),
        "weights_dir": WEIGHTS_DIR,
    }


# ---------------------------------------------------------------- Poisson 预览

class PoissonPreviewReq(BaseModel):
    target_b64: str
    layer_b64: str          # RGBA patch（外观增强后）
    origin: list[int]       # patch 在目标图中的左上角
    blend_flag: str = "normal"
    feather_radius: float = 4.0
    opacity: float = 1.0
    harmonize: str = "off"
    harmonize_strength: float = 0.8
    harmonize_ring: int = 10


@app.post("/api/poisson_preview")
async def api_poisson_preview(req: PoissonPreviewReq):
    target = b64_to_img(req.target_b64)
    if target.ndim == 3 and target.shape[2] == 4:
        target = cv2.cvtColor(target, cv2.COLOR_BGRA2BGR)
    layer = to_bgra(b64_to_img(req.layer_b64))
    inst = {"blend": "poisson", "blend_flag": req.blend_flag,
            "feather_radius": req.feather_radius, "opacity": req.opacity,
            "harmonize": req.harmonize, "harmonize_strength": req.harmonize_strength,
            "harmonize_ring": req.harmonize_ring,
            "_origin": [int(req.origin[0]), int(req.origin[1])]}
    notes: list = []
    from .composite import blend_poisson, blend_direct_or_feather, harmonize_layer
    if inst["harmonize"] != "off":
        harmonize_layer(target, layer, inst)   # 预览同样先协调后融合
    canvas = target.astype(np.float32)
    canvas_u8 = np.clip(canvas, 0, 255).astype(np.uint8)
    ok = blend_poisson(canvas_u8, layer, inst, notes)
    if not ok:
        blend_direct_or_feather(canvas, layer, inst)
        canvas_u8 = np.clip(canvas, 0, 255).astype(np.uint8)
    return JSONResponse({"result_b64": img_to_b64(canvas_u8),
                         "used": "poisson" if ok else "feather", "notes": notes})


class PaintDefectReq(BaseModel):
    base_b64: str = ""       # 目标图 patch（BGR，PNG）；与 base_raw_b64 二选一
    mask_b64: str = ""       # 白=缺陷，alpha 承载笔迹软硬
    base_raw_b64: str = ""   # 原始像素 BGR（绕开浏览器 PNG 编码器，前端烘焙主路径）
    mask_raw_b64: str = ""   # 原始单通道 α
    raw_w: int = 0
    raw_h: int = 0
    kind: str = "spot_bright"  # spot_bright | spot_dark | scratch | stain | dent | plain
    strength: float = 0.7
    grain: float = 0.3
    seed: int = 7
    tint_rgb: list[int] | None = None   # 自定义颜色 RGB（三色盘/吸管），None=按类型自动明暗
    warp: dict | None = None            # 凹坑/划伤内陷扭曲 {strength, falloff, radius}


class RawEncodeReq(BaseModel):
    raw_b64: str             # BGR 原始像素（前端画布直读，绕开 convertToBlob 挂起）
    w: int
    h: int
    fmt: str = ".png"        # ".png" | ".jpg"（jpg 质量 95）


@app.get("/favicon.ico")
async def api_favicon():
    return Response(status_code=204)


@app.post("/api/raw_encode")
async def api_raw_encode(req: RawEncodeReq):
    """原始像素 → 编码图像（绘制模式"直接导出图像"用，全分辨率无损保存）。"""
    if req.w <= 0 or req.h <= 0 or req.w * req.h * 3 != len(base64.b64decode(req.raw_b64)):
        raise HTTPException(400, "raw 尺寸与数据不匹配")
    img = np.frombuffer(base64.b64decode(req.raw_b64), np.uint8).reshape(req.h, req.w, 3)
    ext = ".png" if req.fmt not in (".png", ".jpg") else req.fmt
    if ext == ".jpg":
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise HTTPException(500, "JPEG 编码失败")
        return JSONResponse({"img_b64": base64.b64encode(buf.tobytes()).decode("ascii"),
                             "mime": "image/jpeg", "ext": ".jpg"})
    return JSONResponse({"img_b64": img_to_b64(img), "mime": "image/png", "ext": ".png"})


@app.post("/api/paint_defect")
async def api_paint_defect(req: PaintDefectReq):
    """程序化缺陷绘制：把笔迹掩码着色为缺陷形貌层（RGBA），颜色继承底层纹理。"""
    from .composite import paint_defect_layer
    if req.base_raw_b64 and req.mask_raw_b64 and req.raw_w > 0 and req.raw_h > 0:
        base = np.frombuffer(base64.b64decode(req.base_raw_b64), np.uint8).reshape(
            req.raw_h, req.raw_w, 3)                     # BGR
        mask = np.frombuffer(base64.b64decode(req.mask_raw_b64), np.uint8).reshape(
            req.raw_h, req.raw_w)                        # 单通道 α
    else:
        base = b64_to_img(req.base_b64)
        if base.ndim == 3 and base.shape[2] == 4:
            base = cv2.cvtColor(base, cv2.COLOR_BGRA2BGR)
        mask_img = b64_to_img(req.mask_b64)
        if mask_img.ndim == 3:
            mask = mask_img[:, :, 3] if mask_img.shape[2] == 4 else cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        else:
            mask = mask_img
    if mask.shape[:2] != base.shape[:2]:
        mask = cv2.resize(mask, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST)
    tint_bgr = None
    if req.tint_rgb and len(req.tint_rgb) == 3:
        tint_bgr = [float(req.tint_rgb[2]), float(req.tint_rgb[1]), float(req.tint_rgb[0])]  # RGB→BGR
    warp = None
    if req.warp:
        try:
            # v1.4 语义：depth 坑缘最大弯移 px / falloff 衰减指数 / radius 影响带×r_in。
            # 前端只发这三个键；strength 为 v1.3 旧键，保留兼容
            warp = {}
            for k in ("depth", "falloff", "radius", "core", "shade", "swirl", "strength"):
                if req.warp.get(k) is not None:
                    v = float(req.warp[k])
                    if math.isfinite(v):
                        warp[k] = v
            if not warp:
                warp = None
        except (TypeError, ValueError):
            warp = None
    layer = paint_defect_layer(base, mask, req.kind, req.strength, req.grain, req.seed,
                               tint_bgr=tint_bgr, warp=warp)
    # 原始 RGBA 直传：前端用 ImageData 重建（data:URL fetch + createImageBitmap
    # 在部分环境偶发永久挂起，曾导致形貌预览层整体不渲染）
    rgba = np.ascontiguousarray(np.dstack([layer[..., 2], layer[..., 1], layer[..., 0], layer[..., 3]]))
    return JSONResponse({
        "layer_b64": img_to_b64(layer),
        "layer_raw_b64": base64.b64encode(rgba.tobytes()).decode("ascii"),
        "w": int(layer.shape[1]), "h": int(layer.shape[0]),
    })


# ---------------------------------------------------------------- 放置适宜度热力图

class PlacementReq(BaseModel):
    target_b64: str
    top_k: int = 12


class InstancePreviewReq(BaseModel):
    target_b64: str            # 代理全图（imdecode 可解码）
    rgba_raw_b64: str          # 实例层原始像素 RGBA（前端 getImageData 直传，零编码风险）
    w: int
    h: int
    instance: dict             # 与 InstancePayload 同构（position 为 [x,y] 数组）


@app.post("/api/instance_preview")
async def api_instance_preview(req: InstancePreviewReq):
    """单实例预览——与导出共用 render_scene（融合顺序/羽化/协调完全一致）。

    旧 /api/poisson_preview 自行拼装融合流程，与 render_scene 存在顺序差异，
    预览与实际导出会不符；此端点内联资产走同一权威路径，所见即所得。"""
    exp_len = req.w * req.h * 4
    if req.w <= 0 or req.h <= 0 or exp_len != len(base64.b64decode(req.rgba_raw_b64)):
        raise HTTPException(400, "rgba 尺寸与数据不匹配")
    target = b64_to_img(req.target_b64)
    if target.ndim == 3 and target.shape[2] == 4:
        target = cv2.cvtColor(target, cv2.COLOR_BGRA2BGR)
    if target.ndim == 2:
        target = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)
    rgba = np.frombuffer(base64.b64decode(req.rgba_raw_b64), np.uint8).reshape(
        req.h, req.w, 4)[:, :, [2, 1, 0, 3]]           # RGBA → BGRA
    inst = dict(req.instance)
    inst["asset_id"] = "inline"
    notes: list = []
    out = render_scene(target, {"inline": np.ascontiguousarray(rgba)}, [inst], notes)
    return JSONResponse({"result_b64": img_to_b64(out), "notes": notes})


class SceneItem(BaseModel):
    rgba_raw_b64: str
    w: int
    h: int
    instance: dict


class ScenePreviewReq(BaseModel):
    target_b64: str
    items: list[SceneItem] = []


@app.post("/api/scene_preview")
async def api_scene_preview(req: ScenePreviewReq):
    """整场景预览：所有实例一次性交给 render_scene（与导出同一权威路径）。

    逐实例补丁预览只把该实例合成到原背景上，补丁矩形内会覆盖掉其它实例，
    多缺陷场景下"预览与导出不符"。整场景预览把全部实例一起合成，结果与
    导出（view 分辨率）逐像素一致。"""
    target = b64_to_img(req.target_b64)
    if target.ndim == 3 and target.shape[2] == 4:
        target = cv2.cvtColor(target, cv2.COLOR_BGRA2BGR)
    if target.ndim == 2:
        target = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)
    assets: dict = {}
    instances: list = []
    for i, it in enumerate(req.items):
        n = it.w * it.h * 4
        if it.w <= 0 or it.h <= 0 or n != len(base64.b64decode(it.rgba_raw_b64)):
            raise HTTPException(400, "rgba 尺寸与数据不匹配")
        rgba = np.frombuffer(base64.b64decode(it.rgba_raw_b64), np.uint8).reshape(
            it.h, it.w, 4)[:, :, [2, 1, 0, 3]]              # RGBA → BGRA
        aid = "i%d" % i
        assets[aid] = np.ascontiguousarray(rgba)
        inst = dict(it.instance)
        inst["asset_id"] = aid
        instances.append(inst)
    notes: list = []
    out = render_scene(target, assets, instances, notes)
    # 只回原始 RGBA（预览用不上 PNG 编码，省一半传输与编码开销）
    raw = np.ascontiguousarray(np.dstack([out[..., 2], out[..., 1], out[..., 0],
                                          np.full(out.shape[:2], 255, np.uint8)]))
    return JSONResponse({
        "result_raw_b64": base64.b64encode(raw.tobytes()).decode("ascii"),
        "w": int(out.shape[1]), "h": int(out.shape[0]), "notes": notes,
    })


@app.post("/api/placement_map")
async def api_placement_map(req: PlacementReq):
    """放置适宜度：梯度能量低（表面平滑）处适合贴缺陷。

    参考图像合成领域的物体放置评估思想（OPA 等，见 bcmi/Awesome-Image-Composition / Libcom），
    确定性实现：Sobel 梯度能量 → 高斯平滑 → 归一化反转 → jet 热力图 + 平滑热点 top-K。
    """
    target = b64_to_img(req.target_b64)
    if target.ndim == 3 and target.shape[2] == 4:
        target = cv2.cvtColor(target, cv2.COLOR_BGRA2BGR)
    th, tw = target.shape[:2]
    scale = min(1.0, 256.0 / max(th, tw))
    small = cv2.resize(target, (int(tw * scale), int(th * scale)),
                       interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    energy = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 6.0)
    lo, hi = float(energy.min()), float(np.percentile(energy, 99))
    suit = 1.0 - np.clip((energy - lo) / max(hi - lo, 1e-6), 0, 1)
    heat_u8 = (suit * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_b64 = img_to_b64(heat_color)

    # top-K 平滑热点：膨胀式非极大值抑制
    spots = []
    k = max(8, int(min(heat_u8.shape) * 0.08))
    work = suit.copy()
    for _ in range(max(1, min(24, int(req.top_k)))):
        _, _, _, mx = cv2.minMaxLoc(work)   # 取最大适宜度位置
        if mx is None or mx[0] < 0:
            break
        score = float(work[mx[1], mx[0]])
        if score <= 0.05:
            break
        spots.append({"x": int((mx[0] + 0.5) / scale), "y": int((mx[1] + 0.5) / scale),
                      "score": round(score, 3)})
        cv2.circle(work, mx, k, 0.0, -1)
    return JSONResponse({"heat_b64": heat_b64, "w": tw, "h": th,
                         "map_w": int(tw * scale), "map_h": int(th * scale),
                         "spots": spots})


# ---------------------------------------------------------------- 合成请求模型

class AssetPayload(BaseModel):
    rgba_b64: str
    mask_b64: str | None = None


class InstancePayload(BaseModel):
    asset_id: str
    position: list[float] | dict = [0, 0]      # 兼容 [x,y] 与 {x,y}
    affine: dict = Field(default_factory=dict)
    perspective_h: list | None = None
    mesh: dict | None = None
    appearance: dict = Field(default_factory=dict)
    blend: str = "poisson"
    blend_flag: str = "normal"
    feather_radius: float = 0.0
    edge_blur: float = 0.0                     # 边缘模糊 px（v1.4）
    opacity: float = 1.0
    noise_seed: int = 0
    harmonize: str = "off"                     # 光照协调: off | ring
    harmonize_strength: float = 0.8
    harmonize_ring: int = 10
    rel_position: list[float] | None = None    # 批量用：位置占目标图宽高比例
    scale_anchor: float | None = None          # 批量用：缩放锚定的源短边像素数


class CompositeReq(BaseModel):
    target_b64: str
    assets: dict[str, AssetPayload]
    instances: list[InstancePayload]
    format: str = "png"
    quality: int = 95
    filename: str = "composite"


@app.post("/api/composite")
async def api_composite(req: CompositeReq):
    target = b64_to_img(req.target_b64)
    if target.ndim == 3 and target.shape[2] == 4:
        target = cv2.cvtColor(target, cv2.COLOR_BGRA2BGR)
    assets = {aid: to_bgra(b64_to_img(a.rgba_b64)) for aid, a in req.assets.items()}
    notes: list = []
    out = render_scene(target, assets, [i.model_dump() for i in req.instances], notes)
    data, ext, media = encode_image(out, req.format, req.quality)
    name = (req.filename or "composite") + ext
    return Response(data, media_type=media,
                    headers={"Content-Disposition": 'attachment; filename="%s"' % name,
                             "X-Composite-Notes": json.dumps(notes, ensure_ascii=False)})


# ---------------------------------------------------------------- 批量作业

class BatchReq(BaseModel):
    assets: dict[str, AssetPayload]
    instances: list[InstancePayload]
    background: dict                      # {mode:'dir', dir} | {mode:'files', files:[{name,b64}]}
    output_dir: str
    format: str = "png"
    quality: int = 95
    jitter: dict | None = None            # {pos_pct, rot_deg, brightness}
    seed: int = 42


def _run_batch(job_id: str, req: BatchReq):
    job = _JOBS[job_id]
    try:
        assets = {aid: to_bgra(b64_to_img(a.rgba_b64)) for aid, a in req.assets.items()}
        if req.background.get("mode") == "dir":
            d = req.background["dir"]
            paths = sorted(p for p in glob.glob(os.path.join(d, "*")) if p.lower().endswith(IMG_EXTS))
            if not paths:
                job["error"] = "背景目录中未找到图像文件: %s" % d
                job["done"] = True
                return

            def load(idx, _paths=paths):
                p = _paths[idx]
                return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR), os.path.basename(p)
        else:
            items = [(b64_to_img(f["b64"]), f["name"]) for f in req.background.get("files", [])]
            paths = [f["name"] for f in req.background.get("files", [])]

            def load(idx, _items=items):
                return _items[idx]
        os.makedirs(req.output_dir, exist_ok=True)
        job["total"] = len(paths)
        base_instances = [i.model_dump() for i in req.instances]
        ok = fail = skip = 0
        for i, path in enumerate(paths):
            if job["cancel"]:
                job["cancelled"] = True
                break
            t0 = time.time()
            name = os.path.basename(path)
            try:
                img, bname = load(i)
                if img is None:
                    raise ValueError("解码失败")
                if img.ndim == 3 and img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                insts = jitter_instances(
                    materialize_instances(base_instances, img.shape[1], img.shape[0]),
                    img.shape[1], img.shape[0], req.jitter, i, req.seed)
                if not insts:
                    skip += 1
                    job["results"].append({"name": bname, "status": "skipped",
                                           "note": "无实例", "ms": 0})
                    job["done_count"] = i + 1
                    continue
                notes: list = []
                out = render_scene(img, assets, insts, notes)
                stem = os.path.splitext(bname)[0]
                data, ext, _ = encode_image(out, req.format, req.quality)
                out_path = os.path.join(req.output_dir, stem + "_synth" + ext)
                with open(out_path, "wb") as f:
                    f.write(data)
                ok += 1
                job["results"].append({"name": bname, "status": "ok",
                                       "note": "；".join(notes) if notes else "",
                                       "ms": int((time.time() - t0) * 1000)})
            except Exception as e:
                fail += 1
                job["results"].append({"name": name, "status": "failed",
                                       "note": str(e)[:200], "ms": int((time.time() - t0) * 1000)})
            job["done_count"] = i + 1
        job["summary"] = {"ok": ok, "failed": fail, "skipped": skip,
                          "total_ms": int((time.time() - job["t0"]) * 1000)}
    except Exception as e:
        job["error"] = str(e)[:300]
        job["traceback"] = traceback.format_exc()[-800:]
    finally:
        job["done"] = True


@app.post("/api/batch/start")
async def api_batch_start(req: BatchReq):
    bg = req.background or {}
    if bg.get("mode") == "dir":
        if not os.path.isdir(bg.get("dir", "")):
            raise HTTPException(400, "背景目录不存在: %s" % bg.get("dir"))
    elif bg.get("mode") != "files" or not bg.get("files"):
        raise HTTPException(400, "background 需为 {mode:'dir',dir} 或 {mode:'files',files:[...]}")
    if not os.path.isdir(os.path.dirname(os.path.abspath(req.output_dir))):
        raise HTTPException(400, "输出目录的父目录不存在: %s" % req.output_dir)
    jid = uuid.uuid4().hex[:12]
    _JOBS[jid] = {"total": 0, "done_count": 0, "results": [], "done": False,
                  "cancel": False, "cancelled": False, "error": None,
                  "summary": None, "t0": time.time()}
    th = threading.Thread(target=_run_batch, args=(jid, req), daemon=True)
    th.start()
    return {"job_id": jid}


@app.get("/api/batch/status/{jid}")
async def api_batch_status(jid: str):
    job = _JOBS.get(jid)
    if job is None:
        raise HTTPException(404, "作业不存在")
    return job


@app.post("/api/batch/cancel/{jid}")
async def api_batch_cancel(jid: str):
    job = _JOBS.get(jid)
    if job is None:
        raise HTTPException(404, "作业不存在")
    job["cancel"] = True
    return {"ok": True}


@app.get("/api/health")
async def health():
    return {"ok": True}


# ---------------------------------------------------------------- 静态前端（必须最后挂载，避免遮蔽 /api 路由）

if os.path.isdir(os.path.join(ROOT, "testdata")):
    app.mount("/testdata", StaticFiles(directory=os.path.join(ROOT, "testdata")), name="testdata")
app.mount("/", StaticFiles(directory=os.path.join(ROOT, "web"), html=True), name="web")


def create_app() -> "FastAPI":
    """应用工厂：供 Qt 壳窗口在子线程 uvicorn 中托管；路由/挂载在 import 期
    构建，工厂返回同一实例，端口由调用方以 port=0 动态分配。"""
    return app
