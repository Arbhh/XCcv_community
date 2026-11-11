# -*- coding: utf-8 -*-
"""AI 智能标注引擎（V0.6 T5-2）：SAM2 / MobileSAM 经 ultralytics 预测器直连。

选型（R4 修订：GPU 环境已到位，onnxruntime 仅 CPU EP——改为 ultralytics 原生
SAM 预测器，编码走 GPU/CPU 均可，与训练/推理共用同一依赖，无新 wheels）：
- 默认质量档 **SAM2.1 Hiera-Tiny**（weights/sam/sam2.1_t.pt，~74MB）；
- 轻量档 **MobileSAM**（weights/sam/mobile_sam.pt，~40MB）。
权重缺失时 `available_models()` 为空、调用方按钮置灰并给中文指引（T5-3）。

性能契约（G-8 验收口径）：
- **编码器按图缓存**：predictor.set_image 一次，之后逐次点提示只跑轻量解码器
  （canvas 侧再按 image.tobytes() 缓存嵌入，切图才重算）；
- 解码（点提示）毫秒~百毫秒级；编码 GPU ~百毫秒级 / CPU 数秒（一次性）。

对外接口（与 labelme 画布骨架的 osam 形状契约对齐，widgets/canvas.py 直接消费）：
- `available_models()` → [{name, title, weights, available, message}]；
- `get_model_type_by_name(name)` → 模型类（canvas: `osam.apis.get_model_type_by_name`）；
- 模型对象：`.name` + `.encode_image(image) -> embedding`（canvas: `sam.encode_image`）；
- `generate(request)` → `GenerateResponse(annotations=[Annotation(mask, bounding_box)])`
  （canvas: `osam.apis.generate(GenerateRequest(model, image_embedding, Prompt))`）；
- `decode_points(weights, image_bgr, points, labels, multimask)` → (masks, scores)
  （供批量预标注 T5-4 与画布悬停预览直接调用，绕开请求对象）。

线程纪律：编码/解码在【工作线程】执行（画布悬停预览由 QTimer 节流 + 后台线程；
批量预标注走 QThread）——同一 predictor 并发调用不安全，加 threading.Lock。
"""

import itertools
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / "sam"

# 权重注册（name 与 labelme config `ai.default` 对齐的显示名见 TITLES）
_MODELS = [
    {"name": "Sam2 (balanced)", "title": "SAM2.1-T：均衡（默认）",
     "file": "sam2.1_t.pt", "predictor": "sam2"},
    {"name": "Sam2.1-S (精度+)", "title": "SAM2.1-S：精度更高",
     "file": "sam2.1_s.pt", "predictor": "sam2"},
    {"name": "Sam2.1-B (高精度)", "title": "SAM2.1-B+：高精度（建议 GPU）",
     "file": "sam2.1_b.pt", "predictor": "sam2"},
    {"name": "Sam2.1-L (最高精度)", "title": "SAM2.1-L：最高精度（需大显存）",
     "file": "sam2.1_l.pt", "predictor": "sam2"},
    {"name": "EfficientSAM (onnx·快)", "title": "EfficientSAM：CPU 快速",
     "file": "efficient_sam_vits_encoder.onnx",
     "file2": "efficient_sam_vits_decoder.onnx", "predictor": "efficientsam"},
    {"name": "MobileSAM (fast)", "title": "MobileSAM：轻量",
     "file": "mobile_sam.pt", "predictor": "sam"},
    {"name": "SAM-B (ViT-B)", "title": "SAM ViT-B：经典版",
     "file": "sam_b.pt", "predictor": "sam"},
]

# 用户可调推理参数（标注平台「AI 标注参数」对话框写入）
_PARAMS = {
    "zoom": True,            # 小目标放大裁剪重解
    "zoom_win": 256,         # 裁剪窗边长（原图像素）
    "zoom_trigger": 0.05,    # 全图掩膜 bbox 占比超过即触发裁剪重解
    "mask_threshold": None,  # 掩膜 logit 阈值（None=模型默认 0；调高更紧）
    "pick": "auto",          # 候选：auto / anchor=含首锚最小 / score=最高分
    "device": "auto",        # auto / cpu / cuda
    "half": True,            # GPU 上用半精度（编码约快 2 倍）
}
_ENC_SEQ = itertools.count(1)
_NN: dict = {}
_CROP_CACHE: OrderedDict = OrderedDict()


def get_params() -> dict:
    return dict(_PARAMS)


def set_params(**kw) -> None:
    """更新推理参数；设备变更时丢弃已建预测器（下次使用按新设备重建）。"""
    old_dev = (_PARAMS.get("device"), bool(_PARAMS.get("half")))
    for k, v in kw.items():
        if k in _PARAMS:
            _PARAMS[k] = v
    if (_PARAMS.get("device"), bool(_PARAMS.get("half"))) != old_dev:
        with _LOCK:
            _ENGINES.clear()
            _CROP_CACHE.clear()
    else:
        _CROP_CACHE.clear()


def _device_arg() -> str:
    d = str(_PARAMS.get("device") or "auto")
    return {"cpu": "cpu", "cuda": "0"}.get(d, "")


def _mask_thr(pred) -> float:
    t = _PARAMS.get("mask_threshold")
    return float(pred.model.mask_threshold) if t is None else float(t)


def _pick_mode(engine: str) -> str:
    p = _PARAMS.get("pick", "auto")
    if p in ("anchor", "score"):
        return p
    return "anchor" if engine == "efficientsam" else "score"


def _fills(m) -> bool:
    """掩膜外接框贴住窗口 ≥3 条边（≈铺满裁剪窗）。"""
    ys, xs = np.nonzero(m)
    if not len(xs):
        return False
    h, w = m.shape[-2], m.shape[-1]
    return (int(xs.min() <= 1) + int(ys.min() <= 1)
            + int(xs.max() >= w - 2) + int(ys.max() >= h - 2)) >= 3


def _window_decode(cpred, weights, seq, rgb, x1, y1, x2, y2, points, labels):
    """以窗口 (x1,y1,x2,y2) 局部重解：返回 (窗内候选 bool[K,h,w], 分数[K])。"""
    from ultralytics.utils import ops

    _crop_state(cpred, weights, seq, rgb, x1, y1, x2, y2)
    cw, ch = x2 - x1, y2 - y1
    allp = [[float(p[0]) - x1, float(p[1]) - y1] for p in points]
    keep = [k for k, (cx, cy) in enumerate(allp)
            if -64 <= cx <= cw + 64 and -64 <= cy <= ch + 64] \
        or list(range(len(allp)))
    ret = cpred.prompt_inference(
        cpred._xcv_im,
        points=np.asarray([allp[k] for k in keep], dtype=np.float32),
        labels=np.asarray([labels[k] for k in keep], dtype=np.int64),
        multimask_output=True)
    mr, sr = ret[0], ret[1]
    sc = sr.detach().cpu().numpy() if hasattr(sr, "detach") \
        else np.asarray(sr)
    m2 = mr.unsqueeze(1).float() if mr.dim() == 3 else mr.float()
    up = ops.scale_masks(m2, (ch, cw), padding=False)
    if up.dim() == 4:
        up = up[:, 0]
    ms = (up.detach().cpu().numpy() if hasattr(up, "detach")
          else np.asarray(up)) > _mask_thr(cpred)
    return ms, np.asarray(sc).reshape(-1)


def _crop_state(cpred, weights, seq, rgb, x1, y1, x2, y2):
    """裁剪窗编码（按 原图编码序号+窗口 缓存）：同一多边形的后续点击/悬停
    裁剪中心=首锚点不变，命中缓存只跑解码器（原实现每次重编码 1~3s）。"""
    key = (weights, seq, int(x1), int(y1), int(x2), int(y2))
    st = _CROP_CACHE.get(key)
    if st is None:
        import torch

        crop = np.ascontiguousarray(rgb[y1:y2, x1:x2])
        cbgr = np.ascontiguousarray(crop[..., ::-1]) \
            if crop.ndim == 3 else crop
        with torch.no_grad():
            cpred.set_image(cbgr)
            cb = next(iter(cpred.dataset))
            st = (cpred.features, cb, cpred.preprocess(cb[1]))
        _CROP_CACHE[key] = st
        while len(_CROP_CACHE) > 12:
            _CROP_CACHE.popitem(last=False)
    else:
        _CROP_CACHE.move_to_end(key)
    cpred.features, cpred.batch, cpred._xcv_im = st

_LOCK = threading.Lock()          # ultralytics predictor 非线程安全
_ENGINES: dict = {}               # weights path -> predictor（进程内缓存）

# —— 小目标放大裁剪重解（2026-11-05 测试批）——
# 极小目标（如 7×20px）在 SAM 全图 1024 输入中仅 ~2px，得分远低于大
# 背景块（实测 0.005 vs 0.77），按得分 argmax 仍选中整块背景。当落点
# 命中"全图尺度大掩膜"时，以落点为中心裁窗局部重解（不再整体下采样），
# 小目标在窗内放大 2 倍即可被正确分割。
_ZOOM_TRIGGER = 0.05   # 更积极走裁剪重解（用户反馈：单锚偶取全图主体）   # 全图最佳掩膜 bbox 面积占比超过→触发裁剪重解
_ZOOM_WIN = 256        # 裁剪窗边长（原图像素；SAM 输入 1024→4 倍放大，利于 4~10px 小目标）
_ZOOM_KEEP = 0.7    # 窗内占比过半即保留全图结果 → 收紧为 0.7      # 窗内最佳掩膜占窗比超过→目标本就是大块，保留全图结果
_LAST_RGB: dict = {}   # weights -> 最近编码原图（_engine_by_name 每次
                       # 返回新包装实例，原图必须挂模块级才能跨实例共享）


def weights_dir() -> Path:
    return _WEIGHTS_DIR


def available_models() -> list:
    out = []
    for m in _MODELS:
        p = _WEIGHTS_DIR / m["file"]
        ok = p.is_file() and (m.get("file2") is None
                              or (_WEIGHTS_DIR / m["file2"]).is_file())
        out.append({**m, "weights": str(p), "available": ok,
                    "message": "" if ok else
                    f"缺少权重文件 {p}（随安装包分发）"})
    return out


def _predictor_for(weights: str, cache_key: str | None = None):
    """确保预测器就绪（任务类型对应的 SAM2/SAM 预测器类；进程内缓存）。

    cache_key：独立缓存键（如 "<weights>#crop"）——裁剪重解用独立
    predictor，set_image 不冲掉全图编码缓存。
    """
    dev = _device_arg()
    use_half = bool(_PARAMS.get("half", True)) and dev != "cpu" \
        and Path(weights).name.lower().startswith("sam2")
    if use_half:
        try:
            import torch

            use_half = torch.cuda.is_available()
        except Exception:  # noqa: BLE001
            use_half = False
    key = (cache_key or weights) + "@" + (dev or "auto") + \
        ("#fp16" if use_half else "")
    eng = _ENGINES.get(key)
    if eng is not None:
        return eng
    from ultralytics import SAM

    nn_key = weights + "@" + (dev or "auto") + ("#fp16" if use_half else "")
    model = _NN.get(nn_key)
    if model is None:   # 全图/裁剪两个预测器共享同一网络（权重只载一次）
        model = _NN[nn_key] = SAM(weights)
    # Model.get_predictor / 预测器构建：走一次空推理会浪费编码——直接手工装配
    task_map = model.task_map   # 实例 property（类上是 property 对象）
    pred_cls = task_map["segment"]["predictor"]
    ov = dict(model=weights, mode="predict", task="segment",
              conf=0.25, imgsz=1024, verbose=False)
    if dev:
        ov["device"] = dev
    if use_half:
        ov["half"] = True
    eng = pred_cls(overrides=ov)
    eng.setup_model(model.model)
    eng.model = model.model
    _ENGINES[key] = eng
    return eng


@dataclass
class Point:
    x: float
    y: float


@dataclass
class BoundingBox:
    ymin: float
    xmin: float
    ymax: float
    xmax: float


@dataclass
class ImageEmbedding:
    """编码结果句柄（predictor 内部 features + 所属引擎）。"""

    weights: str
    key: bytes = b""
    meta: dict = field(default_factory=dict)


@dataclass
class Prompt:
    points: list = field(default_factory=list)          # [[x, y], ...] 原图像素
    point_labels: list = field(default_factory=list)    # 1=前景点 0=背景点
    bboxes: list = field(default_factory=list)


@dataclass
class Annotation:
    mask: np.ndarray                                    # (H, W) bool，原图尺寸
    bounding_box: BoundingBox | None = None
    score: float = 0.0


@dataclass
class GenerateRequest:
    model: str = ""
    image_embedding: ImageEmbedding | None = None
    prompt: Prompt = field(default_factory=Prompt)
    image: np.ndarray | None = None                     # 嵌入缺失时随请求补编码


@dataclass
class GenerateResponse:
    annotations: list = field(default_factory=list)


class _UltralyticsSam:
    """osam.types.Model 形状的引擎包装（canvas.initializeAiModel 直接消费）。"""

    def __init__(self, name: str, weights: str):
        self.name = name
        self.weights = weights

    @property
    def _last_rgb(self):
        return _LAST_RGB.get(self.weights)

    def encode_image(self, image: np.ndarray) -> ImageEmbedding:
        import torch

        with torch.no_grad():
            return self._encode_image(image)

    def _encode_image(self, image: np.ndarray) -> ImageEmbedding:
        """编码并缓存（predictor.features）；image 为 RGB(A) uint8（canvas
        侧 img_qt_to_arr 产物为 RGBA 4 通道——丢 alpha 后再编码）。

        同时缓存预处理张量与 batch（ultralytics set_image 不持久化 batch，
        解码端 prompt_inference 需要——见 _UltralyticsSam.decode）。
        """
        with _LOCK:
            pred = _predictor_for(self.weights)
            if image.ndim == 3 and image.shape[-1] == 4:
                image = image[..., :3]   # RGBA -> RGB（canvas QImage 路径）
            bgr = np.ascontiguousarray(image[..., ::-1]) \
                if (image.ndim == 3 and image.shape[-1] == 3) \
                else np.ascontiguousarray(image)
            pred.set_image(bgr)   # features 缓存在 predictor 内
            batch = next(iter(pred.dataset))     # 单图源可重复迭代
            pred.batch = batch
            pred._xcv_im = pred.preprocess(batch[1])
            rgb = np.asarray(image)
            _LAST_RGB[self.weights] = rgb        # 供小目标裁剪重解
            seq = next(_ENC_SEQ)
            pred._xcv_seq = seq
            state = (pred.features, batch, pred._xcv_im, rgb)
        # 嵌入自带编码态：predictor 只保存"最后一次编码"，画布按图缓存多张
        # 嵌入时必须按句柄恢复，否则切回旧图会用别的图的特征解码
        return ImageEmbedding(weights=self.weights,
                              meta={"seq": seq, "state": state})

    def warmup(self):
        """模型后台加载阶段预热：建好全图/裁剪预测器并各跑一次编码+解码
        （首次 CUDA 内核初始化挪出首次点击）。"""
        img = np.full((320, 320, 3), 128, np.uint8)
        img[140:180, 140:180] = 40
        emb = self.encode_image(img)
        self.decode(emb, np.array([[160.0, 160.0]]), [1])
        with _LOCK:
            _predictor_for(self.weights, cache_key=self.weights + "#crop")

    def _decode(self, points, labels, multimask: bool, emb=None):
        from ultralytics.utils import ops

        with _LOCK:
            pred = _predictor_for(self.weights)
            meta = getattr(emb, "meta", None) or {}
            st = meta.get("state")
            if st is not None and getattr(pred, "_xcv_seq", None) \
                    != meta.get("seq"):
                pred.features, pred.batch, pred._xcv_im, _rgb = st
                pred._xcv_seq = meta.get("seq")
                _LAST_RGB[self.weights] = _rgb
            seq = getattr(pred, "_xcv_seq", 0)
            zw = int(_PARAMS.get("zoom_win") or _ZOOM_WIN)
            b1 = getattr(pred, "batch", None)
            if b1 is None or b1[1] is None:
                raise ValueError("编码缓存失效（请先 encode_image）")
            f0 = b1[1][0] if isinstance(b1[1], (list, tuple)) else b1[1]
            raw_hw = tuple(int(v) for v in np.asarray(f0).shape[:2])
            ret = pred.prompt_inference(
                pred._xcv_im, points=np.asarray(points, dtype=np.float32),
                labels=np.asarray(labels, dtype=np.int64),
                multimask_output=bool(multimask))
            masks_raw, scores = ret[0], ret[1]   # SAM2 返回 2 元组，SAM 或 3 元组
            scores = scores.detach().cpu().numpy() if hasattr(scores, "detach") \
                else np.asarray(scores)
            # 解码分辨率 → 原图尺寸（与 postprocess 同口径的 letterbox 缩放）
            m = masks_raw.unsqueeze(1).float() if masks_raw.dim() == 3 \
                else masks_raw.float()   # scale_masks 需要 (N, 1, H, W)
            up = ops.scale_masks(m, raw_hw, padding=False)
            if up.dim() == 4:
                up = up[:, 0]   # (N, 1, H, W) -> (N, H, W)，保留候选维
            if hasattr(up, "detach"):
                masks = (up > _mask_thr(pred)).cpu().numpy()
            else:
                masks = np.asarray(up) > _mask_thr(pred)
            # —— 小目标放大裁剪重解 ——
            if self._last_rgb is not None and _PARAMS.get("zoom", True):
                pick = 0
                if masks.ndim == 3 and masks.shape[0] > 1 \
                        and scores.size > 1 and multimask:
                    pick = int(np.argmax(scores))
                m0 = masks[pick]
                _xs = np.flatnonzero(m0.any(axis=0))
                _ys = np.flatnonzero(m0.any(axis=1))
                H, W = m0.shape[-2], m0.shape[-1]
                _fx0 = int(min(max(float(points[0][0]), 0), W - 1))
                _fy0 = int(min(max(float(points[0][1]), 0), H - 1))
                _miss = not bool(m0[_fy0, _fx0])   # 首锚不在掩膜内=点哪没标哪
                # 有正锚点落在裁剪窗（±64 余量）之外 = 用户在框比窗口更大的
                # 目标：不走窗内重解，用整图结果（单击小目标行为不变；大目标
                # 两击即可整体选中——此前大目标在 zoom 下永远只出窗内一小块）
                _reach = zw // 2 + 64
                _far = any(
                    int(lb) == 1 and (
                        abs(float(p[0]) - float(points[0][0])) > _reach
                        or abs(float(p[1]) - float(points[0][1])) > _reach)
                    for p, lb in zip(points, labels))
                if len(_xs) and not _far and (
                        _miss
                        or float(_xs.max() - _xs.min()) *
                        float(_ys.max() - _ys.min()) >
                        float(_PARAMS.get("zoom_trigger", _ZOOM_TRIGGER))
                        * W * H):
                    # 裁剪中心=首锚点（2026-11-06 锚点加权：首锚主导，
                    # 与加权提示的 points[0] 一致；原取末锚与首锚主导矛盾）
                    px = float(np.clip(float(points[0][0]), 0, W - 1))
                    py = float(np.clip(float(points[0][1]), 0, H - 1))
                    half = zw // 2
                    x1 = int(min(max(px - half, 0), max(W - zw, 0)))
                    y1 = int(min(max(py - half, 0), max(H - zw, 0)))
                    x2, y2 = min(x1 + zw, W), min(y1 + zw, H)
                    cpred = _predictor_for(
                        self.weights, cache_key=self.weights + "#crop")
                    _crop_state(cpred, self.weights, seq, self._last_rgb,
                                x1, y1, x2, y2)
                    # 窗外提示点过滤（加权展开后大量重复点在窗外时，
                    # 只保留窗口±64px 内的提示，避免窗外点干扰局部解码；
                    # 首锚点=裁剪中心，必在窗内）
                    _cw, _ch = x2 - x1, y2 - y1
                    _all = [[float(p[0]) - x1, float(p[1]) - y1]
                            for p in points]
                    _keep = [k for k, (cx, cy) in enumerate(_all)
                             if -64 <= cx <= _cw + 64
                             and -64 <= cy <= _ch + 64]
                    if not _keep:
                        _keep = list(range(len(_all)))
                    cpts = np.asarray([_all[k] for k in _keep],
                                      dtype=np.float32)
                    clabs = np.asarray(
                        [labels[k] for k in _keep], dtype=np.int64)
                    cret = cpred.prompt_inference(
                        cpred._xcv_im, points=cpts,
                        labels=clabs,
                        multimask_output=True)
                    cm_raw, cs_raw = cret[0], cret[1]
                    cs_np = cs_raw.detach().cpu().numpy() \
                        if hasattr(cs_raw, "detach") else np.asarray(cs_raw)
                    cm2 = cm_raw.unsqueeze(1).float() \
                        if cm_raw.dim() == 3 else cm_raw.float()
                    cup = ops.scale_masks(cm2, (y2 - y1, x2 - x1),
                                          padding=False)
                    if cup.dim() == 4:
                        cup = cup[:, 0]
                    cmasks = (cup.detach().cpu().numpy()
                              if hasattr(cup, "detach") else
                              np.asarray(cup)) > _mask_thr(cpred)
                    # 窗内候选选择与整图路径一致：包含首锚的候选中取
                    # 面积最小（防窗内仍取大结构，2026-11-06 小目标优化）
                    _hit = []
                    try:
                        _cw2, _ch2 = cmasks.shape[1], cmasks.shape[2]
                        _cx2, _cy2 = int(cpts[0][0]), int(cpts[0][1])
                        _hit = []
                        for _k in range(cmasks.shape[0]):
                            _mk = cmasks[_k]
                            if 0 <= _cy2 < _ch2 and 0 <= _cx2 < _cw2 \
                                    and bool(_mk[_cy2, _cx2]):
                                _hit.append((int(_mk.sum()), _k))
                        if _hit:
                            cbest = min(_hit)[1]
                    except Exception:  # noqa: BLE001
                        pass
                    _big = False
                    if cmasks.ndim == 3:
                        cbest = int(np.argmax(cs_np)) \
                            if cs_np.size > 1 else 0
                        if _pick_mode("sam") == "anchor" and _hit:
                            cbest = min(_hit)[1]
                        elif _fills(cmasks[cbest]) and _hit:
                            # 最高分候选铺满裁剪窗：目标大于窗口，或低对比
                            # 小目标被背景板吞并——放大窗口复核
                            zw2 = min(2 * zw, max(W, H))
                            bx1 = int(min(max(px - zw2 // 2, 0),
                                          max(W - zw2, 0)))
                            by1 = int(min(max(py - zw2 // 2, 0),
                                          max(H - zw2, 0)))
                            bx2, by2 = min(bx1 + zw2, W), min(by1 + zw2, H)
                            bms, bsc = _window_decode(
                                cpred, self.weights, seq, self._last_rgb,
                                bx1, by1, bx2, by2, points, labels)
                            _ax, _ay = int(px) - bx1, int(py) - by1
                            _bh = [(float(bsc[k]), k)
                                   for k in range(bms.shape[0])
                                   if 0 <= _ay < bms.shape[1]
                                   and 0 <= _ax < bms.shape[2]
                                   and bool(bms[k][_ay, _ax])
                                   and not _fills(bms[k])]
                            _bk = max(_bh)[1] if _bh else None
                            if _bk is not None and not _fills(bms[_bk]):
                                _gm = bms[_bk]
                                _y2s, _x2s = np.nonzero(_gm)
                                _span = max(int(_x2s.max() - _x2s.min()),
                                            int(_y2s.max() - _y2s.min()))
                                _m0w = m0[by1:by2, bx1:bx2]
                                _un = int(np.logical_or(_gm, _m0w).sum())
                                _iou0 = (int(np.logical_and(_gm, _m0w).sum())
                                         / _un) if _un else 0.0
                                # 确为更大目标：尺寸接近/超过原窗，且不是
                                # 全图解码已看到的同一大结构（背景板/纹理区）
                                if _span >= 0.8 * zw and _iou0 < 0.3:
                                    cmasks, cs_np, cbest = bms, bsc, _bk
                                    x1, y1, x2, y2 = bx1, by1, bx2, by2
                                    _big = True
                            if not _big:
                                _alt = [(_a, k) for _a, k in _hit
                                        if not _fills(cmasks[k])]
                                if _alt:
                                    cbest = min(_alt)[1]
                        cm = cmasks[cbest]
                        cys, cxs = np.nonzero(cm)
                        if len(cxs):
                            # 采用判据（2026-11-06 修正）：窗内结果包含
                            # 首锚点 且 面积显著小于全图掩膜（<50%）→
                            # 用户点的是窗内缺陷，采用窗内结果。原
                            # "窗内 bbox 占比<KEEP 保留"在纹路密集区
                            # 恒保留全图块（真机"单锚取全图主体"根因）
                            _carea = int(cm.sum())
                            _m0area = int(m0.sum())
                            _fx = min(max(int(points[0][0]) - x1, 0),
                                      (x2 - x1) - 1)
                            _fy = min(max(int(points[0][1]) - y1, 0),
                                      (y2 - y1) - 1)
                            _has_anchor = bool(cm[_fy, _fx])
                            _use2 = False
                            if _has_anchor and _carea < _m0area * 0.5 \
                                    and _big:
                                gmask = np.zeros((H, W), dtype=bool)
                                gmask[y1:y2, x1:x2] = cm
                                masks = gmask[None]
                                scores = np.asarray(
                                    [float(cs_np[cbest])
                                     if cs_np.size else 1.0])
                            elif _has_anchor and _carea < _m0area * 0.5:
                                # 二段递归 zoom（2026-11-06 小目标/低对比
                                # 优化）：窗内结果 bbox 仍占窗 >30% 时，
                                # 以首锚为中心再裁 128 窗重解一次
                                _cys0, _cxs0 = np.nonzero(cm)
                                _bw = float(_cxs0.max() - _cxs0.min()) \
                                    * float(_cys0.max() - _cys0.min())
                                _bw_max = max(
                                    float(_cxs0.max() - _cxs0.min()),
                                    float(_cys0.max() - _cys0.min()))
                                if _carea > 0.2 * (x2 - x1) * (y2 - y1) \
                                        and _bw_max < 0.7 * (x2 - x1) \
                                        and (x2 - x1) > 128:
                                    _h2 = 128
                                    _x1b = int(min(max(px - _h2 // 2, 0),
                                                   max(W - _h2, 0)))
                                    _y1b = int(min(max(py - _h2 // 2, 0),
                                                   max(H - _h2, 0)))
                                    _x2b, _y2b = _x1b + _h2, _y1b + _h2
                                    _crop_state(cpred, self.weights, seq,
                                                self._last_rgb, _x1b, _y1b,
                                                _x2b, _y2b)
                                    _pts2 = np.asarray(
                                        [[float(points[0][0]) - _x1b,
                                          float(points[0][1]) - _y1b]],
                                        dtype=np.float32)
                                    _ret2 = cpred.prompt_inference(
                                        cpred._xcv_im, points=_pts2,
                                        labels=np.array([1], dtype=np.int64),
                                        multimask_output=True)
                                    _mr, _sr = _ret2[0], _ret2[1]
                                    _s2 = _sr.detach().cpu().numpy() \
                                        if hasattr(_sr, "detach") \
                                        else np.asarray(_sr)
                                    _m2 = _mr.unsqueeze(1).float() \
                                        if _mr.dim() == 3 else _mr.float()
                                    _u2 = ops.scale_masks(
                                        _m2, (_h2, _h2), padding=False)
                                    if _u2.dim() == 4:
                                        _u2 = _u2[:, 0]
                                    _ms2 = (_u2.detach().cpu().numpy()
                                            if hasattr(_u2, "detach")
                                            else np.asarray(_u2)) > \
                                        _mask_thr(cpred)
                                    _fx2, _fy2 = _pts2[0][0], _pts2[0][1]
                                    _ih2, _iw2 = _ms2.shape[1], _ms2.shape[2]
                                    _hit2 = []
                                    for _k2 in range(_ms2.shape[0]):
                                        _mk2 = _ms2[_k2]
                                        if 0 <= _fy2 < _ih2 and \
                                                0 <= _fx2 < _iw2 and \
                                                bool(_mk2[int(_fy2),
                                                          int(_fx2)]):
                                            _hit2.append(
                                                (int(_mk2.sum()), _k2))
                                    if _hit2:
                                        _b2 = min(_hit2)[1]
                                        _m2f = _ms2[_b2]
                                        if int(_m2f.sum()) < _carea * 0.6:
                                            gmask = np.zeros((H, W),
                                                             dtype=bool)
                                            gmask[_y1b:_y2b,
                                                  _x1b:_x2b] = _m2f
                                            masks = gmask[None]
                                            scores = np.asarray(
                                                [float(_s2[_b2])
                                                 if _s2.size else 1.0])
                                            _use2 = True
                                if not _use2:
                                    gmask = np.zeros((H, W), dtype=bool)
                                    gmask[y1:y2, x1:x2] = cm
                                    masks = gmask[None]
                                    scores = np.asarray(
                                        [float(cs_np[cbest])
                                         if cs_np.size else 1.0])
        if masks.ndim == 3 and masks.shape[0] > 1 and scores.size > 1 and multimask:
            best = int(np.argmax(scores))
            if _pick_mode("sam") == "anchor":
                _py = int(min(max(float(points[0][1]), 0), masks.shape[1] - 1))
                _px = int(min(max(float(points[0][0]), 0), masks.shape[2] - 1))
                _hh = [(int(masks[k].sum()), k) for k in range(masks.shape[0])
                       if masks[k][_py, _px]]
                if _hh:
                    best = min(_hh)[1]
            return masks[best], float(scores[best])
        return (masks[0] if masks.ndim == 3 else masks,
                float(scores[0]) if scores.size else 1.0)

    def decode(self, embedding: ImageEmbedding, points, labels,
               multimask: bool = True) -> Annotation | None:
        """点提示解码 → 最佳掩码（原图尺寸 bool）。"""
        if points is None or len(points) == 0:   # P150：ndarray 禁真值判断
            return None
        try:
            import torch

            with torch.no_grad():
                mask, score = self._decode(points, labels, multimask,
                                           emb=embedding)
        except Exception as e:  # noqa: BLE001
            from loguru import logger

            logger.warning("SAM 解码失败：{}", e)
            return None
        bb = None
        ys, xs = np.nonzero(mask)
        if len(xs):
            bb = BoundingBox(ymin=float(ys.min()), xmin=float(xs.min()),
                             ymax=float(ys.max()), xmax=float(xs.max()))
        return Annotation(mask=np.asarray(mask, dtype=bool),
                          bounding_box=bb, score=round(float(score), 4))


def _resize_mask(mask: np.ndarray, H: int, W: int) -> np.ndarray:
    import cv2

    return cv2.resize(mask.astype(np.uint8), (W, H),
                      interpolation=cv2.INTER_NEAREST).astype(bool)


class _EfficientSamOnnx:
    """EfficientSAM onnx 引擎（编码一次缓存、解码毫秒级 onnxruntime；
    接口与 _UltralyticsSam 一致供 canvas 消费）。
    解码输出 (1,1,3,H,W) 三候选 + scores——按分数 argmax 取最佳。"""

    def __init__(self, name: str, encoder: str, decoder: str):
        import onnxruntime as _ort

        self.name = name
        self._encoder_path = encoder
        self._decoder_path = decoder
        self._sessions = None

    def _ensure(self):
        if self._sessions is None:
            import onnxruntime as _ort

            so = _ort.SessionOptions()
            so.graph_optimization_level = _ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            providers = _ort.get_available_providers()
            eps = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
                if "CUDAExecutionProvider" in providers else None
            self._sessions = {
                "encoder": _ort.InferenceSession(self._encoder_path, so,
                                                 providers=eps),
                "decoder": _ort.InferenceSession(self._decoder_path, so,
                                                 providers=eps),
            }
        return self._sessions

    def encode_image(self, image) -> ImageEmbedding:
        enc = self._ensure()["encoder"]
        img = np.asarray(image)
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[..., :3]
        batched = img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        emb = enc.run(output_names=None,
                      input_feed={"batched_images": batched})[0][0]
        return ImageEmbedding(weights=self.name,
                              key=emb.astype(np.float32).tobytes(),
                              meta={
                                  "embedding_shape": emb.shape,
                                  "original_height": int(img.shape[0]),
                                  "original_width": int(img.shape[1]),
                              })

    def decode(self, embedding, points, labels,
               multimask: bool = True) -> Annotation | None:
        if points is None or len(points) == 0:
            return None
        dec = self._ensure()["decoder"]
        pts = np.asarray(points, dtype=np.float32)[None, None, :, :]
        labs = np.asarray(labels, dtype=np.float32)[None, None, :]
        emb = np.frombuffer(embedding.key, dtype=np.float32).reshape(
            1, *tuple(embedding.meta.get("embedding_shape", ())))
        meta = getattr(embedding, "meta", None) or {}
        oh = float(meta.get("original_height", emb.shape[-2] * 16))
        ow = float(meta.get("original_width", emb.shape[-1] * 16))
        try:
            masks, scores, _ = dec.run(None, {
                "image_embeddings": emb.astype(np.float32),
                "batched_point_coords": pts,
                "batched_point_labels": labs,
                "orig_im_size": np.array((oh, ow), dtype=np.int64),
            })
        except Exception as e:  # noqa: BLE001
            from loguru import logger
            logger.warning("EfficientSAM 解码失败：{}", e)
            return None
        # 候选选择（避免恒取首个=「取全图主体」）：在包含
        # 首锚点的候选中取面积最小——交互标注"点哪标哪"
        _t = _PARAMS.get("mask_threshold")
        cand = masks[0, 0] > (0.0 if _t is None else float(_t))  # (3, H, W)
        py = int(min(max(float(points[0][1]), 0), cand.shape[1] - 1))
        px = int(min(max(float(points[0][0]), 0), cand.shape[2] - 1))
        containing = [(int(cand[k, py, px]), int(cand[k].sum()), k)
                      for k in range(cand.shape[0])]
        hit = [c for c in containing if c[0]]
        best = min(hit, key=lambda c: c[1])[2] if hit else \
            int(np.argmax(scores[0, 0]))
        if _pick_mode("efficientsam") == "score":
            best = int(np.argmax(scores[0, 0]))
        mask = cand[best]
        if mask.shape != (int(oh), int(ow)):
            import cv2 as _cv
            mask = _cv.resize(mask.astype(np.uint8),
                              (int(ow), int(oh)),
                              interpolation=_cv.INTER_NEAREST).astype(bool)
        ys, xs = np.nonzero(mask)
        bb = None
        if len(xs):
            bb = BoundingBox(ymin=float(ys.min()), xmin=float(xs.min()),
                             ymax=float(ys.max()), xmax=float(xs.max()))
        return Annotation(mask=np.asarray(mask, dtype=bool),
                          bounding_box=bb, score=1.0)

    def _decode(self, points, labels, multimask, emb=None):
        ann = self.decode(emb, points, labels, multimask)
        return (ann.mask, 1.0) if ann is not None else (None, 0.0)


def _engine_by_name(name: str):
    for m in _MODELS:
        if m["name"] == name:
            if m.get("predictor") == "efficientsam":
                pe = _WEIGHTS_DIR / m["file"]
                pd = _WEIGHTS_DIR / (m.get("file2") or "")
                if pe.is_file() and pd.is_file():
                    return _EfficientSamOnnx(name=m["name"],
                                             encoder=str(pe), decoder=str(pd))
                return None
            p = _WEIGHTS_DIR / m["file"]
            if p.is_file():
                return _UltralyticsSam(name=m["name"], weights=str(p))
            return None
    return None


def get_model_type_by_name(name: str):
    """canvas.initializeAiModel：`osam.apis.get_model_type_by_name(name)()`。"""
    eng = _engine_by_name(name)
    if eng is None:
        raise ValueError(
            f"AI 模型「{name}」不可用：权重缺失。请将权重放到 {weights_dir()} 下"
            "（sam2.1_t.pt / mobile_sam.pt），或换用已就绪的档位。")
    if hasattr(eng, "warmup"):
        try:
            eng.warmup()
        except Exception as e:  # noqa: BLE001
            from loguru import logger

            logger.warning("SAM 预热失败（不影响使用）：{}", e)
    return lambda: eng


def decode_points(name: str, image_bgr: np.ndarray, points, labels,
                  multimask: bool = True):
    """便捷入口（T5-4 批量预标注/测试用）：一次编码 + 一次解码，返回 Annotation。"""
    eng = _engine_by_name(name)
    if eng is None:
        raise ValueError(f"AI 模型「{name}」不可用（权重缺失，见 available_models）")
    emb = eng.encode_image(image_bgr[..., ::-1] if image_bgr.ndim == 3 else image_bgr)
    emb.meta["orig_hw"] = (int(image_bgr.shape[0]), int(image_bgr.shape[1]))
    return eng.decode(emb, np.asarray(points, dtype=float),
                      list(labels), multimask=multimask)


def generate(request: GenerateRequest) -> GenerateResponse:
    """canvas._update_shape_with_sam：`osam.apis.generate(request)`。"""
    eng = _engine_by_name(request.model)
    if eng is None:
        raise ValueError(f"AI 模型「{request.model}」不可用（权重缺失）")
    emb = request.image_embedding
    if emb is None and request.image is not None:
        emb = eng.encode_image(request.image)
    if emb is None:
        raise ValueError("generate 需要 image_embedding 或 image")
    ann = eng.decode(emb, np.asarray(request.prompt.points, dtype=float),
                     list(request.prompt.point_labels or [1] * len(request.prompt.points)))
    return GenerateResponse(annotations=[ann] if ann is not None else [])


def decode_best(name: str, image_rgb: np.ndarray, points, labels) -> np.ndarray | None:
    """canvas 悬停预览用：返回最佳掩码（H, W）bool 或 None。"""
    eng = _engine_by_name(name)
    if eng is None:
        return None
    emb = eng.encode_image(image_rgb)
    emb.meta["orig_hw"] = (int(image_rgb.shape[0]), int(image_rgb.shape[1]))
    ann = eng.decode(emb, np.asarray(points, dtype=float), list(labels))
    return None if ann is None else ann.mask
