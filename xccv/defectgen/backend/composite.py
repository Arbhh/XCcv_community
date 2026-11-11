"""Defect Augment Studio — 全分辨率确定性合成引擎。

供 /api/composite（单张导出）、/api/batch（背景池批量）、/api/poisson_preview（编辑器内
泊松预览）复用。实例几何链与前端渲染 Worker 严格一致：

    源像素 s --网格FFD(+d)--> u --透视H--> p --仿射A--> 目标坐标 t

渲染采用逆映射：对每个目标像素依次做 A⁻¹ → H⁻¹ → (u - d(u)) 得到源坐标，
单次 cv2.remap 采样，避免多级 warp 的累加插值损失。
"""

from __future__ import annotations

import base64
import math

import cv2
import numpy as np


# ---------------------------------------------------------------- 基础编解码

def b64_to_img(s: str):
    arr = np.frombuffer(base64.b64decode(s), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)


def img_to_b64(img, ext: str = ".png") -> str:
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError("图像编码失败: " + ext)
    return base64.b64encode(buf.tobytes()).decode("ascii")


def to_bgra(img) -> np.ndarray:
    if img is None:
        raise ValueError("图像解码失败")
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    if img.shape[2] == 4:
        return img
    raise ValueError("不支持的通道数: %s" % img.shape[2])


# ---------------------------------------------------------------- 几何

def affine_matrix_3x3(inst: dict, w: int, h: int) -> np.ndarray:
    """仿射 A：缩放(sx,sy) → 错切(shx,shy) → 旋转，绕资产中心，平移到 position。"""
    a = inst.get("affine") or {}
    sx = float(a.get("scale_x", 1.0))
    sy = float(a.get("scale_y", 1.0))
    rot = math.radians(float(a.get("rotation", 0.0)))
    shx = float(a.get("shear_x", 0.0))
    shy = float(a.get("shear_y", 0.0))
    pos = inst.get("position") or [0.0, 0.0]

    S = np.array([[sx, 0.0], [0.0, sy]])
    SH = np.array([[1.0, shx], [shy, 1.0]])
    R = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
    M = R @ SH @ S
    c = np.array([w / 2.0, h / 2.0])
    t = np.asarray(pos, np.float64) - M @ c
    return np.array([[M[0, 0], M[0, 1], t[0]],
                     [M[1, 0], M[1, 1], t[1]],
                     [0.0, 0.0, 1.0]])


def make_disp_fn(mesh: dict | None, w: int, h: int):
    """网格 FFD 位移场 d(s)：网格顶点位移的双线性插值。顶点 (i,j) 位于 (i*cellx, j*celly)。"""
    if not mesh:
        return None
    cols = int(mesh["cols"])
    rows = int(mesh["rows"])
    grid = np.asarray(mesh["grid"], np.float64).reshape(rows + 1, cols + 1, 2)
    cellx = w / cols
    celly = h / rows

    def d(xs: np.ndarray, ys: np.ndarray):
        gx = np.clip(xs / cellx, 0.0, float(cols))
        gy = np.clip(ys / celly, 0.0, float(rows))
        i0 = np.minimum(np.floor(gx).astype(np.int64), cols - 1)
        j0 = np.minimum(np.floor(gy).astype(np.int64), rows - 1)
        i1 = i0 + 1
        j1 = j0 + 1
        fx = (gx - i0)[..., None]
        fy = (gy - j0)[..., None]
        v = (grid[j0, i0] * (1 - fx) * (1 - fy) + grid[j0, i1] * fx * (1 - fy)
             + grid[j1, i0] * (1 - fx) * fy + grid[j1, i1] * fx * fy)
        return v[..., 0], v[..., 1]

    return d


def _safe_inv(m: np.ndarray) -> np.ndarray | None:
    try:
        return np.linalg.inv(m)
    except np.linalg.LinAlgError:
        return None


def render_instance_layer(tw: int, th: int, inst: dict, rgba: np.ndarray):
    """渲染单个实例 → (layer BGRA [bh,bw,4], origin(x,y))；完全超出画布时返回 (None, None)。"""
    h, w = rgba.shape[:2]
    A = affine_matrix_3x3(inst, w, h)
    Ainv = _safe_inv(A)
    H = None
    Hinv = None
    ph = inst.get("perspective_h")
    if ph is not None:
        H = np.asarray(ph, np.float64).reshape(3, 3)
        Hinv = _safe_inv(H)
        if Hinv is None:
            H = Hinv = None  # 奇异透视矩阵 → 忽略透视
    dfn = make_disp_fn(inst.get("mesh"), w, h)

    def forward(pt):
        s = np.array(pt, np.float64)
        if dfn is not None:
            dx, dy = dfn(s[0:1], s[1:2])
            s = s + np.array([dx[0], dy[0]])
        p = s
        if H is not None:
            q = H @ np.array([s[0], s[1], 1.0])
            p = q[:2] / q[2]
        return A[:2, :2] @ p + A[:2, 2]

    corners = [(0, 0), (w, 0), (w, h), (0, h)]
    ts = np.array([forward(c) for c in corners])
    eb_pad = 3.0 * max(0.0, float(inst.get("edge_blur", 0) or 0))
    x0 = max(0, int(math.floor(ts[:, 0].min())) - 4 - int(math.ceil(eb_pad)))
    y0 = max(0, int(math.floor(ts[:, 1].min())) - 4 - int(math.ceil(eb_pad)))
    x1 = min(tw, int(math.ceil(ts[:, 0].max())) + 4 + int(math.ceil(eb_pad)))
    y1 = min(th, int(math.ceil(ts[:, 1].max())) + 4 + int(math.ceil(eb_pad)))
    if x1 <= x0 or y1 <= y0:
        return None, None

    xs, ys = np.meshgrid(np.arange(x0, x1, dtype=np.float64),
                         np.arange(y0, y1, dtype=np.float64))
    ones = np.ones_like(xs)
    p = Ainv @ np.stack([xs, ys, ones]).reshape(3, -1)
    p /= p[2:3, :]
    if Hinv is not None:
        u = Hinv @ p
        u /= u[2:3, :]
    else:
        u = p
    bh, bw = ys.shape
    if dfn is not None:
        dx, dy = dfn(u[0], u[1])
        sx = (u[0] - dx).reshape(bh, bw)
        sy = (u[1] - dy).reshape(bh, bw)
    else:
        sx = u[0].reshape(bh, bw)
        sy = u[1].reshape(bh, bw)

    # 预乘采样：资产在 alpha=0 处的 RGB 为 0（Canvas 合成/导出的必然结果），
    # 若对 BGRA 逐通道独立插值，半透明边缘会掺进黑色 → 放置/导出后出现黑边。
    # Canvas 内部按预乘插值，所以预览没有黑边——这里保持同样语义：
    # 先乘 alpha 再插值，最后反预乘还原 unpremultiplied RGB。
    src = rgba.astype(np.float32)
    a0 = (src[:, :, 3:4] / 255.0)
    pm = np.concatenate([src[:, :, :3] * a0, src[:, :, 3:4]], axis=2)
    out = cv2.remap(pm, sx.astype(np.float32), sy.astype(np.float32),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0, 0))
    oa = np.clip(out[:, :, 3:4], 0, 255) / 255.0
    safe = np.maximum(oa, 1e-4)
    rgb = np.where(oa > 1e-3, out[:, :, :3] / safe, 0.0)
    layer = np.dstack([np.clip(rgb, 0, 255), np.clip(out[:, :, 3:4], 0, 255)]).astype(np.uint8)
    return layer, (x0, y0)


# ---------------------------------------------------------------- 外观增强

def apply_appearance(layer: np.ndarray, inst: dict, rng: np.random.Generator):
    """外观增强仅作用于 mask 内有效像素（alpha 加权），不污染边缘外背景。"""
    app = inst.get("appearance") or {}
    b = float(app.get("brightness", 0.0))
    c = float(app.get("contrast", 1.0))
    hue = float(app.get("hue", 0.0))
    sat = float(app.get("saturation", 1.0))
    blur = float(app.get("blur", 0.0))
    noise = float(app.get("noise", 0.0))
    if abs(b) < 1e-6 and abs(c - 1) < 1e-6 and abs(hue) < 1e-6 \
            and abs(sat - 1) < 1e-6 and blur < 1e-6 and noise < 1e-6:
        return

    rgb = layer[:, :, :3].astype(np.float32)
    if abs(b) >= 1e-6:
        rgb += b
    if abs(c - 1) >= 1e-6:
        rgb = (rgb - 128.0) * c + 128.0
    if abs(hue) >= 1e-6 or abs(sat - 1) >= 1e-6:
        hsv = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = np.mod(hsv[:, :, 0] + hue / 2.0, 180.0)  # cv2 色相 0..179 对应 0..360°
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat, 0, 255)
        rgb = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    if blur >= 1e-6:
        rgb = cv2.GaussianBlur(rgb, (0, 0), blur)
    if noise >= 1e-6:
        alpha = layer[:, :, 3:4].astype(np.float32) / 255.0
        rgb += rng.normal(0.0, noise, rgb.shape).astype(np.float32) * alpha
    layer[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- 融合

def apply_edge_blur(layer: np.ndarray, inst: dict):
    """边缘模糊（v1.4）：RGBA 以 premultiplied 方式整体高斯模糊——缺陷轮廓的
    颜色与 alpha 一起向背景渐隐，消除贴图硬边。与外观模糊（仅 mask 内内容）
    互补；σ = edge_blur px，与前端 Worker 预览语义一致。"""
    eb = float(inst.get("edge_blur", 0) or 0)
    if eb <= 1e-3:
        return
    a = layer[:, :, 3].astype(np.float32) / 255.0
    rgb_p = layer[:, :, :3].astype(np.float32) * a[..., None]   # premultiply 防黑边
    rgb_pb = cv2.GaussianBlur(rgb_p, (0, 0), eb, borderType=cv2.BORDER_CONSTANT)
    ab = cv2.GaussianBlur(a, (0, 0), eb, borderType=cv2.BORDER_CONSTANT)
    safe = np.maximum(ab, 1e-4)
    rgb_out = np.where(ab[..., None] > 1e-3, rgb_pb / safe[..., None], 0.0)
    layer[:, :, :3] = np.clip(rgb_out + 0.5, 0, 255).astype(np.uint8)
    layer[:, :, 3] = np.clip(ab * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _feather_layers(layer: np.ndarray, radius: float):
    """羽化：与画布预览完全一致的 `destination-in` 语义。

    画布做法：先画实例层，再用"模糊后的 alpha"做 destination-in ——
    最终 alpha = a0 × blur(a0)。这一点很关键：
      · 形状外 a0=0 → 结果 alpha 恒为 0，颜色不会外溢（既无黑边也无光晕、
        无需做颜色延展，之前的"迭代膨胀延展"会留下放射状条纹伪影）；
      · 形状内 alpha 被平滑收窄，得到柔和边缘；
      · RGB 保持原样（内部材质不模糊）。
    后端按同一公式实现，保证预览 == 导出。
    """
    a0 = layer[:, :, 3].astype(np.float32) / 255.0
    rgb = layer[:, :, :3].astype(np.float32)
    if radius <= 1e-3:
        return a0, rgb
    k = max(1.0, float(radius))
    ab = cv2.GaussianBlur(a0, (0, 0), k)
    return a0 * ab, rgb


def blend_direct_or_feather(canvas: np.ndarray, layer: np.ndarray, inst: dict):
    mode = inst.get("blend", "direct")
    radius = float(inst.get("feather_radius", 0))
    a, rgb = _feather_layers(layer, radius if mode == "feather" else 0.0)
    a = a * float(inst.get("opacity", 1.0))
    a3 = a[..., None]
    origin_x, origin_y = inst["_origin"]
    roi = canvas[origin_y:origin_y + a.shape[0], origin_x:origin_x + a.shape[1]]
    canvas[origin_y:origin_y + a.shape[0], origin_x:origin_x + a.shape[1]] = roi * (1 - a3) + rgb * a3


def blend_poisson(canvas_u8: np.ndarray, layer: np.ndarray, inst: dict, notes: list) -> bool:
    """seamlessClone。失败/过小/出界时回退 Feather 并记录原因（降级链）。"""
    x0, y0 = inst["_origin"]
    bh, bw = layer.shape[:2]
    th, tw = canvas_u8.shape[:2]
    if x0 < 0 or y0 < 0 or x0 + bw > tw or y0 + bh > th:
        notes.append("Poisson→Feather：实例区域超出目标图像边界")
        return False
    m_roi = ((layer[:, :, 3] >= 60).astype(np.uint8)) * 255
    m_roi = cv2.dilate(m_roi, np.ones((9, 9), np.uint8))  # 外扩 4px：给求解器保留缺陷自身梯度
    if int((m_roi > 0).sum()) < 25:
        notes.append("Poisson→Feather：缺陷小于 5×5 px，seamlessClone 不适用")
        return False
    m = np.zeros_like(canvas_u8[:, :, 0])          # seamlessClone 要求 mask 与 src 同尺寸
    m[y0:y0 + bh, x0:x0 + bw] = m_roi
    src = canvas_u8.copy()
    a3 = (layer[:, :, 3:4].astype(np.float32) / 255.0)
    roi = src[y0:y0 + bh, x0:x0 + bw].astype(np.float32)
    src[y0:y0 + bh, x0:x0 + bw] = (roi * (1 - a3) + layer[:, :, :3].astype(np.float32) * a3).astype(np.uint8)
    center = (int(x0 + bw / 2), int(y0 + bh / 2))
    flag = cv2.MIXED_CLONE if inst.get("blend_flag") == "mixed" else cv2.NORMAL_CLONE
    try:
        out = cv2.seamlessClone(src, canvas_u8, m, center, flag)
        changed = int((cv2.absdiff(out, canvas_u8).sum(-1) > 2).sum())
        if changed == 0:
            notes.append("Poisson→Feather：融合结果无变化，已回退")
            return False
        canvas_u8[:] = out
        return True
    except cv2.error as e:
        notes.append("Poisson→Feather：seamlessClone 失败(%s)" % str(e).splitlines()[0][:80])
        return False


# ---------------------------------------------------------------- 程序化缺陷绘制

def _dent_crater_warp(base: np.ndarray, mask: np.ndarray, kind: str,
                      warp: dict | None = None, tint_bgr=None,
                      strength: float = 0.7, seed: int = 0):
    """多缺陷拆分入口。

    凹坑的深色核心、明暗相位、旋涡方向都是"每坑一套"——若把整张掩码当单一
    缺陷处理，距离场 argmax 只落在一个坑上，同组其余凹坑会失去核心与明暗
    （绘制模式导出时"只有第一个缺陷"的真正原因）。此处按连通域逐个成像再
    融合；单个连通域（含划伤）直接走原路径。
    """
    core = (mask > 40).astype(np.uint8)
    if kind != "dent" or int(core.sum()) < 4:
        return _dent_one(base, mask, kind, warp, tint_bgr, strength, seed)
    n, lbl = cv2.connectedComponents(core)
    if n <= 2:                       # 0=背景，1=唯一的坑
        return _dent_one(base, mask, kind, warp, tint_bgr, strength, seed)
    wf = warp or {}
    rad = float(np.clip(wf.get("radius", 1.0), 0.2, 3.0))
    out = base.copy()
    alpha = mask.copy()
    for i in range(1, n):
        comp = (lbl == i)
        if int(comp.sum()) < 4:
            continue
        m_i = np.where(comp, mask, 0).astype(np.uint8)
        if int(m_i.max()) < 8:
            continue
        r_i = float(cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 5).max())
        pad = int(max(24, r_i * (1.6 * rad + 1.4) + 12))
        ys0, xs0 = np.nonzero(comp)
        y0 = max(0, int(ys0.min()) - pad); y1 = min(mask.shape[0], int(ys0.max()) + pad + 1)
        x0 = max(0, int(xs0.min()) - pad); x1 = min(mask.shape[1], int(xs0.max()) + pad + 1)
        o_c, a_c = _dent_one(base[y0:y1, x0:x1], m_i[y0:y1, x0:x1], kind, warp,
                             tint_bgr, strength, seed + 7919 * i)
        w = (a_c.astype(np.float32) / 255.0)[..., None]
        region = out[y0:y1, x0:x1].astype(np.float32)
        out[y0:y1, x0:x1] = np.clip(region * (1.0 - w) + o_c.astype(np.float32) * w,
                                    0, 255).astype(np.uint8)
        alpha[y0:y1, x0:x1] = np.maximum(alpha[y0:y1, x0:x1], a_c)
    return out, alpha


def _dent_one(base: np.ndarray, mask: np.ndarray, kind: str,
              warp: dict | None = None, tint_bgr=None,
              strength: float = 0.7, seed: int = 0):
    """凹坑/划伤形貌合成（两套独立模型）。

    凹坑（dent）——径向内陷"酒窝"模型，自真实模板（test/模板.jpg 黑圈处）逆向：
      1. 条纹四周对称包绕（上拱/下弯/左右端头吸入坑缘），形变延伸至坑外 ~1.5R；
      2. 坑底冲蚀为光滑灰面，带凹面明暗（光向随种子 ±40° 偏转，强度随噪声起伏）；
      3. 中心为**不规则椭圆深色核心区**（随机朝向/长轴比 1.3~2.0/噪声边界，
         65% 概率叠加二级子斑），与外围明暗平滑融合，无分界线。
      明暗/弯移/核心边界均以种子化低频噪声调制（真实坑壁非理想碗面）。

    划伤（scratch）——沟槽模型（独立实现）：亮核暗晕由 paint_defect_layer 的
    delta 剖面承担，此处只做**横向挤压**形变：沿划痕线把两侧条纹轻微压向沟内
    （幅度 ≈ 0.35·depth，封顶 0.85·内切半径），强度沿走向随机起伏（时断时续）；
    不冲蚀沟底、无深点/凹面明暗——细长刻痕不应出现"包绕眼"。

    实现：平滑带符号距离场 S 的负梯度 = 指向脊线的内陷方向（任意形状自适应）；
    幅度为坑缘峰值的高斯环 × 内部门控（脊线处归零，防方向翻转"扇开"断裂）。
    warp: {depth 脊缘最大弯移 px（缺省 0.6·r_in）, radius 影响带尺度×r_in,
           falloff 衰减指数（越大影响带越紧，2.0 为基准）}。返回 (BGR patch, alpha_ext)。
    """
    h_img, w_img = mask.shape[:2]
    core = (mask > 40).astype(np.uint8)
    if int(core.sum()) < 4:
        return base, mask
    d_in = cv2.distanceTransform(core, cv2.DIST_L2, 5)
    r_in = float(d_in.max())
    if r_in < 2.0:
        return base, mask
    wf = warp or {}
    A = float(np.clip(wf.get("depth", 0.6 * r_in), 0.0, 400.0))
    R_h = float(np.clip(wf.get("radius", 1.0) * r_in, 3.0, min(h_img, w_img) * 0.6))
    fall = float(np.clip(wf.get("falloff", 2.0), 0.5, 4.0))      # 衰减指数：越大影响带越紧
    sig = max(2.0, R_h * 0.55 * (2.0 / fall))                    # 影响带宽度（坑外~1.5R 衰减）
    core_scale = float(np.clip(wf.get("core", 0.35), 0.10, 0.70))  # 深色核心大小×r_in
    shade_amp = float(np.clip(wf.get("shade", 0.20), 0.0, 0.50))   # 凹面明暗对比
    swirl_user = float(np.clip(wf.get("swirl", 0.5), 0.0, 1.0))    # 外围旋涡强度

    d_out = cv2.distanceTransform(1 - core, cv2.DIST_L2, 5)
    s = np.where(d_in > 0, -d_in, d_out).astype(np.float32)      # 带符号距离
    S = cv2.GaussianBlur(s, (0, 0), max(2.0, r_in * 0.35))       # 同心等值线场

    # 种子化低频噪声（确定性）：真实坑壁并非理想碗面，明暗/弯移带不规则起伏
    rng = np.random.default_rng((seed & 0x7FFFFFFF) ^ 0x5EED1)

    def _noise(sig_n: float) -> np.ndarray:
        n = rng.normal(0, 1, (h_img, w_img)).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), max(1.5, sig_n))
        return n / (float(np.abs(n).max()) + 1e-6)

    n_amp = _noise(r_in * 0.7)      # 弯移不对称（低频）
    n_shade = _noise(r_in * 0.35)   # 明暗沿壁面强弱（中频）
    ys, xs = np.mgrid[0:h_img, 0:w_img].astype(np.float32)

    gx = cv2.Sobel(S, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(S, cv2.CV_32F, 0, 1, ksize=5)
    gn = np.sqrt(gx * gx + gy * gy) + 1e-6
    ux, uy = -gx / gn, -gy / gn                                  # 指向脊线（坑心/划痕线）

    if kind == "scratch":
        # ── 划伤：沟槽模型（横向挤压，时断时续，不冲蚀沟底）──
        A_s = float(np.clip(A * 0.35, 1.0, 0.85 * r_in + 2.0))
        n_dir = _noise(r_in * 1.6)                               # 沿走向起伏
        wamp = np.exp(-((s / max(2.0, r_in * 1.3)) ** 2)).astype(np.float32)
        inner = np.clip((-s) / (0.6 * r_in), 0.0, 1.0)           # 脊线处归零（防扇开）
        amp = A_s * wamp * (1.0 - inner) * (0.55 + 0.45 * np.abs(n_dir))
        warped = cv2.remap(base, (xs + ux * amp).astype(np.float32),
                           (ys + uy * amp).astype(np.float32), cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT).astype(np.float32)
        wsoft = np.clip(wamp * 1.2, 0.0, 1.0)[..., None]
        out = warped * (1.0 - wsoft) + cv2.GaussianBlur(warped, (0, 0), 0.8) * wsoft
        floor_w = mask.astype(np.float32) / 255.0                # tint 只在沟内生效
    else:
        # ── 凹坑：径向内陷酒窝 ──
        wamp = np.exp(-((s / sig) ** 2)).astype(np.float32)      # 坑缘峰值环带
        # 内部门控：脊线处幅度归零（方向翻转会导致采样"扇开"出相位断裂线），
        # 坑缘满幅——包绕集中在坑缘及外部，坑底内容保持连续
        inner = np.clip((-s) / (0.25 * r_in), 0.0, 1.0)
        amp = A * wamp * (1.0 - inner) * (1.0 + 0.16 * n_amp)
        # 螺旋分量：只取外围弯移的一部分比例叠加切向旋转（避免纯径向的僵硬），
        # 且集中在上下扇区（水平条纹图上左右侧的竖向搓动会非常突兀）——
        # 上方暗带/下方亮带的包绕弧向坑心卷入、互相接触（太极环抱的位移侧）。
        # 比例上限 + 随机基准 + 逐位置噪声调制（位置不同比例不同），随同一
        # 环带包络向外衰减过渡；旋向随种子，并与明暗卷入方向一致。
        cy0, cx0 = np.unravel_index(int(np.argmax(d_in)), d_in.shape)
        swirl_dir = 1.0 if rng.uniform() < 0.5 else -1.0
        swirl_k = 0.35 * swirl_user * (0.7 + 0.6 * rng.uniform())   # 旋涡比例（面板可调）
        n_sw = _noise(r_in * 0.9)                             # 逐位置强弱
        dy0 = ys - cy0
        dx0 = xs - cx0
        sw_w = np.abs(dy0) / (np.sqrt(dx0 * dx0 + dy0 * dy0) + 1e-6)   # 上下 1、左右 0
        sw_w = sw_w ** 1.6
        sw_field = np.clip(swirl_k * sw_w * (0.4 + 0.6 * np.abs(n_sw)), 0.0, 0.45)
        vx = ux - swirl_dir * uy * sw_field
        vy = uy + swirl_dir * ux * sw_field
        vn = np.sqrt(vx * vx + vy * vy) + 1e-6
        warped = cv2.remap(base, (xs + (vx / vn * amp)).astype(np.float32),
                           (ys + (vy / vn * amp)).astype(np.float32), cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT).astype(np.float32)

        # 弯移区轻微柔化（真实压缩条纹端头略糊）
        wsoft = np.clip(wamp * 1.2, 0.0, 1.0)[..., None]
        warped = warped * (1.0 - wsoft) + cv2.GaussianBlur(warped, (0, 0), 0.8) * wsoft

        # 坑底接管：条纹冲蚀成光滑灰面（微留条纹痕）
        floor_w = np.clip((-S - 0.10 * r_in) / (0.30 * r_in), 0.0, 1.0)
        floor = cv2.GaussianBlur(warped, (0, 0), max(4.0, r_in * 0.5))

        # 不规则椭圆深色核心区（凹坑最深处）：真实凹坑底部常为一整块不规则深色
        # 区域（椭圆居多），而非对称小圆点——随机朝向/长轴比/噪声起伏边界，
        # 65% 概率叠加二级子斑；核心与外围明暗平滑融合（无"眼睛"分界线）
        st_cl = float(np.clip(strength, 0.0, 1.0))

        def _dark_core(c_x: float, c_y: float, base_r: float, elong: float, ang: float):
            dx0 = xs - c_x
            dy0 = ys - c_y
            ca, sa = math.cos(ang), math.sin(ang)
            ru = dx0 * ca + dy0 * sa
            rv = -dx0 * sa + dy0 * ca
            r_e = np.sqrt((ru / elong) ** 2 + rv ** 2)
            n_edge = _noise(r_in * 0.13)
            r_c = np.maximum(base_r * (1.0 + 0.40 * n_edge), 1.5)
            core0 = np.clip(1.0 - r_e / r_c, 0.0, 1.0) ** 1.25
            return core0 * (1.0 - 0.22 * np.abs(n_edge))       # 核内明暗斑驳

        core = _dark_core(float(cx0), float(cy0),
                          r_in * core_scale * (0.85 + 0.30 * rng.uniform()),
                          rng.uniform(1.3, 2.0),
                          rng.uniform(0.0, math.pi)) * floor_w
        out_core = (46.0 + 38.0 * st_cl) * core
        if rng.uniform() < 0.65:
            ang2 = rng.uniform(0.0, math.pi)
            off = rng.uniform(0.15, 0.30) * r_in
            core2 = _dark_core(float(cx0) + math.cos(ang2) * off,
                               float(cy0) + math.sin(ang2) * off,
                               r_in * rng.uniform(0.14, 0.22), rng.uniform(1.2, 2.2),
                               ang2) * floor_w
            out_core = out_core + (30.0 + 24.0 * st_cl) * core2

        # 凹面明暗——太极式交融环抱：暗带/亮带不是对顶两瓣，而是随深度旋转
        # 卷入坑心（相位 φ = φ0 + 旋向·(1-dn)·旋量），如阴阳鱼互相环抱——
        # 卷入量取 2.0~3.0 rad，上方暗带卷入后触及下方亮月牙；旋向与位移涡旋
        # 一致。幅度 ∝ (1-dn)（缘部最陡、坑底平坦，避免平台区直边分界），
        # 强弱乘噪声，在深色核心区淡出。φ0/旋向/旋量全部随种子变化。
        light_ang = rng.uniform(-0.7, 0.7)                 # 起始光向（相对竖直 ±40°）
        swirl_amt = rng.uniform(2.0, 3.0)                  # 卷入总旋量（弧度）
        dn = d_in / max(r_in, 1.0)
        wall = np.clip(1.0 - dn, 0.0, 1.0)
        theta = np.arctan2(ys - cy0, xs - cx0)
        phi = light_ang + swirl_dir * (1.0 - dn) * swirl_amt
        lobe = np.cos(theta - phi)                          # ±1 交变，无直线分界
        shade = (1.0 + shade_amp * lobe * wall
                 * (1.0 + 0.55 * n_shade) * (1.0 - 0.85 * core))[..., None]
        out = warped * (1.0 - floor_w[..., None]) + floor * shade * floor_w[..., None]
        out = out - out_core[..., None]
    if tint_bgr is not None:
        # 用户色只在坑底混入（保留凹面明暗的立体感）
        k = (0.30 * float(np.clip(strength, 0.0, 1.0)) * floor_w)[..., None]
        tint = np.array([float(tint_bgr[0]), float(tint_bgr[1]), float(tint_bgr[2])], np.float32)
        out = out * (1.0 - k) + tint * k
    out = np.clip(out, 0, 255).astype(np.uint8)

    # alpha：掩码 ∪ 弯移环带（随资产带走的形貌区）
    band = (amp > 0.6).astype(np.uint8) * 255
    band = cv2.dilate(band, np.ones((3, 3), np.uint8))
    band = cv2.GaussianBlur(band, (0, 0), 1.2)
    alpha = np.maximum(mask.astype(np.float32), band.astype(np.float32))
    alpha = np.clip(alpha, 0, 255).astype(np.uint8)
    return out, alpha


def paint_defect_layer(base_bgr: np.ndarray, mask: np.ndarray, kind: str,
                       strength: float, grain: float, seed: int,
                       tint_bgr=None, warp: dict | None = None) -> np.ndarray:
    """把用户笔迹掩码着色为缺陷形貌层（RGBA）。缺陷颜色继承底层纹理，
    按类型叠加明暗剖面：白点/黑点/划伤/污渍/凹坑。确定性（seed）。

    mask: uint8 (0..255)，alpha 即笔迹软硬；base_bgr 与 mask 同尺寸。
    tint_bgr: 自定义颜色（BGR 三元组）——按距离变换剖面把笔迹向该颜色混合
    （v1.4 三色盘/吸管），同时保留类型形貌附加（划伤暗晕/凹坑高光环）。
    warp: 凹坑/划伤内陷扭曲参数 {strength, falloff, radius}（面板可调）。
    """
    h, w = mask.shape[:2]
    alpha = mask.astype(np.float32) / 255.0
    rng = np.random.default_rng(seed & 0x7FFFFFFF)
    base = base_bgr.astype(np.float32)

    # 距离变换剖面：核心 1 → 边缘 0（笔画软硬已含在 alpha 中）
    dist = cv2.distanceTransform((mask > 40).astype(np.uint8), cv2.DIST_L2, 5)
    prof = np.zeros((h, w), np.float32)
    mmax = float(dist.max())
    if mmax > 1e-3:
        prof = np.clip(dist / mmax, 0, 1)
    profile = prof * alpha                       # 软边缘同时受 alpha 调制
    strength = float(np.clip(strength, 0.0, 1.0))
    grain = float(np.clip(grain, 0.0, 1.0))

    delta = np.zeros((h, w, 3), np.float32)
    if kind == "scratch":
        # 沿走向时断时续（真实划伤亮暗不均）：种子化低频噪声调制亮核与暗晕
        n_al = rng.normal(0, 1, (h, w)).astype(np.float32)
        n_al = cv2.GaussianBlur(n_al, (0, 0), max(4.0, mmax * 2.5))
        n_al = n_al / (float(np.abs(n_al).max()) + 1e-6)
    if tint_bgr is not None:
        # 自定义颜色：向目标色混合（核心强度高、边缘渐隐），保留底层纹理。
        # 凹坑例外：颜色在坑底由 _dent_crater_warp 按凹面明暗混入（避免整块
        # 半透明色斑盖掉条纹包绕/明暗立体感）
        tint = np.array([float(tint_bgr[0]), float(tint_bgr[1]), float(tint_bgr[2])], np.float32)
        if kind != "dent":
            wgt = np.clip(strength * 1.15 * (profile ** 0.85), 0.0, 1.0)[..., None]
            delta = (tint - base) * wgt
            if kind == "scratch":
                # 划伤形貌：微暗晕圈保持"刻痕"感（随走向起伏）
                halo = cv2.dilate(alpha, np.ones((5, 5), np.uint8)) - alpha
                delta -= 70.0 * strength * (halo * (0.62 + 0.38 * n_al))[..., None]
            elif kind == "stain":
                # 污渍形貌：低频明暗起伏（保留用户色相）
                n = rng.normal(0, 1, (h, w)).astype(np.float32)
                n = cv2.GaussianBlur(n, (0, 0), max(3.0, min(h, w) / 24.0))
                nmax = float(np.abs(n).max()) + 1e-6
                n = n / nmax
                delta -= 55.0 * strength * alpha * (0.5 + 0.5 * n)[..., None]
    elif kind in ("spot_bright", "scratch"):
        d_add = (255.0 * strength) * (profile ** 0.75)
        if kind == "scratch":
            d_add = d_add * (0.62 + 0.38 * n_al)            # 亮核时断时续
        delta += d_add[..., None]
    elif kind in ("spot_dark",):
        delta -= (200.0 * strength) * (profile ** 0.9)[..., None]
    elif kind == "dent":
        pass   # 凹坑形貌全部由 _dent_crater_warp 合成（包绕+冲蚀+明暗+深点）
    elif kind == "stain":
        # 污渍：低频噪声调制 + 轻微棕黄色偏（BGR: B-, G中, R+）
        n = rng.normal(0, 1, (h, w)).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), max(3.0, min(h, w) / 24.0))
        nmax = float(np.abs(n).max()) + 1e-6
        n = n / nmax
        delta[..., 0] -= 45.0 * strength * alpha * (0.5 + 0.5 * n)
        delta[..., 1] -= 20.0 * strength * alpha * (0.5 + 0.5 * n)
        delta[..., 2] += 30.0 * strength * alpha * (0.5 + 0.5 * n)
        delta -= 60.0 * strength * (alpha ** 1.5)[..., None] * 0.4
    if kind == "scratch" and tint_bgr is None:
        # 划伤亮核 + 微暗晕圈：用轻微膨胀的 alpha 反差（随走向起伏）
        halo = cv2.dilate(alpha, np.ones((5, 5), np.uint8)) - alpha
        delta -= 70.0 * strength * (halo * (0.62 + 0.38 * n_al))[..., None]

    if grain > 1e-3:
        delta += rng.normal(0, 1, (h, w, 3)).astype(np.float32) * (34.0 * grain) * (alpha ** 0.6)[..., None]

    out = np.clip(base + delta, 0, 255).astype(np.uint8)
    # 凹坑/划伤：径向内陷酒窝合成（条纹四周包绕 + 坑底冲蚀 + 凹面明暗 + 深点），
    # 形貌区（弯移环带 ∪ 掩码）以羽化 alpha 随资产带走
    if kind in ("dent", "scratch"):
        out, alpha_out = _dent_crater_warp(out, mask, kind, warp, tint_bgr, strength, seed)
    else:
        alpha_out = mask
    rgba = np.dstack([out, alpha_out])
    return rgba




# ---------------------------------------------------------------- 光照协调

def harmonize_layer(canvas: np.ndarray, layer: np.ndarray, inst: dict):
    """光照协调（Reinhard 风格统计迁移）：把缺陷像素的逐通道均值/方差对齐到
    其周边环带背景，消除"贴图色调与本地光照不符"的违和感。

    参考 bcmi/Color-Transfer-for-Image-Harmonization 的全局颜色迁移思想，
    此处为确定性实现：strength=0 关闭，1 完全对齐；环带排除缺陷自身区域。
    """
    mode = inst.get("harmonize") or "off"
    strength = float(inst.get("harmonize_strength", 0.8))
    ring_px = max(2, int(inst.get("harmonize_ring", 10)))
    if mode == "off" or strength <= 1e-3:
        return
    x0, y0 = inst["_origin"]
    bh, bw = layer.shape[:2]
    th, tw = canvas.shape[:2]
    rx0 = max(0, x0 - ring_px)
    ry0 = max(0, y0 - ring_px)
    rx1 = min(tw, x0 + bw + ring_px)
    ry1 = min(th, y0 + bh + ring_px)
    ring = canvas[ry0:ry1, rx0:rx1]
    ring_sel = np.ones(ring.shape[:2], bool)
    iy0, ix0 = y0 - ry0, x0 - rx0
    ring_sel[iy0:iy0 + bh, ix0:ix0 + bw] = False
    if int(ring_sel.sum()) < 50:
        return
    alpha = layer[:, :, 3].astype(np.float32) / 255.0
    src_sel = alpha > 0.15
    if int(src_sel.sum()) < 20:
        return
    src = layer[:, :, :3].astype(np.float32)[src_sel]
    dst = ring.astype(np.float32)[ring_sel]
    luts = []
    for c in range(3):
        mu_s = float(src[:, c].mean())
        sd_s = float(src[:, c].std()) + 1e-6
        mu_t = float(dst[:, c].mean())
        sd_t = float(dst[:, c].std()) + 1e-6
        k = 1.0 + (sd_t / sd_s - 1.0) * strength
        mu = mu_s + (mu_t - mu_s) * strength
        lut = np.clip((np.arange(256, dtype=np.float32) - mu_s) * k + mu,
                      0, 255).astype(np.uint8)
        luts.append(lut)
    roi = layer[:, :, :3]
    for c in range(3):
        roi[:, :, c] = cv2.LUT(roi[:, :, c], luts[c])


# ---------------------------------------------------------------- 场景合成

def _norm_position(p):
    """position 兼容 [x,y] 与 {x,y} 两种形式。"""
    if isinstance(p, dict):
        return [float(p.get("x", 0.0)), float(p.get("y", 0.0))]
    return [float(p[0]), float(p[1])] if p else [0.0, 0.0]


def render_scene(target_bgr: np.ndarray, assets: dict, instances: list,
                 notes: list | None = None) -> np.ndarray:
    """assets: {asset_id: BGRA}; instances 按 z 序（列表顺序即叠放顺序，末位在最上）。

    外观增强在资产坐标系完成后（render_instance_layer 的 layer 上）再送入融合，
    避免"先融合后增强"导致的背景色偏（规格 §11 风险项）。
    """
    notes = notes if notes is not None else []
    canvas = target_bgr.astype(np.float32)
    for idx, inst in enumerate(instances):
        inst = dict(inst)
        inst["position"] = _norm_position(inst.get("position"))
        asset = assets.get(inst.get("asset_id"))
        if asset is None:
            notes.append("实例 %d：资产缺失，已跳过" % idx)
            continue
        layer, origin = render_instance_layer(target_bgr.shape[1], target_bgr.shape[0],
                                              inst, asset)
        if layer is None:
            notes.append("实例 %d：完全位于画布外，已跳过" % idx)
            continue
        inst = dict(inst)
        inst["_origin"] = origin
        seed = int(inst.get("noise_seed", 0)) & 0x7FFFFFFF
        apply_appearance(layer, inst, np.random.default_rng(seed))
        apply_edge_blur(layer, inst)
        harmonize_layer(canvas, layer, inst)   # 协调在增强后、融合前（规格 §11 顺序约束）
        mode = inst.get("blend", "direct")
        if mode == "poisson":
            canvas_u8 = np.clip(canvas, 0, 255).astype(np.uint8)
            if not blend_poisson(canvas_u8, layer, inst, notes):
                blend_direct_or_feather(canvas, layer, inst)
            else:
                canvas = canvas_u8.astype(np.float32)
        else:
            blend_direct_or_feather(canvas, layer, inst)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def encode_image(bgr: np.ndarray, fmt: str, quality: int = 95) -> tuple[bytes, str, str]:
    """返回 (bytes, ext, media_type)。fmt: png | jpeg | tiff"""
    fmt = (fmt or "png").lower()
    if fmt in ("jpg", "jpeg"):
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        return buf.tobytes(), ".jpg", "image/jpeg"
    if fmt in ("tif", "tiff"):
        ok, buf = cv2.imencode(".tiff", bgr)
        return buf.tobytes(), ".tiff", "image/tiff"
    ok, buf = cv2.imencode(".png", bgr)
    return buf.tobytes(), ".png", "image/png"


# ---------------------------------------------------------------- 批量抖动

def materialize_instances(instances: list, img_w: int, img_h: int) -> list:
    """按背景图实例化：rel_position(比例) → 绝对位置；scale_anchor(源短边) → 缩放映射。

    与前端 recipe.materializeSteps 的短边比例映射语义一致。
    """
    out = []
    short = min(img_w, img_h)
    for inst in instances:
        j = dict(inst)
        pos = j.get("position")
        if isinstance(pos, dict):
            j["position"] = [float(pos.get("x", 0.0)), float(pos.get("y", 0.0))]
        rel = j.pop("rel_position", None)
        anchor = j.pop("scale_anchor", None)
        if rel:
            j["position"] = [float(rel[0]) * img_w, float(rel[1]) * img_h]
        if anchor:
            k = short / max(1e-6, float(anchor))
            aff = dict(j.get("affine") or {})
            aff["scale_x"] = float(aff.get("scale_x", 1.0)) * k
            aff["scale_y"] = float(aff.get("scale_y", 1.0)) * k
            j["affine"] = aff
            mesh = j.get("mesh")
            if mesh:
                mesh = dict(mesh)
                mesh["grid"] = [[dx * k, dy * k] for dx, dy in mesh["grid"]]
                j["mesh"] = mesh
            j["feather_radius"] = float(j.get("feather_radius", 0.0)) * k
            j["edge_blur"] = float(j.get("edge_blur", 0.0)) * k
        out.append(j)
    return out


def jitter_instances(instances: list, img_w: int, img_h: int,
                     jitter: dict | None, idx: int, base_seed: int) -> list:
    """Recipe 回放随机抖动：位置 ±pos%（短边比例）、旋转 ±rot°、亮度 ±db。确定性：seed+idx。"""
    if not jitter:
        return instances
    rng = np.random.default_rng(base_seed * 100003 + idx)
    short = min(img_w, img_h)
    pos_pct = float(jitter.get("pos_pct", 0))
    rot_deg = float(jitter.get("rot_deg", 0))
    db = float(jitter.get("brightness", 0))
    out = []
    for inst in instances:
        j = dict(inst)
        pos = list(j.get("position") or [0, 0])
        pos[0] += float(rng.uniform(-1, 1)) * pos_pct / 100.0 * short
        pos[1] += float(rng.uniform(-1, 1)) * pos_pct / 100.0 * short
        j["position"] = pos
        aff = dict(j.get("affine") or {})
        aff["rotation"] = float(aff.get("rotation", 0)) + float(rng.uniform(-1, 1)) * rot_deg
        j["affine"] = aff
        if db:
            app = dict(j.get("appearance") or {})
            app["brightness"] = float(app.get("brightness", 0)) + float(rng.uniform(-1, 1)) * db
            j["appearance"] = app
        out.append(j)
    return out
