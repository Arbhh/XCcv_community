"""生成 XCcv 子功能图标（v0.5.4 任务二，参考 chatglm 对话方案 +
Lucide/Heroicons 线性风格，与主图标族同底座）。

三模型（家族统一：同蓝底 + 单一暖橙焦点，焦点位置各不相同——
分类在「签」、检测在「牌」、分割在「面」，源自 ChatGLM 参考对话）：
- model_classification：圆角相框（内含山日符号）→ 垂线 → 三枚类别签
  （中间橙实心 = argmax，两侧白描边 = 被抑制类）
- model_detection：矩形边界框 + 四角方形锚点 + 框内浅色剪影 + 顶部探出
  橙色标签牌
- model_segmentation：有机曲线（唯一无直线）+ 白虚线选区轮廓 +
  半透明橙蒙版 + 一实一虚轮廓锚点

模型工具：
- prune        ：神经网络三层节点连线 + 橙色虚线剪切线 + 剪刀字形
- crypto       ：锁体 + 钥匙（加解密）
- accelerate   ：闪电 + 底部速度线

模型加载：
- model_load   ：接口插头 + 「Load」字样

运行：PYTHONPATH=仓库根 python xccv/assets/generate_icons_sub.py
"""

from pathlib import Path

import qtawesome as qta
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from PyQt5.QtWidgets import QApplication  # noqa: E402

HERE = Path(__file__).resolve().parent
S = 1024
OUT = 512
RADIUS = 0.24
WHITE = (255, 255, 255, 248)
ORANGE = "#ff9d4d"          # 家族焦点色（平台橙的亮变体）
ORANGE_SOLID = (255, 157, 77, 255)

_app = QApplication.instance() or QApplication([])


def _hex(c: str, a=255):
    c = c.lstrip("#")
    return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16), a)


def _badge(top: str, bottom: str) -> Image.Image:
    """对角渐变圆角底（与主图标族同款）。"""
    import numpy as np

    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    xs = np.linspace(0.0, 1.0, S)[None, :]
    ys = np.linspace(0.0, 1.0, S)[:, None]
    t = ((xs + ys) / 2.0) ** 1.12
    c_top, c_bot = _hex(top), _hex(bottom)
    arr = np.zeros((S, S, 4), dtype=np.uint8)
    for k in range(3):
        arr[:, :, k] = (c_top[k] + (c_bot[k] - c_top[k]) * t).astype(np.uint8)
    arr[:, :, 3] = 255
    grad = Image.fromarray(arr, "RGBA")
    mask = Image.new("L", (S, S), 0)
    md = ImageDraw.Draw(mask)
    r = int(S * RADIUS)
    md.rounded_rectangle([0, 0, S - 1, S - 1], radius=r, fill=255)
    img.paste(grad, (0, 0), mask)
    glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    gld = ImageDraw.Draw(glow)
    gld.ellipse([S * 0.02, -S * 0.5, S * 0.98, S * 0.36],
                fill=(255, 255, 255, 56))
    glow = glow.filter(ImageFilter.GaussianBlur(S * 0.05))
    transparent = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    img = Image.alpha_composite(img, Image.composite(glow, transparent, mask))
    bd = ImageDraw.Draw(img)
    bd.rounded_rectangle([3, 3, S - 4, S - 4], radius=r - 2,
                         outline=(255, 255, 255, 52), width=max(4, S // 340))
    return img


def _glyph(name: str, size: int, color="#ffffff") -> Image.Image:
    pm = qta.icon(name, color=color).pixmap(size, size)
    pm.setDevicePixelRatio(1.0)
    from PyQt5.QtGui import QImage

    img = QImage(pm.toImage()).convertToFormat(QImage.Format_ARGB32)
    w, h = img.width(), img.height()
    nbytes = img.sizeInBytes() if hasattr(img, "sizeInBytes") else img.byteCount()
    return Image.frombytes("RGBA", (w, h),
                           img.constBits().asstring(nbytes), "raw", "BGRA")


def _paste(img: Image.Image, layer_img: Image.Image, glyph=None,
           size_ratio: float = 0.5, dy: float = 0.0, dx: float = 0.0):
    if glyph:
        g = _glyph(glyph, S).resize((int(S * size_ratio),) * 2, Image.LANCZOS)
        alpha = g.split()[3]
        x = int((S - g.width) / 2 + S * dx)
        y = int((S - g.height) / 2 + S * dy)
        shadow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        sh = Image.new("RGBA", g.size, (0, 20, 30, 110))
        shadow.paste(sh, (x + S // 110, y + S // 90), alpha)
        shadow = shadow.filter(ImageFilter.GaussianBlur(S * 0.018))
        img.alpha_composite(shadow)
        layer_img.paste(g, (x, y), alpha)
    img.alpha_composite(layer_img)


def _line(d, p1, p2, w, fill=WHITE):
    d.line([p1, p2], fill=fill, width=w)
    for (x, y) in (p1, p2):
        d.ellipse([x - w / 2, y - w / 2, x + w / 2, y + w / 2], fill=fill)


def _dot(d, cx, cy, r, fill=WHITE):
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)


def _font(size: int):
    for name in ("arialbd.ttf", "segoeuib.ttf", "msyhbd.ttc", "arial.ttf"):
        try:
            return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)
        except OSError:
            continue
    return ImageFont.load_default()


# ---------------- 三模型（家族：蓝底 + 橙焦点） ----------------

def icon_classification() -> Image.Image:
    """分类：圆角相框（山日符号）→ 垂线 → 三枚类别签（中橙实心）。"""
    img = _badge("#7fb0ff", "#3f5fe8")
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    w = int(S * 0.030)
    # 相框
    fx0, fy0, fx1, fy1 = S * 0.26, S * 0.14, S * 0.74, S * 0.52
    d.rounded_rectangle([fx0, fy0, fx1, fy1], radius=int(S * 0.05),
                        outline=WHITE, width=w)
    # 框内山日符号
    d.ellipse([S * 0.42, S * 0.21, S * 0.52, S * 0.31], fill=WHITE)
    d.polygon([(fx0 + w, fy1 - w), (S * 0.42, S * 0.36),
               (S * 0.52, S * 0.28), (S * 0.60, S * 0.36),
               (fx1 - w, fy1 - w)], fill=WHITE)
    # 垂线连接
    _line(d, (S / 2, fy1), (S / 2, S * 0.60), int(w * 0.8))
    # 三枚类别签（中间橙实心）
    tag_w, tag_h = S * 0.16, S * 0.13
    y0 = S * 0.62
    for i in range(3):
        x0 = S / 2 + (i - 1) * (tag_w + S * 0.03) - tag_w / 2
        if i == 1:
            d.rounded_rectangle([x0, y0, x0 + tag_w, y0 + tag_h],
                                radius=int(tag_h * 0.3), fill=ORANGE_SOLID)
        else:
            d.rounded_rectangle([x0, y0, x0 + tag_w, y0 + tag_h],
                                radius=int(tag_h * 0.3),
                                outline=(255, 255, 255, 220), width=w)
    _paste(img, layer)
    return img.resize((OUT, OUT), Image.LANCZOS)


def icon_detection() -> Image.Image:
    """检测：边界框 + 四角方形锚点 + 框内剪影 + 顶部橙色标签牌。"""
    img = _badge("#5ec6dd", "#1f8fb5")
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    w = int(S * 0.030)
    x0, y0, x1, y1 = S * 0.22, S * 0.26, S * 0.78, S * 0.80
    d.rectangle([x0, y0, x1, y1], outline=WHITE, width=w)
    # 四角方形锚点
    ah = S * 0.045
    for hx, hy in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
        d.rectangle([hx - ah / 2, hy - ah / 2, hx + ah / 2, hy + ah / 2],
                    fill=WHITE)
    # 框内浅色剪影（抽象山形）
    d.polygon([(x0 + w * 1.6, y1 - w * 1.6), (S * 0.40, S * 0.58),
               (S * 0.52, S * 0.66), (S * 0.62, S * 0.54),
               (x1 - w * 1.6, y1 - w * 1.6)], fill=(255, 255, 255, 150))
    # 顶部探出标签牌（橙）
    tw, th = S * 0.26, S * 0.115
    tx = S / 2 - tw / 2
    d.rounded_rectangle([tx, y0 - th * 0.75, tx + tw, y0 + th * 0.25],
                        radius=int(th * 0.35), fill=ORANGE_SOLID)
    img.alpha_composite(layer)
    # 牌上小字（白色短横，代表类别文本）
    lay2 = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d2 = ImageDraw.Draw(lay2)
    d2.rounded_rectangle([tx + tw * 0.18, y0 - th * 0.42,
                          tx + tw * 0.82, y0 - th * 0.10],
                         radius=int(th * 0.12), fill=(255, 255, 255, 235))
    img.alpha_composite(lay2)
    return img.resize((OUT, OUT), Image.LANCZOS)


def icon_segmentation() -> Image.Image:
    """分割：有机曲线 + 细密白虚线轮廓 + 柔和白蒙版 + 青/白锚点。"""
    import math

    img = _badge("#7fb0ff", "#3f5fe8")
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    cx, cy = S * 0.5, S * 0.52
    pts = []
    for i in range(16):
        ang = i * 2 * math.pi / 16
        wob = 1 + 0.11 * math.sin(i * 2.1) + 0.06 * math.cos(i * 1.3)
        pts.append((cx + math.cos(ang) * S * 0.205 * wob,
                    cy + math.sin(ang) * S * 0.235 * wob))
    poly = [(p[0], p[1]) for p in pts]
    # 柔和白蒙版（低透明度，不与蓝底冲突）
    d.polygon(poly, fill=(255, 255, 255, 88))
    # 细密蚂蚁线
    step, dash = 18, 11
    for i in range(len(poly)):
        p1, p2 = poly[i], poly[(i + 1) % len(poly)]
        seg = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        n = max(1, int(seg // step))
        for k in range(n):
            if k % 2 == 0:
                t0 = k / n
                t1 = min(1.0, (k * step + dash) / seg) if seg else 1
                a = (p1[0] + (p2[0] - p1[0]) * t0,
                     p1[1] + (p2[1] - p1[1]) * t0)
                b = (p1[0] + (p2[0] - p1[0]) * t1,
                     p1[1] + (p2[1] - p1[1]) * t1)
                d.line([a, b], fill=WHITE, width=int(S * 0.014))
    # 锚点：一白实心、一青空心（呼应平台主色，替代突兀橙）
    d.ellipse([poly[0][0] - 15, poly[0][1] - 15,
               poly[0][0] + 15, poly[0][1] + 15], fill=WHITE)
    px, py = poly[8]
    d.ellipse([px - 15, py - 15, px + 15, py + 15],
              outline=(94, 198, 221, 255), width=int(S * 0.014))
    img.alpha_composite(layer)
    return img.resize((OUT, OUT), Image.LANCZOS)


# ---------------- 模型工具三枚 ----------------

def icon_prune() -> Image.Image:
    """剪枝：神经网络三层分布 + 橙虚线剪切线 + 剪刀字形。"""
    img = _badge("#b795ff", "#7243e2")
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    layers = [(S * 0.24, [0.30, 0.50, 0.70]),
              (S * 0.50, [0.22, 0.42, 0.62, 0.80]),
              (S * 0.74, [0.38, 0.60])]
    coords = []
    for x, fracs in layers:
        col = [(x, S * f) for f in fracs]
        coords.append(col)
        for (px, py) in col:
            _dot(d, px, py, S * 0.030, fill=WHITE)
    for a, b in zip(coords, coords[1:]):
        for p1 in a:
            for p2 in b:
                d.line([p1, p2], fill=(255, 255, 255, 110), width=int(S * 0.006))
    # 重新画亮节点（连线叠上）
    for col in coords:
        for (px, py) in col:
            _dot(d, px, py, S * 0.028, fill=WHITE)
    # 橙虚线剪切线（对角跨过网络）
    for k in range(6):
        t0, t1 = k / 6, min(1.0, (k + 0.55) / 6)
        a = (S * 0.16 + (S * 0.84 - S * 0.16) * t0,
             S * 0.88 - (S * 0.88 - S * 0.14) * t0)
        b = (S * 0.16 + (S * 0.84 - S * 0.16) * t1,
             S * 0.88 - (S * 0.88 - S * 0.14) * t1)
        if k % 2 == 0:
            d.line([a, b], fill=ORANGE_SOLID, width=int(S * 0.014))
    img.alpha_composite(layer)
    # 剪刀字形（右下角）
    g = _glyph("fa5s.cut", S).resize((int(S * 0.20),) * 2, Image.LANCZOS)
    img.alpha_composite(g, (int(S * 0.70), int(S * 0.70)))
    return img.resize((OUT, OUT), Image.LANCZOS)


def icon_crypto() -> Image.Image:
    """加解密：锁体（fa5s.lock 线性感觉）+ 斜插钥匙。"""
    img = _badge("#b795ff", "#7243e2")
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    w = int(S * 0.034)
    # 锁体（圆角矩形）+ 锁梁
    bx0, by0, bx1, by1 = S * 0.24, S * 0.42, S * 0.68, S * 0.82
    d.rounded_rectangle([bx0, by0, bx1, by1], radius=int(S * 0.06),
                        outline=WHITE, width=w)
    d.arc([S * 0.32, S * 0.20, S * 0.60, S * 0.50], 180, 360,
          fill=WHITE, width=w)
    _line(d, (S * 0.32, S * 0.36), (S * 0.32, S * 0.46), w)
    _line(d, (S * 0.60, S * 0.36), (S * 0.60, S * 0.46), w)
    # 锁孔（橙）
    _dot(d, (bx0 + bx1) / 2, (by0 + by1) / 2 - S * 0.02, S * 0.030,
         fill=ORANGE_SOLID)
    d.rounded_rectangle([(bx0 + bx1) / 2 - S * 0.014,
                         (by0 + by1) / 2 - S * 0.01,
                         (bx0 + bx1) / 2 + S * 0.014,
                         (by0 + by1) / 2 + S * 0.09],
                        radius=int(S * 0.008), fill=ORANGE_SOLID)
    img.alpha_composite(layer)
    # 钥匙（右下，斜插）
    g = _glyph("fa5s.key", S).resize((int(S * 0.26),) * 2, Image.LANCZOS)
    img.alpha_composite(g, (int(S * 0.62), int(S * 0.62)))
    return img.resize((OUT, OUT), Image.LANCZOS)


def icon_accelerate() -> Image.Image:
    """加速：闪电 + 底部速度线。"""
    img = _badge("#ffbc63", "#ec7412")
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    y0 = S * 0.84
    for i, alpha in enumerate((210, 150, 95)):
        half = S * (0.22 - i * 0.04)
        y = y0 + i * S * 0.045
        _line(d, (S / 2 - half, y), (S / 2 + half, y), int(S * 0.026),
              fill=(255, 255, 255, alpha))
    img.alpha_composite(layer)
    _paste(img, layer, glyph="fa5s.bolt", size_ratio=0.52, dy=-0.04)
    return img.resize((OUT, OUT), Image.LANCZOS)


def icon_model_load() -> Image.Image:
    """模型加载：接口插头 + 「Load」字样。"""
    img = _badge("#b795ff", "#7243e2")
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    # 接口座（圆角方 + 两针孔）
    d.rounded_rectangle([S * 0.22, S * 0.16, S * 0.78, S * 0.50],
                        radius=int(S * 0.06), outline=WHITE, width=int(S * 0.030))
    _line(d, (S * 0.42, S * 0.28), (S * 0.42, S * 0.38), int(S * 0.026))
    _line(d, (S * 0.58, S * 0.28), (S * 0.58, S * 0.38), int(S * 0.026))
    # 连接箭头向下
    _line(d, (S / 2, S * 0.52), (S / 2, S * 0.60), int(S * 0.026))
    d.polygon([(S * 0.44, S * 0.58), (S * 0.56, S * 0.58), (S / 2, S * 0.66)],
              fill=WHITE)
    img.alpha_composite(layer)
    # 「Load」文字
    g = _glyph("fa5s.plug", S).resize((int(S * 0.001),) * 2)  # 预热字体缓存
    text = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    td = ImageDraw.Draw(text)
    font = _font(int(S * 0.16))
    td.text((S / 2, S * 0.80), "Load", font=font, fill=WHITE, anchor="mm")
    shadow = text.filter(ImageFilter.GaussianBlur(S * 0.010))
    sh_col = Image.new("RGBA", (S, S), (0, 20, 30, 110))
    sh_col.putalpha(shadow.split()[3])
    img.alpha_composite(sh_col)
    img.alpha_composite(text)
    return img.resize((OUT, OUT), Image.LANCZOS)


def icon_flow() -> Image.Image:
    """流程图编辑器（独立于模型工具）：输入节点 → 分叉连线 → 双输出节点。

    家族规则同三模型：单一暖橙焦点（汇合后右侧的实心节点），
    其余节点白描边；连线带方向箭头，表达数据流。
    """
    img = _badge("#9aa7ff", "#5460e6")
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    w = int(S * 0.028)
    node = S * 0.155   # 节点边长
    # 输入节点（左中，白描边）
    ax0, ay0 = S * 0.13, S * 0.425
    d.rounded_rectangle([ax0, ay0, ax0 + node, ay0 + node],
                        radius=int(S * 0.045), outline=WHITE, width=w)
    _dot(d, ax0 + node / 2, ay0 + node / 2, S * 0.022)
    # 输出节点（右上 橙实心=焦点；右下 白描边）
    bx0, by0 = S * 0.685, S * 0.145
    d.rounded_rectangle([bx0, by0, bx0 + node, by0 + node],
                        radius=int(S * 0.045), fill=ORANGE_SOLID)
    cx0, cy0 = S * 0.685, S * 0.70
    d.rounded_rectangle([cx0, cy0, cx0 + node, cy0 + node],
                        radius=int(S * 0.045), outline=WHITE, width=w)
    # 分叉连线（输入 → 两输出）+ 箭头
    sx, sy = ax0 + node, ay0 + node / 2
    arrow = int(S * 0.038)
    for ex, ey in ((bx0 - S * 0.012, by0 + node / 2),
                   (cx0 - S * 0.012, cy0 + node / 2)):
        _line(d, (sx, sy), (ex - arrow, ey), int(S * 0.024))
        d.polygon([(ex, ey),
                   (ex - arrow, ey - arrow * 0.62),
                   (ex - arrow, ey + arrow * 0.62)], fill=WHITE)
    _paste(img, layer)
    return img.resize((OUT, OUT), Image.LANCZOS)


def icon_defectgen() -> Image.Image:
    """缺陷生成（DefectAugmentStudio，P6/T6-3）：画布 + 橙色瑕疵斑 + 加号。

    家族规则：单一暖橙焦点=画布上的瑕疵斑（有机形）；画布白描边圆角框；
    右下角白圆底加号=「生成」语义。
    """
    img = _badge("#7dd3c8", "#0f9488")
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    w = int(S * 0.030)
    # 画布（白描边圆角框，微透视留出右下角加号位）
    d.rounded_rectangle([S * 0.14, S * 0.17, S * 0.80, S * 0.72],
                        radius=int(S * 0.05), outline=WHITE, width=w)
    # 瑕疵斑（橙实心有机形 = 焦点）
    import math as _math

    cx, cy = S * 0.44, S * 0.42
    pts = []
    for i in range(14):
        ang = i / 14 * 2 * _math.pi
        r = S * (0.105 + 0.030 * _math.sin(ang * 3 + 1.2)
                 + 0.018 * _math.cos(ang * 5))
        pts.append((cx + r * _math.cos(ang), cy + r * 0.86 * _math.sin(ang)))
    d.polygon(pts, fill=ORANGE_SOLID)
    # 画布内淡网格线（标注感）
    for gx in (0.30, 0.46, 0.62):
        d.line([(S * gx, S * 0.185), (S * gx, S * 0.705)],
               fill=(255, 255, 255, 60), width=max(2, int(S * 0.006)))
    for gy in (0.32, 0.47, 0.62):
        d.line([(S * 0.15, S * gy), (S * 0.79, S * gy)],
               fill=(255, 255, 255, 60), width=max(2, int(S * 0.006)))
    # 右下角加号（白圆底 + 主底色十字）
    ccx, ccy, cr = S * 0.74, S * 0.72, S * 0.115
    d.ellipse([ccx - cr, ccy - cr, ccx + cr, ccy + cr],
              fill=(255, 255, 255, 240))
    tw = int(S * 0.038)
    d.rounded_rectangle([ccx - S * 0.062, ccy - tw / 2,
                         ccx + S * 0.062, ccy + tw / 2],
                        radius=tw // 2, fill=_hex("#0f9488"))
    d.rounded_rectangle([ccx - tw / 2, ccy - S * 0.062,
                         ccx + tw / 2, ccy + S * 0.062],
                        radius=tw // 2, fill=_hex("#0f9488"))
    _paste(img, layer)
    return img.resize((OUT, OUT), Image.LANCZOS)


BUILDERS = {
    "defectgen": icon_defectgen,
    "model_classification": icon_classification,
    "model_detection": icon_detection,
    "model_segmentation": icon_segmentation,
    "prune": icon_prune,
    "crypto": icon_crypto,
    "accelerate": icon_accelerate,
    "model_load": icon_model_load,
    "flow": icon_flow,
}


def save_ico(img: Image.Image, path: Path):
    img.save(path, format="ICO",
             sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64),
                    (128, 128), (256, 256)])


def main():
    for name, builder in BUILDERS.items():
        img = builder()
        img.save(HERE / f"{name}.png")
        save_ico(img, HERE / f"{name}.ico")
        print("generated", name)
    print("ALL DONE ->", HERE)


if __name__ == "__main__":
    main()
