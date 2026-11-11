"""生成 XCcv v0.5.3-b 全套品牌图标（复合构成版，回应「元素单一」批注）。

设计语言（Mobbin / Godly 一类现代应用图标风格：多元素几何构图 + 层次
+ 负空间），每枚 = 对角渐变底座 + 主字形（FontAwesome）+ 手绘辅助元素：
- app        ：取景框四角 + 眼睛主字形 + 底部四平台色点（讲品牌故事）
- annotation ：四角取景框 + 中央准星 + 顶点圆点（标注语义）
- training   ：三根递升柱 + 上升折线箭头（全手绘，层次递进）
- testing    ：放大镜 + 镜内准星十字 + 角部小 bbox
- tools      ：齿轮环（衬底）+ 密钥主字形
- deploy     ：云上传 + 底部服务器横线 ×3 + 信号点
- csharp_demo：终端窗口框 + 三个窗口点 + 提示符线

运行：PYTHONPATH=仓库根 python xccv/assets/generate_icons_v4.py
"""

from pathlib import Path

import qtawesome as qta
from PIL import Image, ImageDraw, ImageFilter

from PyQt5.QtWidgets import QApplication  # noqa: E402

HERE = Path(__file__).resolve().parent
S = 1024
OUT = 512
RADIUS = 0.24
WHITE = (255, 255, 255, 248)

_app = QApplication.instance() or QApplication([])

# (文件名, 主字形, 渐变起, 渐变止)
SPECS = [
    ("app",         "fa5s.eye",               "#79dcf0", "#2f9db8"),
    ("deploy",      "fa5s.cloud-upload-alt",  "#5ec6dd", "#1f8fb5"),
    ("annotation",  "fa5s.crosshairs",        "#7fb0ff", "#3f5fe8"),
    ("training",    None,                     "#ffbc63", "#ec7412"),
    ("testing",     "fa5s.search",            "#5edcae", "#17a768"),
    ("tools",       "fa5s.key",               "#b795ff", "#7243e2"),
    ("csharp_demo", "fa5s.terminal",          "#5edcae", "#17a768"),
]

# 平台色点（app 图标底部：标注/训练/测试/工具）
_PLATFORM_DOTS = ["#7fb0ff", "#ffbc63", "#5edcae", "#b795ff"]


def _hex(c: str, a=255):
    c = c.lstrip("#")
    return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16), a)


def _badge(top: str, bottom: str) -> Image.Image:
    """对角渐变圆角底 + 顶部柔光 + 底部轻压暗 + 亮内描边。"""
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

    shade = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shade)
    sd.ellipse([S * 0.1, S * 0.78, S * 0.9, S * 1.25], fill=(0, 0, 0, 46))
    shade = shade.filter(ImageFilter.GaussianBlur(S * 0.06))
    img = Image.alpha_composite(img, Image.composite(shade, transparent, mask))

    bd = ImageDraw.Draw(img)
    bd.rounded_rectangle([3, 3, S - 4, S - 4], radius=r - 2,
                         outline=(255, 255, 255, 52), width=max(4, S // 340))
    return img


def _glyph(name: str, size: int) -> Image.Image:
    pm = qta.icon(name, color="#ffffff").pixmap(size, size)
    pm.setDevicePixelRatio(1.0)
    from PyQt5.QtGui import QImage

    img = QImage(pm.toImage()).convertToFormat(QImage.Format_ARGB32)
    w, h = img.width(), img.height()
    nbytes = img.sizeInBytes() if hasattr(img, "sizeInBytes") else img.byteCount()
    arr = Image.frombytes("RGBA", (w, h),
                          img.constBits().asstring(nbytes), "raw", "BGRA")
    return arr


def _paste_glyph(img: Image.Image, name: str, size_ratio: float,
                 dy: float = -0.02):
    """主字形（含投影）居中；size_ratio 相对画布。"""
    glyph = _glyph(name, S).resize((int(S * size_ratio),) * 2, Image.LANCZOS)
    alpha = glyph.split()[3]
    x = (S - glyph.width) // 2
    y = int((S - glyph.height) / 2 + S * dy)
    shadow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    sh = Image.new("RGBA", glyph.size, (0, 20, 30, 110))
    shadow.paste(sh, (x + S // 110, y + S // 90), alpha)
    shadow = shadow.filter(ImageFilter.GaussianBlur(S * 0.018))
    img.alpha_composite(shadow)
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    layer.paste(glyph, (x, y), alpha)
    img.alpha_composite(layer)


def _line(d, p1, p2, w, fill=WHITE):
    d.line([p1, p2], fill=fill, width=w)
    for (x, y) in (p1, p2):
        d.ellipse([x - w / 2, y - w / 2, x + w / 2, y + w / 2], fill=fill)


def _dot(d, cx, cy, r, fill=WHITE):
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)


# ---------------- 复合构成（每枚 1 主 + ≤2 辅） ----------------

def deco_app(img):
    """主程序：四角取景框 + 眼睛 + 底部四平台色点。"""
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    w = int(S * 0.030)
    arm = S * 0.085
    inset = S * 0.13
    for cx, cy, dx, dy in ((inset, inset, 1, 1), (S - inset, inset, -1, 1),
                           (inset, S - inset, 1, -1),
                           (S - inset, S - inset, -1, -1)):
        _line(d, (cx, cy), (cx + dx * arm, cy), w)
        _line(d, (cx, cy), (cx, cy + dy * arm), w)
    # 底部四平台色点
    mid = S / 2
    gap = S * 0.075
    for i, color in enumerate(_PLATFORM_DOTS):
        x = mid + (i - 1.5) * gap
        _dot(d, x, S * 0.845, S * 0.018, fill=_hex(color, 235))
    img.alpha_composite(layer)


def deco_annotation(img):
    """标注：四角取景框 + 顶点圆点（准星已由主字形承担）。"""
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    w = int(S * 0.028)
    arm = S * 0.07
    inset = S * 0.155
    corners = [(inset, inset, 1, 1), (S - inset, inset, -1, 1),
               (inset, S - inset, 1, -1), (S - inset, S - inset, -1, -1)]
    for i, (cx, cy, dx, dy) in enumerate(corners):
        _line(d, (cx, cy), (cx + dx * arm, cy), w)
        _line(d, (cx, cy), (cx, cy + dy * arm), w)
        # 对角两个顶点用青色点，两个用白点（呼应参考图顶点手柄）
        color = (94, 198, 221, 255) if i % 2 == 0 else WHITE
        _dot(d, cx, cy, w * 0.85, fill=color)
    img.alpha_composite(layer)


def deco_training(img):
    """训练（全手绘）：三根递升柱 + 上升折线 + 箭头。"""
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    base = S * 0.78
    bw = S * 0.085
    for i, h_ratio in enumerate((0.18, 0.30, 0.44)):
        x = S * 0.24 + i * S * 0.145
        alpha = int(90 + i * 55)
        d.rounded_rectangle([x, base - S * h_ratio, x + bw, base],
                            radius=int(bw * 0.3),
                            fill=(255, 255, 255, alpha))
    pts = [(S * 0.24, S * 0.50), (S * 0.40, S * 0.34),
           (S * 0.55, S * 0.42), (S * 0.74, S * 0.20)]
    for a, b in zip(pts, pts[1:]):
        _line(d, a, b, int(S * 0.035))
    tip = pts[-1]
    ah = S * 0.045
    d.polygon([tip, (tip[0] - ah, tip[1] + ah * 0.35),
               (tip[0] - ah * 0.35, tip[1] + ah)], fill=WHITE)
    img.alpha_composite(layer)


def deco_testing(img):
    """测试：镜内准星十字 + 右下角部小 bbox。"""
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    cx, cy = S * 0.45, S * 0.42   # 与 glyph 放大镜镜心一致
    arm = S * 0.075
    _line(d, (cx - arm, cy), (cx + arm, cy), int(S * 0.026))
    _line(d, (cx, cy - arm), (cx, cy + arm), int(S * 0.026))
    # 角部小 bbox（右下，检测语义）
    w = int(S * 0.020)
    bx, by, bs = S * 0.66, S * 0.66, S * 0.14
    ab = bs * 0.55
    for hx, hy, dx, dy in ((bx, by, 1, 1), (bx + bs, by, -1, 1),
                           (bx, by + bs, 1, -1), (bx + bs, by + bs, -1, -1)):
        _line(d, (hx, hy), (hx + dx * ab, hy), w)
        _line(d, (hx, hy), (hx, hy + dy * ab), w)
    img.alpha_composite(layer)


def deco_tools(img):
    """工具：衬底齿轮环 + 密钥主字形。"""
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    cx, cy, r = S * 0.34, S * 0.34, S * 0.16
    d.ellipse([cx - r, cy - r, cx + r, cy + r],
              outline=(255, 255, 255, 130), width=int(S * 0.024))
    import math

    for i in range(8):
        ang = i * 45 * math.pi / 180.0
        r1, r2 = S * 0.185, S * 0.215
        d.line([(cx + math.cos(ang) * r1, cy + math.sin(ang) * r1),
                (cx + math.cos(ang) * r2, cy + math.sin(ang) * r2)],
               fill=(255, 255, 255, 130), width=int(S * 0.024))
    img.alpha_composite(layer)


def deco_deploy(img):
    """部署：云上传 + 底部三条服务器线 + 信号点。"""
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    y0 = S * 0.80
    for i, alpha in enumerate((200, 140, 90)):
        y = y0 + i * S * 0.045
        half = S * (0.20 - i * 0.035)
        _line(d, (S / 2 - half, y), (S / 2 + half, y), int(S * 0.024),
              fill=(255, 255, 255, alpha))
    _dot(d, S * 0.80, S * 0.24, S * 0.022, fill=(94, 198, 221, 230))
    img.alpha_composite(layer)


def deco_csharp(img):
    """C# Demo：窗口三点 + 提示符线。"""
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    y = S * 0.20
    for i, color in enumerate(("#ff9f6b", "#ffe08a", "#8ef0b6")):
        _dot(d, S * 0.30 + i * S * 0.055, y, S * 0.016, fill=_hex(color, 235))
    # 提示符折线 + 代码线（右下）
    _line(d, (S * 0.68, S * 0.70), (S * 0.60, S * 0.76), int(S * 0.024))
    _line(d, (S * 0.60, S * 0.76), (S * 0.68, S * 0.82), int(S * 0.024))
    _line(d, (S * 0.72, S * 0.82), (S * 0.80, S * 0.82), int(S * 0.024))
    img.alpha_composite(layer)


DECO = {
    "app": deco_app, "annotation": deco_annotation, "training": deco_training,
    "testing": deco_testing, "tools": deco_tools, "deploy": deco_deploy,
    "csharp_demo": deco_csharp,
}


def build(fname: str, glyph_name, top: str, bottom: str) -> Image.Image:
    img = _badge(top, bottom)
    if glyph_name:
        _paste_glyph(img, glyph_name, 0.50)
    DECO[fname](img)
    return img.resize((OUT, OUT), Image.LANCZOS)


def save_ico(img: Image.Image, path: Path):
    img.save(path, format="ICO",
             sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64),
                    (128, 128), (256, 256)])


def main():
    for name, glyph_name, top, bottom in SPECS:
        img = build(name, glyph_name, top, bottom)
        img.save(HERE / f"{name}.png")
        save_ico(img, HERE / f"{name}.ico")
        print("generated", name)
    print("ALL DONE ->", HERE)


if __name__ == "__main__":
    main()
