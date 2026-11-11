"""生成 XCcv v0.5.3 全套品牌图标（设计升级版）。

设计语言（对照用户批注「更多设计美感与丰富性，保留功能语义」）：
- 底座：**对角渐变** + 顶部柔光 + 底部轻压暗 + 亮内描边（比垂直渐变更有层次）；
- 字形：**FontAwesome 5 Solid**（qtawesome 现成字形库渲染，专业比例），
  白色主体 + 柔和投影（浮起感）；
- 功能语义保留：annotation=准星（用户点名）、training=曲线、testing=放大镜、
  tools=密钥、deploy=云上传、app=eye、csharp_demo=终端。

运行：python xccv/assets/generate_icons_v3.py
"""

from pathlib import Path

import qtawesome as qta
from PIL import Image, ImageDraw, ImageFilter

from xccv.qt_slots import apply_auto_slot_module  # noqa: F401,E402

HERE = Path(__file__).resolve().parent
S = 1024          # 字形渲染/合成画布
OUT = 512
RADIUS = 0.24

from PyQt5.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

# (文件名, 字形, 渐变起, 渐变止)  对角渐变：左上 → 右下
SPECS = [
    ("app",         "fa5s.eye",               "#79dcf0", "#2f9db8"),
    ("deploy",      "fa5s.cloud-upload-alt",  "#5ec6dd", "#1f8fb5"),
    ("annotation",  "fa5s.crosshairs",        "#7fb0ff", "#3f5fe8"),
    ("training",    "fa5s.chart-line",        "#ffbc63", "#ec7412"),
    ("testing",     "fa5s.search",            "#5edcae", "#17a768"),
    ("tools",       "fa5s.key",               "#b795ff", "#7243e2"),
    ("csharp_demo", "fa5s.terminal",          "#5edcae", "#17a768"),
]

_WHITE = (255, 255, 255, 248)


def _hex(c: str, a=255):
    c = c.lstrip("#")
    return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16), a)


def _badge(top: str, bottom: str) -> Image.Image:
    """对角渐变圆角底 + 顶部柔光 + 底部轻压暗 + 亮内描边。"""
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    grad = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    gd = ImageDraw.Draw(grad)
    c_top, c_bot = _hex(top), _hex(bottom)
    for i in range(S):
        # 对角：以 (x+y)/(2S) 为进度，从左上到右下
        t = i / (S - 1)
        for j in range(0, S, 8):   # 步进 8px 画竖线段（性能可接受，一次生成）
            pass
    # 对角渐变用 numpy 更快；无 numpy 时退化为逐行（视觉近似即可）
    try:
        import numpy as np

        xs = np.linspace(0.0, 1.0, S)[None, :]
        ys = np.linspace(0.0, 1.0, S)[:, None]
        t = ((xs + ys) / 2.0) ** 1.12
        arr = np.zeros((S, S, 4), dtype=np.uint8)
        for k in range(3):
            arr[:, :, k] = (c_top[k] + (c_bot[k] - c_top[k]) * t).astype(np.uint8)
        arr[:, :, 3] = 255
        grad = Image.fromarray(arr, "RGBA")
    except ImportError:
        gd = ImageDraw.Draw(grad)
        for y in range(S):
            t = (y / (S - 1)) ** 1.12
            color = tuple(int(c_top[k] + (c_bot[k] - c_top[k]) * t)
                          for k in range(3)) + (255,)
            gd.line([(0, y), (S, y)], fill=color)
    mask = Image.new("L", (S, S), 0)
    md = ImageDraw.Draw(mask)
    r = int(S * RADIUS)
    md.rounded_rectangle([0, 0, S - 1, S - 1], radius=r, fill=255)
    img.paste(grad, (0, 0), mask)

    # 顶部柔光
    glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    gld = ImageDraw.Draw(glow)
    gld.ellipse([S * 0.02, -S * 0.5, S * 0.98, S * 0.36],
                fill=(255, 255, 255, 56))
    glow = glow.filter(ImageFilter.GaussianBlur(S * 0.05))
    transparent = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    img = Image.alpha_composite(img, Image.composite(glow, transparent, mask))

    # 底部轻压暗（增加纵深）
    shade = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shade)
    sd.ellipse([S * 0.1, S * 0.78, S * 0.9, S * 1.25], fill=(0, 0, 0, 46))
    shade = shade.filter(ImageFilter.GaussianBlur(S * 0.06))
    img = Image.alpha_composite(img, Image.composite(shade, transparent, mask))

    # 亮内描边
    bd = ImageDraw.Draw(img)
    bd.rounded_rectangle([3, 3, S - 4, S - 4], radius=r - 2,
                         outline=(255, 255, 255, 52), width=max(4, S // 340))
    return img


def _glyph(name: str, size: int) -> Image.Image:
    """FontAwesome 字形 → 透明 RGBA（白）。"""
    pm = qta.icon(name, color="#ffffff").pixmap(size, size)
    pm.setDevicePixelRatio(1.0)
    from PyQt5.QtGui import QImage

    img = QImage(pm.toImage()).convertToFormat(QImage.Format_ARGB32)
    w, h = img.width(), img.height()
    nbytes = img.sizeInBytes() if hasattr(img, "sizeInBytes") else img.byteCount()
    ptr = img.constBits()
    arr = Image.frombytes("RGBA", (w, h), ptr.asstring(nbytes), "raw", "BGRA")
    return arr


def build(fname: str, glyph_name: str, top: str, bottom: str) -> Image.Image:
    img = _badge(top, bottom)

    # 字形（约 56% 宽，居中略偏上）+ 柔和投影
    glyph = _glyph(glyph_name, S).resize((int(S * 0.56),) * 2, Image.LANCZOS)
    alpha = glyph.split()[3]
    shadow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    sh = Image.new("RGBA", glyph.size, (0, 20, 30, 110))
    shadow.paste(sh, (int((S - glyph.width) / 2) + S // 110,
                      int((S - glyph.height) / 2 - S * 0.015) + S // 90),
                 alpha)
    shadow = shadow.filter(ImageFilter.GaussianBlur(S * 0.018))
    img = Image.alpha_composite(img, shadow)
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    layer.paste(glyph, (int((S - glyph.width) / 2),
                        int((S - glyph.height) / 2) - S // 90), alpha)
    img = Image.alpha_composite(img, layer)
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
