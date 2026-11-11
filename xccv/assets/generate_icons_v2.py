"""生成 XCcv v0.5.2 全套品牌图标（精致版：4x 超采样抗锯齿 + 渐变高光 + 统一线稿）。

输出（覆盖原文件，文件名不变——各窗口引用处零改动）：
  app / annotation / training / testing / tools / deploy 的 .png(512) + .ico
  csharp_demo.ico（C# 测试程序 win32icon）

运行：python xccv/assets/generate_icons_v2.py
"""

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
S = 2048          # 超采样画布（降到 512 抗锯齿）
OUT = 512
RADIUS = 0.24     # 圆角比例

WHITE = (255, 255, 255, 242)


def _hex(c: str, a=255):
    c = c.lstrip("#")
    return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16), a)


# (文件名, 渐变顶色, 渐变底色) —— 主色族统一青蓝，平台色与其呼应
SPECS = [
    ("app",       "#6fd3e8", "#2f9db8"),
    ("deploy",    "#5ec6dd", "#2f95b3"),
    ("annotation", "#6da4f8", "#3a7ce0"),
    ("training",  "#f7a94b", "#ee7f18"),
    ("testing",   "#5ecb96", "#2fae66"),
    ("tools",     "#a37bf2", "#7c4fe0"),
    ("csharp_demo", "#5ecb96", "#2fae66"),
]


def _badge(top: str, bottom: str) -> Image.Image:
    """圆角渐变底 + 顶部柔光 + 内描边（4x 超采样）。"""
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    grad = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    gd = ImageDraw.Draw(grad)
    c_top, c_bot = _hex(top), _hex(bottom)
    for y in range(S):
        t = y / (S - 1)
        # 轻微 ease：上部渐变更慢，视觉更稳
        t = t ** 1.15
        color = tuple(int(c_top[i] + (c_bot[i] - c_top[i]) * t) for i in range(3)) \
            + (255,)
        gd.line([(0, y), (S, y)], fill=color)
    mask = Image.new("L", (S, S), 0)
    md = ImageDraw.Draw(mask)
    r = int(S * RADIUS)
    md.rounded_rectangle([0, 0, S - 1, S - 1], radius=r, fill=255)
    img.paste(grad, (0, 0), mask)
    # 顶部柔光（径向高光，模糊后仅保留在圆角内）
    glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    gld = ImageDraw.Draw(glow)
    gld.ellipse([S * 0.05, -S * 0.45, S * 0.95, S * 0.38],
                fill=(255, 255, 255, 52))
    glow = glow.filter(ImageFilter.GaussianBlur(S * 0.05))
    transparent = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    img = Image.alpha_composite(img, Image.composite(glow, transparent, mask))
    # 内描边（1.5px@512 → 6px@2048，浅白）
    bd = ImageDraw.Draw(img)
    bd.rounded_rectangle([3, 3, S - 4, S - 4], radius=r - 2,
                         outline=(255, 255, 255, 46), width=max(4, S // 340))
    return img


def _line(d, p1, p2, w, fill=WHITE):
    """圆头线段：线 + 两端圆点。"""
    d.line([p1, p2], fill=fill, width=w)
    for (x, y) in (p1, p2):
        d.ellipse([x - w / 2, y - w / 2, x + w / 2, y + w / 2], fill=fill)


def _dot(d, cx, cy, r, fill=WHITE):
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)


def _poly_pts(pts, w, d, fill=WHITE, close=True):
    seq = list(pts) + ([pts[0]] if close else [])
    for a, b in zip(seq, seq[1:]):
        _line(d, a, b, w, fill)


def draw_app(d, s):
    """品牌：对焦环（外环 + 十字刻度 + 内点）。"""
    w = s * 0.052
    cx = cy = s / 2
    r_out = s * 0.30
    d.ellipse([cx - r_out, cy - r_out, cx + r_out, cy + r_out],
              outline=WHITE, width=int(w))
    tick = s * 0.115
    for ang in (-90, 0, 90, 180):
        ax = math.cos(math.radians(ang))
        ay = math.sin(math.radians(ang))
        _line(d, (cx + ax * (r_out - tick), cy + ay * (r_out - tick)),
              (cx + ax * (r_out + tick * 0.35), cy + ay * (r_out + tick * 0.35)),
              int(w * 0.9))
    _dot(d, cx, cy, s * 0.075)


def draw_deploy(d, s):
    """部署：上箭头 + 底座（与导航图标同构）。"""
    w = s * 0.055
    cx = s / 2
    _line(d, (cx, s * 0.20), (cx, s * 0.60), int(w))
    _line(d, (cx, s * 0.20), (s * 0.34, s * 0.40), int(w))
    _line(d, (cx, s * 0.20), (s * 0.66, s * 0.40), int(w))
    _line(d, (s * 0.24, s * 0.76), (s * 0.76, s * 0.76), int(w))


def draw_annotation(d, s):
    """标注：多边形 + 顶点手柄（2026-09-21 回退：用户定稿此版）。"""
    w = s * 0.048
    pts = [(s * 0.30, s * 0.26), (s * 0.72, s * 0.30), (s * 0.76, s * 0.68),
           (s * 0.42, s * 0.76), (s * 0.24, s * 0.54)]
    _poly_pts(pts, int(w), d)
    for (x, y) in pts:
        _dot(d, x, y, w * 0.95, fill=(255, 255, 255, 255))


def draw_training(d, s):
    """训练：上升曲线 + 数据点。"""
    w = int(s * 0.052)
    pts = [(s * 0.22, s * 0.68), (s * 0.40, s * 0.52),
           (s * 0.56, s * 0.60), (s * 0.78, s * 0.30)]
    seq = pts
    for a, b in zip(seq, seq[1:]):
        _line(d, a, b, w)
    for i, (x, y) in enumerate(pts):
        _dot(d, x, y, w * (1.0 if i in (0, len(pts) - 1) else 0.8))


def draw_testing(d, s):
    """测试：放大镜 + 对勾（2026-09-21 回退：用户定稿此版）。"""
    w = int(s * 0.052)
    cx, cy, r = s * 0.46, s * 0.44, s * 0.22
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=WHITE, width=w)
    _line(d, (cx + r * 0.72, cy + r * 0.72), (s * 0.76, s * 0.76), w)
    _line(d, (cx - r * 0.45, cy + r * 0.02), (cx - r * 0.05, cy + r * 0.38), int(w * 0.85))
    _line(d, (cx - r * 0.05, cy + r * 0.38), (cx + r * 0.48, cy - r * 0.34), int(w * 0.85))


def draw_tools(d, s):
    """工具：扳手（环口 + 柄）。"""
    w = int(s * 0.058)
    d.ellipse([s * 0.24, s * 0.24, s * 0.56, s * 0.56], outline=WHITE, width=w)
    _line(d, (s * 0.52, s * 0.48), (s * 0.76, s * 0.72), int(w * 1.25))
    _line(d, (s * 0.70, s * 0.74), (s * 0.80, s * 0.64), int(w * 0.9))


def draw_csharp(d, s):
    """C# Demo：命令行窗口（框 + 提示符）。"""
    w = int(s * 0.048)
    d.rounded_rectangle([s * 0.20, s * 0.26, s * 0.80, s * 0.74],
                        radius=int(s * 0.07), outline=WHITE, width=w)
    _line(d, (s * 0.32, s * 0.47), (s * 0.42, s * 0.55), int(w * 0.85))
    _line(d, (s * 0.42, s * 0.55), (s * 0.32, s * 0.63), int(w * 0.85))
    _line(d, (s * 0.48, s * 0.63), (s * 0.66, s * 0.63), int(w * 0.85))


DRAW = {"app": draw_app, "deploy": draw_deploy, "annotation": draw_annotation,
        "training": draw_training, "testing": draw_testing, "tools": draw_tools,
        "csharp_demo": draw_csharp}


def build(name: str, top: str, bottom: str) -> Image.Image:
    img = _badge(top, bottom)
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    DRAW[name](d, S)
    img = Image.alpha_composite(img, layer)
    return img.resize((OUT, OUT), Image.LANCZOS)


def save_ico(img: Image.Image, path: Path):
    img.save(path, format="ICO",
             sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64),
                    (128, 128), (256, 256)])


def main():
    for name, top, bottom in SPECS:
        img = build(name, top, bottom)
        img.save(HERE / f"{name}.png")
        save_ico(img, HERE / f"{name}.ico")
        print("generated", name)
    print("ALL DONE ->", HERE)


if __name__ == "__main__":
    main()
