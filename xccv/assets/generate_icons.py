"""生成 XCcv 主题图标（圆角渐变底 + 白色几何图形），输出到本目录。

运行：python xccv/assets/generate_icons.py
"""

from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent

# (文件名, 渐变顶色, 渐变底色, 绘制函数)
ACCENT_TOP = (76, 139, 245)
ACCENT_BOTTOM = (31, 87, 214)
ORANGE_TOP = (245, 158, 66)
ORANGE_BOTTOM = (214, 108, 31)
GREEN_TOP = (84, 162, 75)
GREEN_BOTTOM = (46, 117, 50)
WHITE = (255, 255, 255, 235)


def _gradient(size, top, bottom):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    grad = Image.new("RGBA", (size, size))
    gd = ImageDraw.Draw(grad)
    for y in range(size):
        t = y / (size - 1)
        color = tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3)) + (255,)
        gd.line([(0, y), (size, y)], fill=color)
    mask = Image.new("L", (size, size), 0)
    md = ImageDraw.Draw(mask)
    radius = int(size * 0.24)
    md.rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=255)
    img.paste(grad, (0, 0), mask)
    return img


def _draw_annotation(d, s):
    """多边形 + 顶点圆点。"""
    w = max(3, s // 16)
    pts = [(s * 0.22, s * 0.78), (s * 0.30, s * 0.30), (s * 0.72, s * 0.24), (s * 0.80, s * 0.72)]
    d.polygon(pts, outline=WHITE, width=w)
    r = max(4, s // 14)
    for x, y in pts:
        d.ellipse([x - r, y - r, x + r, y + r], fill=WHITE)


def _draw_training(d, s):
    """上升折线 + 数据点。"""
    w = max(3, s // 15)
    pts = [(s * 0.20, s * 0.74), (s * 0.42, s * 0.52), (s * 0.60, s * 0.60), (s * 0.82, s * 0.26)]
    d.line(pts, fill=WHITE, width=w, joint="curve")
    r = max(4, s // 15)
    for x, y in pts:
        d.ellipse([x - r, y - r, x + r, y + r], fill=WHITE)


def _draw_testing(d, s):
    """十字准星 + 外框角标。"""
    w = max(3, s // 16)
    c, r = s / 2, s * 0.26
    d.ellipse([c - r, c - r, c + r, c + r], outline=WHITE, width=w)
    g = s * 0.08
    for (x1, y1, x2, y2) in (  # 四角角标
        (g, g, g + s * 0.14, g), (g, g, g, g + s * 0.14),
        (s - g, g, s - g - s * 0.14, g), (s - g, g, s - g, g + s * 0.14),
        (g, s - g, g + s * 0.14, s - g), (g, s - g, g, s - g - s * 0.14),
        (s - g, s - g, s - g - s * 0.14, s - g), (s - g, s - g, s - g, s - g - s * 0.14),
    ):
        d.line([(x1, y1), (x2, y2)], fill=WHITE, width=w)
    d.ellipse([c - s * 0.045, c - s * 0.045, c + s * 0.045, c + s * 0.045], fill=WHITE)


def _draw_app(d, s):
    """应用图标：取景框 + 准星。"""
    w = max(4, s // 14)
    g, L = s * 0.14, s * 0.20
    for (x1, y1, x2, y2) in (
        (g, g, g + L, g), (g, g, g, g + L),
        (s - g, g, s - g - L, g), (s - g, g, s - g, g + L),
        (g, s - g, g + L, s - g), (g, s - g, g, s - g - L),
        (s - g, s - g, s - g - L, s - g), (s - g, s - g, s - g, s - g - L),
    ):
        d.line([(x1, y1), (x2, y2)], fill=WHITE, width=w)
    c, r = s / 2, s * 0.19
    d.ellipse([c - r, c - r, c + r, c + r], outline=WHITE, width=w)
    d.ellipse([c - s * 0.05, c - s * 0.05, c + s * 0.05, c + s * 0.05], fill=WHITE)
    d.line([(c - s * 0.30, c), (c - s * 0.08, c)], fill=WHITE, width=max(2, s // 40))
    d.line([(c + s * 0.08, c), (c + s * 0.30, c)], fill=WHITE, width=max(2, s // 40))


ICONS = {
    "app.png": (ACCENT_TOP, ACCENT_BOTTOM, _draw_app, 256),
    "annotation.png": (ACCENT_TOP, ACCENT_BOTTOM, _draw_annotation, 128),
    "training.png": (ORANGE_TOP, ORANGE_BOTTOM, _draw_training, 128),
    "testing.png": (GREEN_TOP, GREEN_BOTTOM, _draw_testing, 128),
}


def main():
    for name, (top, bottom, drawer, out_size) in ICONS.items():
        big = 512
        img = _gradient(big, top, bottom)
        drawer(ImageDraw.Draw(img), big)
        img = img.resize((out_size, out_size), Image.LANCZOS)
        img.save(HERE / name)
        print("生成", name)


if __name__ == "__main__":
    main()
