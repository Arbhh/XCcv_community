"""一次性脚本：生成模型工具平台图标 tools.png（紫色渐变 + 白色扳手造型）。"""

from pathlib import Path

from PIL import Image, ImageDraw

ASSETS = Path(__file__).resolve().parent


def main():
    size = 256
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # 圆角渐变底（紫）
    top, bottom = (139, 92, 246), (109, 62, 210)
    for y in range(size):
        t = y / size
        color = tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3)) + (255,)
        d.line([(0, y), (size, y)], fill=color)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle([8, 8, size - 8, size - 8], radius=48, fill=255)
    img.putalpha(mask)
    d = ImageDraw.Draw(img)
    # 扳手造型：圆环 + 缺口 + 手柄
    white = (255, 255, 255, 255)
    cx, cy, r = 118, 118, 52
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=white, width=26)
    d.ellipse([cx - 16, cy - 16, cx + 16, cy + 16], fill=(0, 0, 0, 0))
    d.line([(150, 150), (206, 206)], fill=white, width=30)
    # 中央小方块点缀
    d.rounded_rectangle([106, 106, 130, 130], radius=6, fill=white)
    img.save(ASSETS / "tools.png")
    print("OK tools.png")


if __name__ == "__main__":
    main()
