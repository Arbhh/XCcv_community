"""一次性脚本：从各平台 PNG 图标生成 Windows .ico（多尺寸）。"""

from pathlib import Path

from PIL import Image

ASSETS = Path(__file__).resolve().parent

for name in ("annotation", "training", "testing"):
    src = ASSETS / f"{name}.png"
    img = Image.open(src).convert("RGBA")
    out = ASSETS / f"{name}.ico"
    img.save(out, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print(f"OK {out.name}  {out.stat().st_size // 1024} KB")
