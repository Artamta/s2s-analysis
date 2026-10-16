"""Build the product figures from the s2s.scdlds.com weekly guidance PDF.

Usage: python make_product_figures.py <report.pdf> <out_dir>
Renders pages 2-4 (rainfall anomaly, 850 hPa wind anomaly, T2M anomaly),
crops the 2x2 week panels and writes:
  product_panels.jpg      three 2x2 blocks side by side
  product_panels_w12.jpg  Weeks 1-2 row of each block (body figure)
  cover_strip.jpg     Weeks 1-4 rainfall anomaly in one row (cover)
"""
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageChops


def trim(im: Image.Image) -> Image.Image:
    """Remove the uniform white border around an image."""
    bg = Image.new("RGB", im.size, "white")
    return im.crop(ImageChops.difference(im, bg).getbbox())


def page_block(png: Path) -> Image.Image:
    """Crop the figure block (panels + colourbar) and drop the page title."""
    im = Image.open(png).convert("RGB")
    w, h = im.size
    block = trim(im.crop((int(0.07 * w), int(0.075 * h), int(0.93 * w), int(0.835 * h))))
    return trim(block.crop((0, int(0.033 * block.height), block.width, block.height)))


def main(pdf: str, out_dir: str) -> None:
    out = Path(out_dir)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdftoppm", "-r", "220", "-f", "2", "-l", "4", "-png", pdf, f"{tmp}/p"], check=True)
        blocks = [page_block(p) for p in sorted(Path(tmp).glob("p-*.png"))]
    # Same layout on every page; cut the T2M page's caption below its colourbar.
    blocks = [b.crop((0, 0, b.width, min(b.height, blocks[0].height + 6))) for b in blocks]

    gap = 40
    hgt = max(b.height for b in blocks)
    panel = Image.new("RGB", (sum(b.width for b in blocks) + gap * 2, hgt), "white")
    x = 0
    for b in blocks:
        panel.paste(b, (x, 0))
        x += b.width + gap
    panel.save(out / "product_panels.jpg", quality=88)

    # Weeks 1-2 only: top row of each block plus its colourbar.
    rows = []
    for b in blocks:
        top = b.crop((0, 0, b.width, int(0.462 * b.height)))
        bar = b.crop((0, int(0.935 * b.height), b.width, b.height))
        row = Image.new("RGB", (b.width, top.height + bar.height), "white")
        row.paste(top, (0, 0))
        row.paste(bar, (0, top.height))
        rows.append(row)
    hgt = max(r.height for r in rows)
    panel = Image.new("RGB", (sum(r.width for r in rows) + gap * 2, hgt), "white")
    x = 0
    for r in rows:
        panel.paste(r, (x, 0))
        x += r.width + gap
    panel.save(out / "product_panels_w12.jpg", quality=88)

    rain = blocks[0]
    w, h = rain.size
    half_w, map_h = w // 2, int(0.452 * h)
    cells = [rain.crop((cx, cy, cx + half_w, cy + map_h)) for cy in (0, int(0.4705 * h)) for cx in (0, half_w)]
    cbar = rain.crop((0, int(0.935 * h), w, h))
    strip = Image.new("RGB", (half_w * 4, map_h), "white")
    for i, c in enumerate(cells):
        strip.paste(c, (i * half_w, 0))
    cbar = cbar.resize((strip.width // 2, int(cbar.height * strip.width / 2 / cbar.width)))
    full = Image.new("RGB", (strip.width, strip.height + cbar.height), "white")
    full.paste(strip, (0, 0))
    full.paste(cbar, (strip.width // 4, strip.height))
    full.save(out / "cover_strip.jpg", quality=88)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
