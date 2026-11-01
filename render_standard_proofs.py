import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from clean_logo import HEIGHT, OUTPUT_SCALE, WIDTH, fold_mask, rounded
from render_variants import distance


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "standard"


def main():
    OUTPUT.mkdir(exist_ok=True)
    source = ROOT / "variants" / "ivory-keycap.png"
    original = source.read_bytes()
    base = np.asarray(Image.open(source).convert("RGBA"))
    height, width = base.shape[:2]
    y, x = np.mgrid[:height, :width].astype(np.float32) / OUTPUT_SCALE
    nx, ny = x / WIDTH, y / HEIGHT
    outer = np.asarray(rounded((1, 1, 537, 570), 96), dtype=np.float32) / 255
    gold = np.asarray(rounded((24, 23, 515, 548), 69), dtype=np.float32) / 255
    slimmer_mask = rounded((18, 17, 521, 554), 75)
    slimmer_gold = np.asarray(slimmer_mask, dtype=np.float32) / 255
    extension = np.clip((slimmer_gold - gold) / np.maximum(1 - gold, 1e-6), 0, 1)
    edge = distance(x, y, (24, 23, 515, 548), 69)
    dy, dx = np.gradient(edge)
    bevel = (-0.65 * dx - 0.75 * dy) * np.exp(-np.abs(edge) / 24)
    light = np.exp(-((nx - 0.28) / 0.65) ** 2 - ((ny - 0.20) / 0.70) ** 2)
    grain = np.random.default_rng(4).normal(0, 0.17, edge.shape).astype(np.float32)
    gold_finish = np.array([195, 176, 138]) + (
        1.25 * (18 * light - 21 * ny + 24 * bevel) + grain
    )[..., None]
    fold_finish = np.array([226, 211, 178]) + (
        1.25 * (12 * light - 13 * ny + 12 * bevel) + grain
    )[..., None]
    fold = np.asarray(fold_mask(slimmer_mask), dtype=np.float32) / 255
    finish = gold_finish + fold[..., None] * (fold_finish - gold_finish)
    slimmer = base.copy()
    slimmer[..., :3] = np.round(np.clip(
        base[..., :3] + extension[..., None] * (finish - base[..., :3]), 0, 255
    )).astype(np.uint8)
    old_edge = (slimmer_gold == 1) & (gold > 0) & (gold < 1)
    assert np.array_equal(slimmer[..., :3][old_edge], np.round(finish).astype(np.uint8)[old_edge]), "Old rim antialias seam remains"
    neutral = base.copy()
    rim = np.clip(outer - gold, 0, 1)
    neutral[..., :3] = np.round(np.clip(
        base[..., :3] + rim[..., None] * np.array([-1, -3, 4]), 0, 255
    )).astype(np.uint8)
    interior = gold == 1
    for filename, pixels in [("rim-slimmer.png", slimmer), ("rim-neutral.png", neutral)]:
        assert np.array_equal(pixels[interior], base[interior]), "Interior artwork changed"
        assert np.array_equal(pixels[..., 3], base[..., 3]), "Outer silhouette changed"
        assert np.any(pixels != base), "Border proof is identical to baseline"
        Image.fromarray(pixels).save(OUTPUT / filename)
    uniform_mask = rounded((23, 23, 515, 548), 74)
    uniform_gold = np.asarray(uniform_mask, dtype=np.float32) / 255
    uniform_fold = np.asarray(fold_mask(uniform_mask), dtype=np.float32) / 255
    uniform_fold = np.clip(uniform_fold / np.maximum(uniform_gold, 1e-6), 0, 1)
    key_finish = gold_finish + uniform_fold[..., None] * (fold_finish - gold_finish)
    frame_finish = np.array([25, 27, 20]) + (4 * light - 3 * ny + 4 * bevel)[..., None]
    corrected = base.copy()
    changed = gold != uniform_gold
    perimeter = frame_finish + uniform_gold[..., None] * (key_finish - frame_finish)
    corrected[..., :3][changed] = np.round(np.clip(perimeter[changed], 0, 255)).astype(np.uint8)
    assert np.array_equal(corrected[~changed], base[~changed]), "Artwork outside border correction changed"
    assert np.array_equal(corrected[..., 3], base[..., 3]), "Outer silhouette changed"
    assert np.any(corrected != base), "Border correction was not applied"
    Image.fromarray(corrected).save(OUTPUT / "logo.png")

    for outline, inset in [
        (outer[height // 2], uniform_gold[height // 2]),
        (outer[height // 2][::-1], uniform_gold[height // 2][::-1]),
        (outer[:, width // 2], uniform_gold[:, width // 2]),
        (outer[:, width // 2][::-1], uniform_gold[:, width // 2][::-1]),
    ]:
        thickness = np.flatnonzero(inset >= 0.5)[0] - np.flatnonzero(outline >= 0.5)[0]
        assert thickness == 44, f"Uneven straight border: {thickness}"
    corner_thickness = []
    radii = np.linspace(65, 102, 1481)

    def crossing(coverage, cx, cy, angle):
        theta = np.deg2rad(angle)
        sx = 2 * (cx + radii * np.cos(theta))
        sy = 2 * (cy + radii * np.sin(theta))
        ix, iy = np.floor(sx).astype(int), np.floor(sy).astype(int)
        fx, fy = sx - ix, sy - iy
        valid = (ix >= 0) & (iy >= 0) & (ix + 1 < width) & (iy + 1 < height)
        samples = np.zeros_like(radii)
        vx, vy = ix[valid], iy[valid]
        samples[valid] = (
            coverage[vy, vx] * (1 - fx[valid]) * (1 - fy[valid])
            + coverage[vy, vx + 1] * fx[valid] * (1 - fy[valid])
            + coverage[vy + 1, vx] * (1 - fx[valid]) * fy[valid]
            + coverage[vy + 1, vx + 1] * fx[valid] * fy[valid]
        )
        index = np.flatnonzero(samples < 0.5)[0]
        assert index > 0
        return float(np.interp(0.5, samples[index - 1:index + 1][::-1], radii[index - 1:index + 1][::-1]))

    for cx, cy, start in [(97, 97, 180), (441, 97, 270), (441, 474, 0), (97, 474, 90)]:
        for angle in [start + 15, start + 30, start + 45, start + 60, start + 75]:
            measured = 2 * (crossing(outer, cx, cy, angle) - crossing(uniform_gold, cx, cy, angle))
            assert abs(measured - 44) < 0.5, f"Uneven rendered corner: {measured}"
            corner_thickness.append(round(measured, 3))
    canonical = (OUTPUT / "logo.png").read_bytes()
    selection = {
        "status": "standard",
        "asset": "logo.png",
        "source": "../variants/ivory-keycap.png",
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "correction": "Constant-width black border; outer silhouette and non-perimeter artwork unchanged.",
        "straight_border_pixels": 44,
        "rendered_corner_measurements_pixels": corner_thickness,
        "previous_standard": "../variants/ivory-keycap.png",
        "border_alternatives": "Earlier slimmer and neutral-color proofs remain available.",
    }
    (OUTPUT / "selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    cards = []
    for title, filename, description in [
        ("Before / original ivory version", "../variants/ivory-keycap.png", "Left edge: 46px. Other straight edges: 44px. Corners: approximately 40-41px."),
        ("After / corrected standard", "logo.png", "A constant 44px border around all four sides and corners. The outside silhouette and artwork away from the border are unchanged."),
    ]:
        image_src = filename if filename != "logo.png" else f"{filename}?v={selection['sha256'][:12]}"
        sizes = "".join(
            f'<figure><img src="{image_src}" style="width:{size}px" alt="{title} at {size} pixels"><figcaption>{size}px</figcaption></figure>'
            for size in [16, 32, 48, 96]
        )
        cards.append(f"""<article><h2>{title}</h2><img class="large" src="{image_src}" width="1078" height="1142" alt="{title}"><p>{description}</p><div class="sizes">{sizes}</div><a href="{image_src}" download>Download PNG</a></article>""")
    page = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Standard ivory logo / border proofs</title><style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#141414;color:#eee}*{box-sizing:border-box}
body{width:100%;max-width:1500px;margin:auto;padding:28px 20px;overflow-wrap:anywhere}
h1{font-size:clamp(25px,4vw,34px);margin:0 0 12px}p{color:#bbb;line-height:1.6}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px;margin:26px auto;max-width:1120px}
article{min-width:0;background:#202020;border:1px solid #393939;border-radius:14px;padding:20px}
h2{font-size:19px;margin:0 0 20px}.large{display:block;width:100%;max-width:360px;height:auto;margin:auto}
.sizes{display:flex;align-items:flex-end;justify-content:space-between;gap:12px;flex-wrap:wrap;margin:24px 0}
.sizes figure{margin:0}.sizes img{display:block;height:auto}.sizes figcaption{text-align:center;color:#aaa;margin-top:10px;font-size:12px}
a{color:#ead2a2;text-underline-offset:4px}button{font:inherit;background:#262626;color:#eee;border:1px solid #555;border-radius:8px;padding:10px 15px;cursor:pointer}
.light article{background:#eeeae3}.light h2{color:#222}.light article p,.light figcaption{color:#555}.light article a{color:#795521}
footer{margin-top:25px;font-size:14px}@media(max-width:950px){.grid{grid-template-columns:1fr}article{max-width:600px;width:100%;margin:auto}}
</style></head><body>
<h1>The standard: ivory keycap.</h1>
<p>The corrected standard has a true constant-width black border. Its inner and outer corner curves now have the same centers, with a 44px offset at this image's full resolution.</p>
<p>The before version is preserved. Only the border-to-keycap perimeter changes; the outside silhouette, symbols, inset face, and internal shading are unchanged.</p>
<button type="button" onclick="document.body.classList.toggle('light')">Toggle light / dark background</button>
<main class="grid">__CARDS__</main>
<footer><p>Small previews are shown at actual CSS pixel sizes. Earlier proofs: <a href="rim-slimmer.png">slimmer border</a> | <a href="rim-neutral.png">neutral-black border</a>. <a href="selection.json">Border measurements</a> | <a href="../variants/">All previous versions</a> | <a href="../backlight/">Earlier HDR experiment</a></p></footer>
</body></html>"""
    (OUTPUT / "index.html").write_text(page.replace("__CARDS__", "\n".join(cards)), encoding="utf-8")
    assert source.read_bytes() == original, "Original ivory version changed"
    print(f"Corrected standard: sides 44px; sampled corners {min(corner_thickness):.3f}-{max(corner_thickness):.3f}px.")
    print("Original version, outer silhouette, and all non-perimeter pixels preserved.")
    print(f"Created {OUTPUT / 'index.html'}")


if __name__ == "__main__":
    main()
