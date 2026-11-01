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
    (OUTPUT / "logo.png").write_bytes(original)
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
    finish = np.array([195, 176, 138]) + (
        1.25 * (18 * light - 21 * ny + 24 * bevel) + grain
    )[..., None]
    fold_finish = np.array([226, 211, 178]) + (
        1.25 * (12 * light - 13 * ny + 12 * bevel) + grain
    )[..., None]
    fold = np.asarray(fold_mask(slimmer_mask), dtype=np.float32) / 255
    finish += fold[..., None] * (fold_finish - finish)
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
    assert (OUTPUT / "logo.png").read_bytes() == original
    selection = {
        "status": "standard",
        "asset": "logo.png",
        "source": "../variants/ivory-keycap.png",
        "sha256": hashlib.sha256(original).hexdigest(),
        "border_alternatives": "Proofs only; the selected standard is unchanged.",
    }
    (OUTPUT / "selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    cards = []
    for title, filename, description in [
        ("A / selected standard", "logo.png", "Current warm-black rim. This is the unchanged standard."),
        ("B / modestly slimmer rim", "rim-slimmer.png", "About one-quarter less rim thickness along the straight edges. Interior artwork is pixel-identical."),
        ("C / neutral-black rim", "rim-neutral.png", "Same thickness, without the subtle olive tint. Interior artwork is pixel-identical."),
    ]:
        sizes = "".join(
            f'<figure><img src="{filename}" style="width:{size}px" alt="{title} at {size} pixels"><figcaption>{size}px</figcaption></figure>'
            for size in [16, 32, 48, 96]
        )
        cards.append(f"""<article><h2>{title}</h2><img class="large" src="{filename}" width="1078" height="1142" alt="{title}"><p>{description}</p><div class="sizes">{sizes}</div><a href="{filename}" download>Download PNG</a></article>""")
    page = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Standard ivory logo / border proofs</title><style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#141414;color:#eee}*{box-sizing:border-box}
body{width:100%;max-width:1500px;margin:auto;padding:28px 20px;overflow-wrap:anywhere}
h1{font-size:clamp(25px,4vw,34px);margin:0 0 12px}p{color:#bbb;line-height:1.6}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:20px;margin-top:26px}
article{min-width:0;background:#202020;border:1px solid #393939;border-radius:14px;padding:20px}
h2{font-size:19px;margin:0 0 20px}.large{display:block;width:100%;max-width:360px;height:auto;margin:auto}
.sizes{display:flex;align-items:flex-end;justify-content:space-between;gap:12px;flex-wrap:wrap;margin:24px 0}
.sizes figure{margin:0}.sizes img{display:block;height:auto}.sizes figcaption{text-align:center;color:#aaa;margin-top:10px;font-size:12px}
a{color:#ead2a2;text-underline-offset:4px}button{font:inherit;background:#262626;color:#eee;border:1px solid #555;border-radius:8px;padding:10px 15px;cursor:pointer}
.light article{background:#eeeae3}.light h2{color:#222}.light article p,.light figcaption{color:#555}.light article a{color:#795521}
footer{margin-top:25px;font-size:14px}@media(max-width:950px){.grid{grid-template-columns:1fr}article{max-width:600px;width:100%;margin:auto}}
</style></head><body>
<h1>The standard: ivory keycap.</h1>
<p>Your selected version is preserved exactly as <a href="logo.png">logo.png</a>. These two alternatives isolate the border: the symbols, face, pale inner rim, folded sweep, and shading inside it are unchanged.</p>
<p>My starting recommendation is A: its border helps the silhouette at small sizes. B is worth comparing, but thinner is not automatically better. C tests only the border's color neutrality.</p>
<button type="button" onclick="document.body.classList.toggle('light')">Toggle light / dark background</button>
<main class="grid">__CARDS__</main>
<footer><p>Small previews are shown at actual CSS pixel sizes. No border alternative has replaced the standard. <a href="../variants/">All previous versions</a> | <a href="../backlight/">Earlier HDR experiment</a></p></footer>
</body></html>"""
    (OUTPUT / "index.html").write_text(page.replace("__CARDS__", "\n".join(cards)), encoding="utf-8")
    print("Selected standard copied byte-for-byte; two rim-only proofs rendered.")
    print(f"Created {OUTPUT / 'index.html'}")


if __name__ == "__main__":
    main()
