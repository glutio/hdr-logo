from pathlib import Path

import numpy as np
from PIL import Image

from clean_logo import HEIGHT, OUTPUT_SCALE, WIDTH, draw_symbols, fold_mask, mask, rounded


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "variants"


def distance(x, y, bounds, radius):
    left, top, right, bottom = bounds
    qx = np.abs(x - (left + right) / 2) - (right - left) / 2 + radius
    qy = np.abs(y - (top + bottom) / 2) - (bottom - top) / 2 + radius
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - radius


def texture_image(color, shading):
    rgb = np.clip(np.array(color) + shading[..., None], 0, 255)
    return Image.fromarray(np.round(rgb).astype(np.uint8)).convert("RGBA")


def render_refined_ivory(x, y, light, fine_grain, outer, symbols):
    nx, ny = x / WIDTH, y / HEIGHT
    cap_bounds = (16, 16, 523, 555)
    cap = rounded(cap_bounds, 84)
    cap_edge = distance(x, y, cap_bounds, 84)
    shoulder_edge = distance(x, y, (81, 51, 463, 444), 69)
    dy, dx = np.gradient(shoulder_edge)
    direction = -0.65 * dx - 0.75 * dy
    transition = np.clip((shoulder_edge + 9) / 43, 0, 1)
    transition = transition * transition * (3 - 2 * transition)
    face_color = np.array([229, 215, 185], dtype=np.float32)
    side_color = np.array([209, 191, 153], dtype=np.float32)
    surface = face_color + transition[..., None] * (side_color - face_color)
    shoulder = np.exp(-((shoulder_edge - 4) / 19) ** 2)
    rounded_light = 20 * direction * shoulder
    right_bounce = 8 * np.exp(-((nx - 0.87) / 0.13) ** 2) * (0.4 + 0.6 * ny)
    lower_slope = 10 * np.clip((ny - 0.75) / 0.21, 0, 1)
    edge_roll = 11 * np.exp(-np.abs(cap_edge) / 6)
    shading = 18 * light - 13 * ny + rounded_light + right_bounce - lower_slope - edge_roll + fine_grain
    image = Image.new("RGBA", (WIDTH * OUTPUT_SCALE, HEIGHT * OUTPUT_SCALE), (0, 0, 0, 0))
    image.paste(texture_image((25, 27, 20), 4 * light - 3 * ny), mask=outer)
    rgb = np.clip(surface + shading[..., None], 0, 255)
    image.paste(Image.fromarray(np.round(rgb).astype(np.uint8)).convert("RGBA"), mask=cap)
    image.paste((25, 27, 20, 255), mask=symbols)
    assert image.getpixel((450, 260))[:3] == (25, 27, 20)
    assert image.getpixel((35, 400))[0] > 100, "Refined rim did not become slimmer"
    return image


def render_defined_ivory(light, outer, symbols, shaded):
    size = (WIDTH * OUTPUT_SCALE, HEIGHT * OUTPUT_SCALE)
    cap = rounded((16, 16, 523, 555), 84)
    shoulder = rounded((61, 36, 483, 462), 84)
    shoulder_inset = rounded((67, 42, 489, 468), 84)
    highlight = Image.fromarray(np.clip(
        np.asarray(shoulder, dtype=np.int16) - np.asarray(shoulder_inset, dtype=np.int16), 0, 255
    ).astype(np.uint8))
    face = rounded((75, 48, 469, 447), 72)
    amount = 1 if shaded else 0
    surface_light = amount * (8 * light - 4)
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    image.paste((25, 27, 20, 255), mask=outer)
    image.paste(texture_image((203, 183, 146), surface_light), mask=cap)
    image.paste(texture_image((218, 200, 165), surface_light), mask=shoulder)
    image.paste(texture_image((247, 237, 213), amount * 3 * light), mask=highlight)
    image.paste(texture_image((238, 225, 196), surface_light), mask=face)
    image.paste((25, 27, 20, 255), mask=symbols)
    if not shaded:
        assert image.getpixel((700, 400))[:3] == (238, 225, 196)
        assert image.getpixel((128, 400))[:3] == (247, 237, 213)
        assert image.getpixel((142, 400))[:3] == (218, 200, 165)
        assert image.getpixel((80, 400))[:3] == (203, 183, 146)
    return image


def main():
    OUTPUT.mkdir(exist_ok=True)
    size = (WIDTH * OUTPUT_SCALE, HEIGHT * OUTPUT_SCALE)
    y, x = np.mgrid[:size[1], :size[0]].astype(np.float32) / OUTPUT_SCALE
    nx, ny = x / WIDTH, y / HEIGHT
    outer = rounded((1, 1, 537, 570), 96)
    gold = rounded((24, 23, 515, 548), 69)
    tube = rounded((65, 34, 479, 460), 85)
    face = rounded((81, 51, 463, 444), 69)
    fold = fold_mask(gold)
    symbols = mask(draw_symbols)
    edge = distance(x, y, (24, 23, 515, 548), 69)
    face_edge = distance(x, y, (81, 51, 463, 444), 69)
    dy, dx = np.gradient(edge)
    directional = -0.65 * dx - 0.75 * dy
    bevel = directional * np.exp(-np.abs(edge) / 24)
    light = np.exp(-((nx - 0.28) / 0.65) ** 2 - ((ny - 0.20) / 0.70) ** 2)
    face_roll = np.exp(-np.abs(face_edge) / 16)
    fine_grain = np.random.default_rng(4).normal(0, 0.17, edge.shape).astype(np.float32)
    variants = [
        ("satin-gold", "Satin gold", "A restrained refinement: warm satin finish, gentle bevels, and crisp black ink.",
         (195, 173, 122), (234, 219, 182), (246, 237, 218), (229, 215, 178), 0.55),
        ("ivory-keycap", "Ivory keycap", "More photographic: softly lit molded plastic, a rounded face, and a shaded lower edge.",
         (195, 176, 138), (229, 215, 185), (239, 230, 207), (226, 211, 178), 1.25),
        ("monochrome", "Monochrome", "Closest in palette to your registered grayscale artwork, with dimensional shading restored.",
         (172, 172, 172), (211, 211, 211), (241, 241, 241), (205, 205, 205), 0.70),
    ]
    previous = {path: path.read_bytes() for path in
                [OUTPUT / f"{variant[0]}.png" for variant in variants]
                + [OUTPUT / "ivory-refined-geometry.png"] if path.exists()}
    cards = []
    for slug, title, description, rim_color, face_color, tube_color, fold_color, strength in variants:
        image = Image.new("RGBA", size, (0, 0, 0, 0))
        frame_shading = 4 * light - 3 * ny + 4 * bevel
        frame_color = (25, 25, 25) if slug == "monochrome" else (25, 27, 20)
        image.paste(texture_image(frame_color, frame_shading), mask=outer)
        rim_shading = strength * (18 * light - 21 * ny + 24 * bevel) + fine_grain
        image.paste(texture_image(rim_color, rim_shading), mask=gold)
        fold_shading = strength * (12 * light - 13 * ny + 12 * bevel) + fine_grain
        image.paste(texture_image(fold_color, fold_shading), mask=fold)
        tube_shading = strength * (8 * light - 8 * ny + 3 * directional)
        image.paste(texture_image(tube_color, tube_shading), mask=tube)
        face_shading = strength * (14 * light - 10 * ny - 7 * face_roll) + fine_grain
        image.paste(texture_image(face_color, face_shading), mask=face)
        image.paste((*frame_color, 255), mask=symbols)
        image.save(OUTPUT / f"{slug}.png")
        assert np.array_equal(np.asarray(image)[..., 3], np.asarray(Image.open(ROOT / "start_logo_clean.png"))[..., 3])
        assert image.getpixel((450, 260))[:3] == frame_color
        if slug == "monochrome":
            rgb = np.asarray(image)[..., :3]
            assert np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2])
        cards.append(f"""<article><h2>{title}</h2><img src="{slug}.png" width="1078" height="1142" alt="{title} logo rendering"><p>{description}</p><a href="{slug}.png" download>Download transparent PNG</a></article>""")
        print(f"Rendered {slug}.png")
    refined = render_refined_ivory(x, y, light, fine_grain, outer, symbols)
    refined.save(OUTPUT / "ivory-refined-geometry.png")
    assert np.array_equal(np.asarray(refined)[..., 3], np.asarray(Image.open(ROOT / "start_logo_clean.png"))[..., 3])
    ink = np.asarray(symbols) == 255
    assert np.array_equal(np.asarray(refined)[ink], np.asarray(Image.open(OUTPUT / "ivory-keycap.png"))[ink])
    cards.insert(2, """<article><h2>Ivory / soft geometry</h2><img src="ivory-refined-geometry.png" width="1078" height="1142" alt="Ivory logo with a slimmer rim and continuously rounded keycap shoulder"><p>The previous geometry study: a slimmer rim and continuous shading, but too dependent on subtle lighting. Preserved for comparison.</p><a href="ivory-refined-geometry.png" download>Download transparent PNG</a></article>""")
    for path, original in previous.items():
        assert path.read_bytes() == original, f"Existing version changed: {path.name}"
    print("Rendered ivory-refined-geometry.png; previous versions preserved byte-for-byte")
    for shaded, filename in [(True, "ivory-defined-bevel.png"), (False, "ivory-defined-bevel-flat.png")]:
        defined = render_defined_ivory(light, outer, symbols, shaded)
        defined.save(OUTPUT / filename)
        assert np.array_equal(np.asarray(defined)[..., 3], np.asarray(refined)[..., 3])
        assert np.array_equal(np.asarray(defined)[ink], np.asarray(refined)[ink])
        print(f"Rendered {filename}: clear inset face and solid bevel")
    html = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Logo material studies</title><style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#141414;color:#eee}
*{box-sizing:border-box}body{max-width:1800px;margin:auto;padding:32px}
h1{margin:0 0 10px;font-size:32px}p{color:#bbb;line-height:1.6}
.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:22px;margin:30px 0}
.next{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px;margin:30px auto;max-width:1120px}
.next img{max-width:480px;margin:auto}
article{padding:20px;border:1px solid #363636;border-radius:18px;background:#1c1c1c}
h2{margin:0 0 20px;font-size:21px}img{width:100%;height:auto;display:block}
a{color:#ecd6a4;text-underline-offset:4px}button{border:1px solid #555;border-radius:7px;padding:10px 16px;background:#222;color:#eee;cursor:pointer}
.light article{background:#ece9e3}.light article h2{color:#222}.light article p{color:#555}.light article a{color:#795521}
.references{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:22px;margin-top:22px}
.references img{height:230px;object-fit:contain}.references figure{margin:0;padding:20px;background:#222;border-radius:12px}
figcaption{text-align:center;color:#bbb;margin-top:14px}footer{margin-top:30px;font-size:14px}
@media(max-width:1250px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:750px){.grid,.next{grid-template-columns:1fr}.grid article{max-width:600px;margin:auto}.references{grid-template-columns:1fr}}
</style></head><body>
<h1>Next iteration: defined ivory bevel.</h1>
<p>A clearly bounded inset face, a slimmer rim, and a solid directional bevel. No folded corner. The right image removes every gradient to test the geometry.</p>
<button type="button" onclick="document.body.classList.toggle('light')">Toggle card background</button>
<main>
<section class="next">
<article><h2>Ivory / defined bevel</h2><img src="ivory-defined-bevel.png" width="1078" height="1142" alt="Ivory keycap with a defined inset face and directional bevel"><p>Restrained material shading enhances the already-defined shapes. The symbols and their positions are unchanged.</p><a href="ivory-defined-bevel.png" download>Download transparent PNG</a></article>
<article><h2>The flat-fill test</h2><img src="ivory-defined-bevel-flat.png" width="1078" height="1142" alt="The same ivory geometry with flat colors and no gradients"><p>No shading or texture: the keycap structure is carried by five solid color regions, including the black ink and rim.</p><a href="ivory-defined-bevel-flat.png" download>Download flat transparent PNG</a></article>
</section>
<h2>All earlier versions / unchanged</h2><section class="grid">__CARDS__</section>
</main>
<h2>Reference artwork</h2><section class="references">
<figure><img src="../start_logo_original.png" width="539" height="571" alt="Original gold AI-generated logo"><figcaption>Your original gold logo</figcaption></figure>
<figure><img src="registered-reference.jpg" width="539" height="571" alt="User-provided registered grayscale logo"><figcaption>Your registered grayscale reference</figcaption></figure>
<figure><img src="../bk-0010-01-4-key.png" width="485" height="501" alt="Photograph of the original keyboard key"><figcaption>The photographed keyboard key</figcaption></figure>
</section><footer><p>These are SDR material studies, not HDR exports. Each PNG is 1078 x 1142 with transparency. <a href="../backlight/">See the new backlit HDR version</a>.</p></footer>
</body></html>"""
    (OUTPUT / "index.html").write_text(html.replace("__CARDS__", "\n".join(cards)), encoding="utf-8")
    print(f"Created {OUTPUT / 'index.html'}")


if __name__ == "__main__":
    main()
