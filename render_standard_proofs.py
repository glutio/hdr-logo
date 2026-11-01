import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from clean_logo import HEIGHT, OUTPUT_SCALE, WIDTH, box, draw_symbols, fold_mask, mask, points, rounded
from render_variants import distance


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "standard"
PHOTO_SIZE = (485, 501)
PHOTO_GUIDE_BOUNDS = {
    "four": (120, 59, 204, 173),
    "currency": (108, 229, 213, 330),
}


def photo_coords(coords):
    return [(x * WIDTH / PHOTO_SIZE[0], y * HEIGHT / PHOTO_SIZE[1]) for x, y in coords]


def draw_photo_proportions(draw):
    number = [(120, 132), (179, 60), (186, 59), (186, 132), (204, 132),
              (204, 140), (186, 140), (186, 173), (178, 173),
              (178, 140), (120, 140)]
    draw.polygon(points(photo_coords(number)), fill=255)
    draw.polygon(points(photo_coords([(128, 132), (178, 72), (178, 132)])), fill=0)
    draw.ellipse(box(np.asarray(photo_coords([(112, 234), (210, 327)])).ravel()), fill=255)
    for tip in [
        [(134, 251), (115, 229), (112, 231), (128, 257)],
        [(190, 251), (210, 229), (213, 232), (195, 257)],
        [(130, 308), (108, 327), (112, 330), (136, 313)],
        [(190, 311), (210, 330), (213, 327), (195, 305)],
    ]:
        draw.polygon(points(photo_coords(tip)), fill=255)
    draw.ellipse(box(np.asarray(photo_coords([(120, 242), (202, 319)])).ravel()), fill=0)


def draw_photo_symbols(draw):
    number = [(158, 164), (234, 66), (240, 64), (240, 164), (260, 164),
              (260, 172), (240, 172), (240, 210), (231, 210),
              (231, 172), (158, 172)]
    draw.polygon(points(number), fill=255)
    draw.polygon(points([(171, 164), (231, 85), (231, 164)]), fill=0)
    draw.ellipse(box((145.5, 284.5, 274.5, 413.5)), fill=255)
    center = np.array([210, 349])
    for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        direction = np.array([dx, dy]) / np.sqrt(2)
        normal = np.array([-direction[1], direction[0]])
        root, tip = center + 54 * direction, center + 83 * direction
        draw.polygon(points([
            root + 4.5 * normal, tip + 1.3 * normal,
            tip - 1.3 * normal, root - 4.5 * normal,
        ]), fill=255)
    draw.ellipse(box((154, 293, 266, 405)), fill=0)


def render_photo_standard(standard, x, y, light, grain,
                          draw_glyphs=draw_photo_symbols, filename="logo-photo-glyphs.png"):
    old_symbols = np.asarray(mask(draw_symbols))
    new_symbols = np.asarray(mask(draw_glyphs))
    face = np.asarray(rounded((81, 51, 463, 444), 69))
    face_roll = np.exp(-np.abs(distance(x, y, (81, 51, 463, 444), 69)) / 16)
    shading = 1.25 * (14 * light - 10 * y / HEIGHT - 7 * face_roll) + grain
    face_rgb = np.round(np.clip(
        np.array([229, 215, 185]) + shading[..., None], 0, 255
    )).astype(np.uint8)
    unprinted = (face == 255) & (old_symbols == 0)
    assert np.array_equal(face_rgb[unprinted], standard[..., :3][unprinted]), "Reconstructed ivory shading differs"
    affected = (old_symbols > 0) | (new_symbols > 0)
    assert np.all(face[affected] == 255), "New glyphs extend outside the ivory face"
    replacement = Image.fromarray(face_rgb).convert("RGBA")
    replacement.paste((25, 27, 20, 255), mask=Image.fromarray(new_symbols))
    pixels = standard.copy()
    pixels[affected] = np.asarray(replacement)[affected]
    assert np.array_equal(pixels[~affected], standard[~affected]), "Artwork outside the glyphs changed"
    assert np.array_equal(pixels[..., 3], standard[..., 3]), "Keycap silhouette changed"
    assert np.all(pixels[old_symbols > new_symbols] == np.asarray(replacement)[old_symbols > new_symbols]), "Old glyph remnants remain"
    old_ink = int(np.count_nonzero(old_symbols == 255))
    new_ink = int(np.count_nonzero(new_symbols == 255))
    assert 0 < new_ink < old_ink, "Photo-inspired glyphs are not lighter"
    Image.fromarray(pixels).save(OUTPUT / filename)
    return {
        "original_opaque_ink_pixels": old_ink,
        "variant_opaque_ink_pixels": new_ink,
        "changed_pixels": int(np.count_nonzero(np.any(pixels != standard, axis=-1))),
        "outside_glyphs_unchanged": True,
        "silhouette_unchanged": True,
    }


def validate_photo_proportions():
    coverage = np.asarray(mask(draw_photo_proportions))
    split = round(200 * HEIGHT * OUTPUT_SCALE / PHOTO_SIZE[1])
    measurements = {}
    for name, section, offset in [
        ("four", coverage[:split], 0),
        ("currency", coverage[split:], split),
    ]:
        ys, xs = np.nonzero(section >= 128)
        actual = np.array([xs.min(), ys.min() + offset, xs.max(), ys.max() + offset])
        expected = np.array(PHOTO_GUIDE_BOUNDS[name]) * np.tile(
            np.array([WIDTH, HEIGHT]) * OUTPUT_SCALE / PHOTO_SIZE, 2
        )
        assert np.all(np.abs(actual - expected) < 2), f"{name} misses the collage guides: {actual} vs {expected}"
        measurements[name] = {
            "photo_bounds": list(PHOTO_GUIDE_BOUNDS[name]),
            "expected_output_bounds": expected.round(3).tolist(),
            "rendered_output_bounds": actual.tolist(),
        }
    return measurements


def create_proportion_proof():
    proof = Image.open(OUTPUT / "photo-proportions-reference.png").convert("RGB").crop((0, 0, 1037, 560))
    variant = Image.open(OUTPUT / "logo-photo-proportions.png").resize(PHOTO_SIZE, Image.Resampling.LANCZOS)
    proof.paste(variant, (12, 11), mask=variant.getchannel("A"))
    draw = ImageDraw.Draw(proof)
    for left, top, right, bottom in PHOTO_GUIDE_BOUNDS.values():
        draw.line([(12, top + 11), (736, top + 11)], fill=(255, 0, 0))
        draw.line([(12, bottom + 11), (736, bottom + 11)], fill=(255, 0, 0))
        draw.rectangle((left + 12, top + 11, right + 12, bottom + 11), outline=(255, 0, 0))
    draw.text((12, 530), "Corrected variant / same key frame", fill="white", font_size=16)
    draw.text((523, 530), "Your photo / original guide lines", fill="white", font_size=16)
    proof.save(OUTPUT / "photo-proportions-proof.png")


def render_flat_standard(standard):
    layers = [
        ("border", rounded((1, 1, 537, 570), 96)),
        ("shoulder", rounded((23, 23, 515, 548), 74)),
        ("fold", fold_mask(rounded((23, 23, 515, 548), 74))),
        ("rim", rounded((65, 34, 479, 460), 85)),
        ("face", rounded((81, 51, 463, 444), 69)),
        ("symbols", mask(draw_symbols)),
    ]
    coverage = [np.asarray(layer) for _, layer in layers]
    pixels = standard.copy()
    pixels[..., :3][pixels[..., 3] > 0] = (25, 27, 20)
    image = Image.fromarray(pixels)
    palette, cores = {}, {}
    for index, (name, layer) in enumerate(layers):
        core = coverage[index] == 255
        for higher in coverage[index + 1:]:
            core &= higher == 0
        assert np.any(core), f"No fully covered pixels in {name}"
        color = (25, 27, 20) if name in ("border", "symbols") else tuple(
            np.round(np.median(standard[..., :3][core], axis=0)).astype(int).tolist()
        )
        palette[name] = color
        cores[name] = core
        if index > 0:
            image.paste((*color, 255), mask=layer)
    flat = np.asarray(image)
    assert np.array_equal(flat[..., 3], standard[..., 3]), "Flat version changed transparency or silhouette"
    assert np.array_equal(flat[coverage[-1] == 255], standard[coverage[-1] == 255]), "Flat version changed symbol geometry or ink"
    assert np.array_equal(flat[coverage[1] == 0], standard[coverage[1] == 0]), "Flat version changed the outer border"
    for name, core in cores.items():
        colors = np.unique(flat[..., :3][core], axis=0)
        assert np.array_equal(colors, np.array(palette[name])[None, :]), f"{name} is not one solid color"
    assert not np.array_equal(flat, standard), "Flat alternative is identical to shaded standard"
    image.save(OUTPUT / "logo-flat-color.png")
    return palette, {name: int(np.count_nonzero(core)) for name, core in cores.items()}


def main():
    OUTPUT.mkdir(exist_ok=True)
    source = ROOT / "variants" / "ivory-keycap.png"
    original = source.read_bytes()
    previous_standard = (OUTPUT / "logo.png").read_bytes() if (OUTPUT / "logo.png").exists() else None
    previous_versions = {path: path.read_bytes() for path in OUTPUT.glob("*.png")
                         if path.name not in ("logo-photo-proportions.png", "photo-proportions-proof.png")}
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
    shaded = corrected.copy()
    archive = OUTPUT / "logo-before-flat-border.png"
    if archive.exists():
        assert np.array_equal(np.asarray(Image.open(archive)), shaded), "Archived border version changed"
    else:
        Image.fromarray(shaded).save(archive)
    ink_color = np.array([25, 27, 20])
    border_pixels = (outer > 0) & (uniform_gold < 1)
    flat_perimeter = ink_color + uniform_gold[..., None] * (key_finish - ink_color)
    corrected[..., :3][border_pixels] = np.round(np.clip(flat_perimeter[border_pixels], 0, 255)).astype(np.uint8)
    assert np.array_equal(corrected[uniform_gold == 1], shaded[uniform_gold == 1]), "Ivory artwork changed"
    assert np.array_equal(corrected[..., 3], shaded[..., 3]), "Border geometry or transparency changed"
    opaque_border = (outer == 1) & (uniform_gold == 0)
    border_colors = np.unique(corrected[..., :3][opaque_border], axis=0)
    assert np.array_equal(border_colors, ink_color[None, :]), f"Border still has multiple colors: {border_colors}"
    shaded_colors = np.unique(shaded[..., :3][opaque_border], axis=0)
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
    if previous_standard is not None:
        assert canonical == previous_standard, "Approved standard changed"
    palette, core_counts = render_flat_standard(corrected)
    flat_bytes = (OUTPUT / "logo-flat-color.png").read_bytes()
    flat_selection = {
        "status": "alternative",
        "asset": "logo-flat-color.png",
        "source": "logo.png",
        "source_sha256": hashlib.sha256(canonical).hexdigest(),
        "sha256": hashlib.sha256(flat_bytes).hexdigest(),
        "colors": {name: "#{:02x}{:02x}{:02x}".format(*color) for name, color in palette.items()},
        "verified_solid_region_pixels": core_counts,
        "gradients": False,
        "dithering": False,
        "texture": False,
        "antialiasing": "Only geometric edge transitions contain blended colors.",
        "geometry": "Same silhouette, 44px border, rim, fold, face, and symbols as the approved standard.",
    }
    (OUTPUT / "flat-color-selection.json").write_text(json.dumps(flat_selection, indent=2) + "\n", encoding="utf-8")
    photo_checks = render_photo_standard(corrected, x, y, light, grain)
    photo_bytes = (OUTPUT / "logo-photo-glyphs.png").read_bytes()
    photo_selection = {
        "status": "alternative",
        "asset": "logo-photo-glyphs.png",
        "source": "logo.png",
        "source_sha256": hashlib.sha256(canonical).hexdigest(),
        "sha256": hashlib.sha256(photo_bytes).hexdigest(),
        "reference": "../bk-0010-01-4-key.png",
        "changes": "Thinner pointed 4, lighter ring, shorter tapered tips, and increased glyph spacing.",
        "fidelity": "A clean photo-inspired approximation, not an exact tracing of the blurred printing.",
        "validation": photo_checks,
    }
    (OUTPUT / "photo-glyph-selection.json").write_text(json.dumps(photo_selection, indent=2) + "\n", encoding="utf-8")
    proportion_checks = render_photo_standard(
        corrected, x, y, light, grain, draw_photo_proportions, "logo-photo-proportions.png"
    )
    guide_measurements = validate_photo_proportions()
    proportion_bytes = (OUTPUT / "logo-photo-proportions.png").read_bytes()
    proportion_selection = {
        "status": "alternative",
        "asset": "logo-photo-proportions.png",
        "source": "logo.png",
        "source_sha256": hashlib.sha256(canonical).hexdigest(),
        "sha256": hashlib.sha256(proportion_bytes).hexdigest(),
        "reference": "photo-proportions-reference.png",
        "photo_size": list(PHOTO_SIZE),
        "collage_photo_origin": [523, 11],
        "guides": guide_measurements,
        "changes": "Glyph size, relative proportions, spacing, and placement follow the user's collage guides.",
        "fidelity": "Measured guide bounds; cleaned outlines approximate the blurred printing.",
        "validation": proportion_checks,
    }
    (OUTPUT / "photo-proportions-selection.json").write_text(
        json.dumps(proportion_selection, indent=2) + "\n", encoding="utf-8"
    )
    create_proportion_proof()
    selection = {
        "status": "standard",
        "asset": "logo.png",
        "source": "../variants/ivory-keycap.png",
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "correction": "Constant-width border with one solid ink color; ivory artwork and geometry unchanged.",
        "border_color": "#191b14",
        "opaque_border_unique_colors": len(border_colors),
        "straight_border_pixels": 44,
        "rendered_corner_measurements_pixels": corner_thickness,
        "previous_standard": "logo-before-flat-border.png",
        "border_alternatives": "Earlier slimmer and neutral-color proofs remain available.",
    }
    (OUTPUT / "selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    cards = []
    for title, filename, description in [
        ("Standard / shaded ivory", "logo.png", "The approved standard: a solid #191b14 border and symbols, with the original shaded ivory artwork."),
        ("Photo proportions / corrected", "logo-photo-proportions.png", "Glyph size, proportions, spacing, and position follow your collage guides. Both glyphs are smaller and further left than in my first attempt; the ring is also higher. All keycap artwork is unchanged."),
        ("Earlier photo attempt / superseded", "logo-photo-glyphs.png", "Preserved for comparison. This attempt changed stroke style but guessed the glyph proportions and placement."),
        ("Alternative / solid colors", "logo-flat-color.png", "Identical geometry with five solid colors: ink, shoulder, fold, pale rim, and ivory face. No gradients, dithering, or texture; only edges are antialiased."),
    ]:
        image_src = f"{filename}?v={hashlib.sha256((OUTPUT / filename).read_bytes()).hexdigest()[:12]}"
        sizes = "".join(
            f'<figure><img src="{image_src}" style="width:{size}px" alt="{title} at {size} pixels"><figcaption>{size}px</figcaption></figure>'
            for size in [16, 32, 48, 96]
        )
        cards.append(f"""<article><h2>{title}</h2><img class="large" src="{image_src}" width="1078" height="1142" alt="{title}"><p>{description}</p><div class="sizes">{sizes}</div><a href="{image_src}" download>Download PNG</a></article>""")
    page = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Standard ivory logo / measured photo proportions</title><style>
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
.reference{margin:28px auto;max-width:900px}.reference img{display:block;width:100%;height:auto}.reference figcaption{color:#bbb;margin-top:12px}
footer{margin-top:25px;font-size:14px}@media(max-width:950px){.grid{grid-template-columns:1fr}article{max-width:600px;width:100%;margin:auto}}
</style></head><body>
<h1>The standard: ivory keycap.</h1>
<p>The border now has both even geometry and even color: 44px wide, with one solid <code>#191b14</code> ink color matching the symbols. No lighting gradient remains on the border.</p>
<p>The approved shaded standard is unchanged. The corrected alternative follows your collage guide lines for glyph size, proportions, spacing, and placement. The first photo attempt is preserved, but superseded. Clean outlines approximate the photographed printing rather than reproduce blur.</p>
<button type="button" onclick="document.body.classList.toggle('light')">Toggle light / dark background</button>
<main class="grid">__CARDS__</main>
<figure class="reference"><img src="photo-proportions-proof.png" width="1037" height="560" alt="Corrected variant and photographed key aligned to the same red proportion guides"><figcaption>Matched key frames and shared guides: 4 spans 84 x 114 photo pixels; currency symbol spans 105 x 101, with a 56px gap. <a href="photo-proportions-reference.png">Your untouched collage</a>. The native logo above keeps its original aspect ratio.</figcaption></figure>
<footer><p>Antialiased edge pixels blend normally with neighboring regions or the transparent background. Earlier versions: <a href="../variants/ivory-keycap.png">original ivory</a> | <a href="logo-before-flat-border.png">even-width shaded border</a> | <a href="rim-slimmer.png">slimmer border</a> | <a href="rim-neutral.png">neutral-black border</a>. <a href="selection.json">Border measurements</a> | <a href="photo-proportions-selection.json">Measured photo proportions and validation</a> | <a href="photo-glyph-selection.json">First photo attempt validation</a> | <a href="flat-color-selection.json">Flat-color palette and validation</a> | <a href="../variants/">All previous versions</a> | <a href="../backlight/">Earlier HDR experiment</a></p></footer>
</body></html>"""
    (OUTPUT / "index.html").write_text(page.replace("__CARDS__", "\n".join(cards)), encoding="utf-8")
    assert source.read_bytes() == original, "Original ivory version changed"
    for path, previous in previous_versions.items():
        assert path.read_bytes() == previous, f"Existing version changed: {path.name}"
    print(f"Corrected standard: sides 44px; sampled corners {min(corner_thickness):.3f}-{max(corner_thickness):.3f}px.")
    print(f"Opaque border now has {len(border_colors)} RGB color: {border_colors[0].tolist()}. Previously: {len(shaded_colors)} colors.")
    print("Ivory artwork, geometry, and transparency unchanged; previous border archived.")
    print(f"Flat-color alternative verified: {palette}; one RGB color per fully covered region.")
    print(f"Photo-inspired glyph alternative verified: {photo_checks}; all existing PNGs preserved byte-for-byte.")
    print(f"Corrected photo proportions verified against collage guides: {guide_measurements}")
    print(f"Created {OUTPUT / 'index.html'}")


if __name__ == "__main__":
    main()
