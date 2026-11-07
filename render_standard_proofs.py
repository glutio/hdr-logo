import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from clean_logo import HEIGHT, OUTPUT_SCALE, WIDTH, box, draw_currency, draw_number, draw_symbols, fold_mask, mask, points, rounded
from render_variants import distance


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "standard"
BASE_STANDARD_ASSET = "logo-before-proportional-standard.png"
PHOTO_SIZE = (485, 501)
PHOTO_GUIDE_BOUNDS = {
    "four": (120, 59, 204, 173),
    "currency": (108, 229, 213, 330),
}
PHOTO_BOLD_SCALE = 0.96
PHOTO_BOLD_DROP = 18


def measure_manual_layout(standard, reference_name="manual-layout-reference.png"):
    reference = np.asarray(Image.open(OUTPUT / reference_name).convert("RGB"))

    def bounds(selection):
        ys, xs = np.nonzero(selection)
        return np.array([xs.min(), ys.min(), xs.max(), ys.max()])

    def warm_bounds(rgb):
        warm = (rgb[..., 0] > 120) & (rgb[..., 1] > 100) & (
            rgb[..., 0].astype(int) - rgb[..., 2] > 15
        )
        return bounds(warm)

    if reference.shape[:2] == standard.shape[:2]:
        left = reference
        registration_scale, registration_offset = np.ones(2), np.zeros(2)
    else:
        left = reference[:, :reference.shape[1] // 2]
        original_cap, reference_cap = warm_bounds(standard[..., :3]), warm_bounds(left)
        registration_scale = (original_cap[2:] - original_cap[:2]) / (reference_cap[2:] - reference_cap[:2])
        registration_offset = original_cap[:2] - reference_cap[:2] * registration_scale
    height, width = left.shape[:2]
    layout = {}
    for name, renderer, y_limits in [
        ("four", draw_number, (0.08, 0.42)),
        ("currency", draw_currency, (0.42, 0.76)),
    ]:
        x0, x1 = int(width * 0.16), int(width * 0.62)
        y0, y1 = (int(height * value) for value in y_limits)
        reference_bounds = bounds(np.max(left[y0:y1, x0:x1], axis=-1) < 120) + [x0, y0, x0, y0]
        target = reference_bounds * np.tile(registration_scale, 2) + np.tile(registration_offset, 2)
        original = bounds(np.asarray(mask(renderer)) >= 128)
        scale = (target[2:] - target[:2]) / (original[2:] - original[:2])
        offset = (target[:2] - original[:2] * scale) / OUTPUT_SCALE
        layout[name] = {
            "reference_bounds": reference_bounds.tolist(),
            "target_output_bounds": target.tolist(),
            "scale": scale.tolist(),
            "source_offset": offset.tolist(),
        }

    def draw_manual(draw):
        def transform(coords, name):
            return np.asarray(coords) * layout[name]["scale"] + layout[name]["source_offset"]

        draw_number(draw, lambda coords: transform(coords, "four"))
        draw_currency(draw, lambda coords: transform(coords, "currency"), np.mean(layout["currency"]["scale"]))

    coverage = np.asarray(mask(draw_manual))
    for name, y_limits in [("four", (0, 500)), ("currency", (500, coverage.shape[0]))]:
        y0, y1 = y_limits
        actual = bounds(coverage[y0:y1] >= 128) + [0, y0, 0, y0]
        assert np.all(np.abs(actual - layout[name]["target_output_bounds"]) < 2), f"{name} misses the manual layout"
        layout[name]["rendered_output_bounds"] = actual.tolist()
    return draw_manual, layout


def render_stroke_weight(standard, x, y, light, grain, layout,
                         four_width=None, filename="logo-proportional-equal-stroke.png"):
    scale, offset = np.array(layout["four"]["scale"]), np.array(layout["four"]["source_offset"])
    circle_width = 15 * np.mean(layout["currency"]["scale"])
    width = circle_width if four_width is None else four_width / OUTPUT_SCALE
    left, top = np.array([151, 81]) * scale + offset
    right, bottom = np.array([254, 217]) * scale + offset
    stem_right = 235 * scale[0] + offset[0]
    stem_left = stem_right - width
    bar_center = 178.5 * scale[1] + offset[1]
    bar_top, bar_bottom = bar_center - width / 2, bar_center + width / 2
    slope = (bar_top - top) / (stem_left - left)
    counter_edge = bar_top + slope * left + width * np.hypot(1, slope)
    outer = [
        (left, bar_top), (stem_left, top), (stem_right, top),
        (stem_right, bar_top), (right, bar_top), (right, bar_bottom),
        (stem_right, bar_bottom), (stem_right, bottom),
        (stem_left, bottom), (stem_left, bar_bottom), (left, bar_bottom),
    ]
    counter = [
        ((counter_edge - bar_top) / slope, bar_top),
        (stem_left, counter_edge - slope * stem_left),
        (stem_left, bar_top),
    ]

    def draw_equal_number(draw):
        draw.polygon(points(outer), fill=255)
        draw.polygon(points(counter), fill=0)

    def transform_currency(coords):
        return np.asarray(coords) * layout["currency"]["scale"] + layout["currency"]["source_offset"]

    def draw_equal(draw):
        draw_equal_number(draw)
        draw_currency(draw, transform_currency, np.mean(layout["currency"]["scale"]))

    checks = render_photo_standard(
        standard, x, y, light, grain, draw_equal, filename
    )
    parent = np.asarray(Image.open(OUTPUT / "logo-proportional-clean.png"))
    result = np.asarray(Image.open(OUTPUT / filename))
    old_number = np.asarray(mask(lambda draw: draw_number(draw, lambda coords: np.asarray(coords) * scale + offset)))
    new_number = np.asarray(mask(draw_equal_number))
    affected = (old_number > 0) | (new_number > 0)
    assert np.array_equal(result[~affected], parent[~affected]), "Circle or artwork outside the 4 changed"

    def bounds(coverage):
        ys, xs = np.nonzero(coverage >= 128)
        return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    assert bounds(new_number) == bounds(old_number), "The 4 moved or changed overall size"
    assert np.count_nonzero(new_number == 255) < np.count_nonzero(old_number == 255), "The 4 did not become thinner"
    coverage = np.asarray(mask(draw_equal), dtype=np.float64) / 255

    def stroke_width(center, normal):
        steps = np.linspace(-24, 24, 1921)
        positions = np.array(center) * OUTPUT_SCALE + steps[:, None] * np.array(normal)
        ix, iy = np.floor(positions).astype(int).T
        fx, fy = (positions - np.floor(positions)).T
        profile = (
            coverage[iy, ix] * (1 - fx) * (1 - fy)
            + coverage[iy, ix + 1] * fx * (1 - fy)
            + coverage[iy + 1, ix] * (1 - fx) * fy
            + coverage[iy + 1, ix + 1] * fx * fy
        )
        hits = np.flatnonzero(profile >= 0.5)
        first, last = hits[0], hits[-1]
        assert first > 0 and last + 1 < len(profile)
        start = np.interp(0.5, profile[first - 1:first + 1], steps[first - 1:first + 1])
        end = np.interp(0.5, profile[last:last + 2][::-1], steps[last:last + 2][::-1])
        return round(float(end - start), 3)

    diagonal_start = np.array([left, bar_top])
    diagonal_end = np.array([stem_left, top])
    normal = np.array([slope, 1]) / np.hypot(slope, 1)
    diagonal_center = diagonal_start + 0.55 * (diagonal_end - diagonal_start) + normal * width / 2
    measured = {
        "four_vertical": stroke_width(((stem_left + stem_right) / 2, (bar_bottom + bottom) / 2), (1, 0)),
        "four_horizontal": stroke_width(((stem_right + right) / 2, bar_center), (0, 1)),
        "four_diagonal": stroke_width(diagonal_center, normal),
        "circle_horizontal": stroke_width(transform_currency([(152.5, 339)])[0], (1, 0)),
        "circle_vertical": stroke_width(transform_currency([(210, 281.5)])[0], (0, 1)),
    }
    for name, actual in measured.items():
        target = width if name.startswith("four_") else circle_width
        assert abs(actual - target * OUTPUT_SCALE) < 1, f"{name} misses its target stroke width: {actual}"
    return {
        "target_stroke_width_output_pixels": round(float(width * OUTPUT_SCALE), 3),
        "measured_stroke_widths_output_pixels": measured,
        "four_bounds": bounds(new_number),
        "circle_and_keycap_pixel_identical": True,
        "validation": checks,
    }


def photo_coords(coords):
    return [(x * WIDTH / PHOTO_SIZE[0], y * HEIGHT / PHOTO_SIZE[1]) for x, y in coords]


def bold_coords(coords, name):
    bounds = np.array(PHOTO_GUIDE_BOUNDS[name]) * np.tile(np.array([WIDTH, HEIGHT]) / PHOTO_SIZE, 2)
    center = (bounds[:2] + bounds[2:]) / 2
    placed = center + PHOTO_BOLD_SCALE * (np.asarray(coords) - center)
    placed[:, 1] += PHOTO_BOLD_DROP
    return placed


def draw_bold_photo_symbols(draw):
    sx, sy = np.array([WIDTH, HEIGHT]) / PHOTO_SIZE
    left, stem_right = 120 * sx, 186 * sx
    stem_left = stem_right - 18 / PHOTO_BOLD_SCALE
    bar_top = 136 * sy - 17 / (2 * PHOTO_BOLD_SCALE)
    bar_bottom = 136 * sy + 17 / (2 * PHOTO_BOLD_SCALE)
    apex = (179 * sx, 60 * sy)
    slope = (bar_top - apex[1]) / (apex[0] - left)
    counter_edge = bar_top + slope * left + 19 / PHOTO_BOLD_SCALE * np.hypot(1, slope)
    number = [
        (left, bar_top), apex, (stem_right, 59 * sy),
        (stem_right, bar_top), (204 * sx, bar_top), (204 * sx, bar_bottom),
        (stem_right, bar_bottom), (stem_right, 173 * sy),
        (stem_left, 173 * sy), (stem_left, bar_bottom), (left, bar_bottom),
    ]
    counter = [
        ((counter_edge - bar_top) / slope, bar_top),
        (stem_left, counter_edge - slope * stem_left),
        (stem_left, bar_top),
    ]
    draw.polygon(points(bold_coords(number, "four")), fill=255)
    draw.polygon(points(bold_coords(counter, "four")), fill=0)
    ring = np.array(photo_coords([(112, 234), (210, 327)]))
    draw.ellipse(box(bold_coords(ring, "currency").ravel()), fill=255)
    for tip in [
        [(138, 248), (115, 229), (112, 231), (125, 260)],
        [(186, 248), (210, 229), (213, 232), (198, 260)],
        [(127, 305), (108, 327), (112, 330), (139, 316)],
        [(186, 315), (210, 330), (213, 327), (198, 302)],
    ]:
        draw.polygon(points(bold_coords(photo_coords(tip), "currency")), fill=255)
    ring[0] += 15 / PHOTO_BOLD_SCALE
    ring[1] -= 15 / PHOTO_BOLD_SCALE
    draw.ellipse(box(bold_coords(ring, "currency").ravel()), fill=0)


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
    assert 0 < new_ink < old_ink, "Glyph variant has invalid ink coverage"
    Image.fromarray(pixels).save(OUTPUT / filename)
    return {
        "original_opaque_ink_pixels": old_ink,
        "variant_opaque_ink_pixels": new_ink,
        "changed_pixels": int(np.count_nonzero(np.any(pixels != standard, axis=-1))),
        "outside_glyphs_unchanged": True,
        "silhouette_unchanged": True,
    }


def validate_photo_proportions(draw_glyphs=draw_photo_proportions, scale=1, drop=0):
    coverage = np.asarray(mask(draw_glyphs))
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
        center = np.tile((expected[:2] + expected[2:]) / 2, 2)
        expected = center + scale * (expected - center)
        expected[[1, 3]] += drop * OUTPUT_SCALE
        assert np.all(np.abs(actual - expected) < 2), f"{name} misses the collage guides: {actual} vs {expected}"
        measurements[name] = {
            "photo_bounds": list(PHOTO_GUIDE_BOUNDS[name]),
            "expected_output_bounds": expected.round(3).tolist(),
            "rendered_output_bounds": actual.tolist(),
        }
    return measurements


def validate_bold_strokes():
    original = np.asarray(mask(draw_symbols))
    bold = np.asarray(mask(draw_bold_photo_symbols))

    def first_width(profile):
        indices = np.flatnonzero(profile >= 128)
        return len(np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)[0])

    stem_y = round(bold_coords(photo_coords([(186, 158)]), "four")[0, 1] * OUTPUT_SCALE)
    bar_x = round(bold_coords(photo_coords([(198, 136)]), "four")[0, 0] * OUTPUT_SCALE)
    ring_x, ring_y = np.round(
        bold_coords(photo_coords([(161, 280.5)]), "currency")[0] * OUTPUT_SCALE
    ).astype(int)
    split = 250 * OUTPUT_SCALE
    baseline = {
        "four_stem": first_width(original[200 * OUTPUT_SCALE]),
        "four_bar": first_width(original[:, 249 * OUTPUT_SCALE]),
        "ring_horizontal": first_width(original[339 * OUTPUT_SCALE]),
        "ring_vertical": first_width(original[split:, 210 * OUTPUT_SCALE]),
    }
    measured = {
        "four_stem": first_width(bold[stem_y]),
        "four_bar": first_width(bold[:, bar_x]),
        "ring_horizontal": first_width(bold[ring_y]),
        "ring_vertical": first_width(bold[split:, ring_x]),
    }
    for name, width in measured.items():
        assert abs(width - baseline[name]) <= 1, f"{name} does not retain the standard's thickness: {width} vs {baseline[name]}"
    return {"standard_output_pixels": baseline, "variant_output_pixels": measured}


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


def create_bold_proof():
    proof = Image.open(OUTPUT / "photo-proportions-reference.png").convert("RGB").crop((0, 0, 1037, 560))
    proof.paste("black", (0, 0, 511, 560))
    variant = Image.open(OUTPUT / "logo-photo-proportions-bold.png").resize(PHOTO_SIZE, Image.Resampling.LANCZOS)
    proof.paste(variant, (12, 11), mask=variant.getchannel("A"))
    draw = ImageDraw.Draw(proof)
    for name, bounds in PHOTO_GUIDE_BOUNDS.items():
        source = np.array(photo_coords([bounds[:2], bounds[2:]]))
        placed = bold_coords(source, name) * np.array(PHOTO_SIZE) / np.array([WIDTH, HEIGHT])
        left, top, right, bottom = (placed + [12, 11]).ravel()
        draw.rectangle((left, top, right, bottom), outline=(255, 0, 0))
    draw.text((12, 530), "Bold strokes / smaller and lowered for rim clearance", fill="white", font_size=16)
    draw.text((523, 530), "Photo / original guides include blur", fill="white", font_size=16)
    proof.save(OUTPUT / "photo-proportions-bold-proof.png")


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
    base_path = OUTPUT / BASE_STANDARD_ASSET
    previous_path = base_path if base_path.exists() else OUTPUT / "logo.png"
    previous_standard = previous_path.read_bytes() if previous_path.exists() else None
    previous_versions = {path: path.read_bytes() for path in OUTPUT.glob("*.png")
                         if path.name != "logo.png"}
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
    Image.fromarray(corrected).save(base_path)

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
    base_bytes = base_path.read_bytes()
    if previous_standard is not None:
        assert base_bytes == previous_standard, "Archived previous standard changed"
    palette, core_counts = render_flat_standard(corrected)
    flat_bytes = (OUTPUT / "logo-flat-color.png").read_bytes()
    flat_selection = {
        "status": "alternative",
        "asset": "logo-flat-color.png",
        "source": BASE_STANDARD_ASSET,
        "source_sha256": hashlib.sha256(base_bytes).hexdigest(),
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
        "source": BASE_STANDARD_ASSET,
        "source_sha256": hashlib.sha256(base_bytes).hexdigest(),
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
        "source": BASE_STANDARD_ASSET,
        "source_sha256": hashlib.sha256(base_bytes).hexdigest(),
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
    bold_checks = render_photo_standard(
        corrected, x, y, light, grain, draw_bold_photo_symbols, "logo-photo-proportions-bold.png"
    )
    assert bold_checks["variant_opaque_ink_pixels"] > proportion_checks["variant_opaque_ink_pixels"], "Bold variant did not restore heavier printing"
    bold_guides = validate_photo_proportions(draw_bold_photo_symbols, PHOTO_BOLD_SCALE, PHOTO_BOLD_DROP)
    stroke_measurements = validate_bold_strokes()
    bold_bytes = (OUTPUT / "logo-photo-proportions-bold.png").read_bytes()
    bold_selection = {
        "status": "alternative",
        "asset": "logo-photo-proportions-bold.png",
        "source": BASE_STANDARD_ASSET,
        "source_sha256": hashlib.sha256(base_bytes).hexdigest(),
        "sha256": hashlib.sha256(bold_bytes).hexdigest(),
        "reference": "photo-proportions-reference.png",
        "glyph_scale_from_guide_bounds": PHOTO_BOLD_SCALE,
        "group_vertical_offset_output_pixels": PHOTO_BOLD_DROP * OUTPUT_SCALE,
        "changes": "Restore standard stroke widths; reduce each measured glyph by 4%; lower both together for pale-rim clearance.",
        "guides": bold_guides,
        "stroke_width_validation": stroke_measurements,
        "validation": bold_checks,
    }
    (OUTPUT / "photo-proportions-bold-selection.json").write_text(
        json.dumps(bold_selection, indent=2) + "\n", encoding="utf-8"
    )
    create_bold_proof()
    draw_manual, manual_layout = measure_manual_layout(corrected)
    manual_checks = render_photo_standard(
        corrected, x, y, light, grain, draw_manual, "logo-manual-layout.png"
    )
    manual_bytes = (OUTPUT / "logo-manual-layout.png").read_bytes()
    manual_selection = {
        "status": "alternative",
        "asset": "logo-manual-layout.png",
        "source": BASE_STANDARD_ASSET,
        "source_sha256": hashlib.sha256(base_bytes).hexdigest(),
        "sha256": hashlib.sha256(manual_bytes).hexdigest(),
        "reference": "manual-layout-reference.png",
        "layout": manual_layout,
        "generation": "Original glyph primitives transformed before 8x supersampling; glyph rasters are not resized.",
        "validation": manual_checks,
    }
    (OUTPUT / "manual-layout-selection.json").write_text(
        json.dumps(manual_selection, indent=2) + "\n", encoding="utf-8"
    )
    draw_proportional, proportional_layout = measure_manual_layout(corrected, "logo-proportional.png")
    proportional_checks = render_photo_standard(
        corrected, x, y, light, grain, draw_proportional, "logo-proportional-clean.png"
    )
    proportional_bytes = (OUTPUT / "logo-proportional-clean.png").read_bytes()
    proportional_selection = {
        "status": "alternative",
        "asset": "logo-proportional-clean.png",
        "source": BASE_STANDARD_ASSET,
        "source_sha256": hashlib.sha256(base_bytes).hexdigest(),
        "sha256": hashlib.sha256(proportional_bytes).hexdigest(),
        "reference": "logo-proportional.png",
        "reference_sha256": hashlib.sha256((OUTPUT / "logo-proportional.png").read_bytes()).hexdigest(),
        "layout": proportional_layout,
        "generation": "Original glyph primitives regenerated at the exact native reference layout before 8x supersampling.",
        "validation": proportional_checks,
    }
    (OUTPUT / "proportional-clean-selection.json").write_text(
        json.dumps(proportional_selection, indent=2) + "\n", encoding="utf-8"
    )
    equal_checks = render_stroke_weight(corrected, x, y, light, grain, proportional_layout)
    equal_bytes = (OUTPUT / "logo-proportional-equal-stroke.png").read_bytes()
    equal_selection = {
        "status": "alternative",
        "asset": "logo-proportional-equal-stroke.png",
        "source": "logo-proportional-clean.png",
        "source_sha256": hashlib.sha256(proportional_bytes).hexdigest(),
        "sha256": hashlib.sha256(equal_bytes).hexdigest(),
        "changes": "Only thin the 4's vertical, horizontal, and diagonal strokes to the unchanged circle's width.",
        "checks": equal_checks,
    }
    (OUTPUT / "proportional-equal-stroke-selection.json").write_text(
        json.dumps(equal_selection, indent=2) + "\n", encoding="utf-8"
    )
    balanced_checks = render_stroke_weight(
        corrected, x, y, light, grain, proportional_layout,
        four_width=28.5, filename="logo-proportional-balanced.png",
    )
    assert (
        equal_checks["validation"]["variant_opaque_ink_pixels"]
        < balanced_checks["validation"]["variant_opaque_ink_pixels"]
        < proportional_checks["variant_opaque_ink_pixels"]
    ), "Intermediate weight is not between the equal-stroke and original versions"
    balanced_bytes = (OUTPUT / "logo-proportional-balanced.png").read_bytes()
    balanced_selection = {
        "status": "previous-standard",
        "asset": "logo-proportional-balanced.png",
        "release": "../logo-final-2026-10-31/",
        "source": "logo-proportional-clean.png",
        "source_sha256": hashlib.sha256(proportional_bytes).hexdigest(),
        "sha256": hashlib.sha256(balanced_bytes).hexdigest(),
        "changes": "Set all three 4 strokes to 28.5px; keep the circle at about 26px, with unchanged glyph bounds and placement.",
        "checks": balanced_checks,
    }
    (OUTPUT / "proportional-balanced-selection.json").write_text(
        json.dumps(balanced_selection, indent=2) + "\n", encoding="utf-8"
    )
    release = ROOT / "logo-final-2026-11-06"
    approved = json.loads((release / "logo.json").read_text(encoding="utf-8"))
    canonical_bytes = (release / "assets" / "logo.png").read_bytes()
    assert hashlib.sha256(canonical_bytes).hexdigest() == approved["standard_sha256"]
    (OUTPUT / "logo.png").write_bytes(canonical_bytes)
    assert (OUTPUT / "logo.png").read_bytes() == canonical_bytes, "Canonical PNG differs from the accepted version"
    selection = {
        "status": "standard",
        "asset": "logo.png",
        "source": "../logo-final-2026-11-06/assets/logo.png",
        "sha256": hashlib.sha256(canonical_bytes).hexdigest(),
        "correction": "Approved 115% glyphs with scaled strokes and a 117px gap; current ivory material unchanged.",
        "release_date": approved["snapshot_date"],
        "dimensions": [width, height],
        "four_stroke_target_pixels": approved["four"]["stroke_width_output_pixels"],
        "glyph_gap_pixels": approved["glyph_gap_output_pixels"],
        "four_bounds": approved["four"]["bounds"],
        "currency_bounds": approved["currency"]["bounds"],
        "border_color": "#191b14",
        "opaque_border_unique_colors": len(border_colors),
        "straight_border_pixels": 44,
        "rendered_corner_measurements_pixels": corner_thickness,
        "previous_standard": "logo-proportional-balanced.png",
        "previous_standard_sha256": hashlib.sha256(balanced_bytes).hexdigest(),
        "border_alternatives": "Earlier slimmer and neutral-color proofs remain available.",
    }
    (OUTPUT / "selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    cards = []
    for title, filename, description in [
        ("Current standard / 115% ivory", "logo.png", "Approved November 6: 115% glyphs, scaled strokes, a 117px gap, current ivory, and the solid 44px border."),
        ("October 31 standard / preserved", "logo-proportional-balanced.png", "The preceding proportional standard at 100% glyph size, preserved byte-for-byte."),
        ("Earlier original-layout standard / preserved", BASE_STANDARD_ASSET, "The earlier standard, preserved with its original larger glyphs and heavier 4."),
        ("Previous / equal widths", "logo-proportional-equal-stroke.png", "The previous study: both glyphs at about 26px stroke width."),
        ("Before / thicker 4", "logo-proportional-clean.png", "Your exact proportional layout with the original stroke weights: the 4 is heavier than the circle."),
    ]:
        image_src = f"{filename}?v={hashlib.sha256((OUTPUT / filename).read_bytes()).hexdigest()[:12]}"
        sizes = "".join(
            f'<figure><img src="{image_src}" style="width:{size}px" alt="{title} at {size} pixels"><figcaption>{size}px</figcaption></figure>'
            for size in [16, 32, 48, 96]
        )
        cards.append(f"""<article><h2>{title}</h2><img class="large" src="{image_src}" width="1078" height="1142" alt="{title}"><p>{description}</p><div class="sizes">{sizes}</div><a href="{image_src}" download>Download PNG</a></article>""")
    page = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>The standard / proportional ivory logo</title><style>
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
<h1>The standard: proportional ivory.</h1>
<p>The border now has both even geometry and even color: 44px wide, with one solid <code>#191b14</code> ink color matching the symbols. No lighting gradient remains on the border.</p>
<p>The approved November 6 standard uses 115% glyphs with scaled strokes and a 117px vertical gap. The current ivory material is unchanged. The canonical download is <a href="logo.png?v=__STANDARD_REVISION__">logo.png</a>. <a href="../">The new four-image set and ZIP</a>. The October 31 standard and all earlier versions remain preserved.</p>
<button type="button" onclick="document.body.classList.toggle('light')">Toggle light / dark background</button>
<main class="grid">__CARDS__</main>
<figure class="reference"><img src="manual-layout-reference.png" width="1374" height="734" alt="Your earlier comparison: manually adjusted glyph layout on the left and the previous standard on the right"><figcaption>Your untouched earlier comparison: adjusted layout on the left, previous standard on the right.</figcaption></figure>
<footer><p>Antialiased edge pixels blend normally with neighboring regions or the transparent background. Earlier versions: <a href="../variants/ivory-keycap.png">original ivory</a> | <a href="logo-proportional.png">your saved raster reference</a> | <a href="logo-flat-color.png">solid-color alternative</a> | <a href="logo-manual-layout.png">earlier screenshot reconstruction</a> | <a href="logo-photo-glyphs.png">first photo attempt</a> | <a href="logo-photo-proportions.png">thin measured correction</a> | <a href="logo-photo-proportions-bold.png">earlier bold photo study</a> | <a href="logo-before-flat-border.png">even-width shaded border</a> | <a href="rim-slimmer.png">slimmer border</a> | <a href="rim-neutral.png">neutral-black border</a>. <a href="selection.json">Border measurements</a> | <a href="proportional-balanced-selection.json">Intermediate stroke-weight and preservation checks</a> | <a href="proportional-equal-stroke-selection.json">Equal-stroke study checks</a> | <a href="proportional-clean-selection.json">Native proportional layout</a> | <a href="../variants/">All previous versions</a> | <a href="../backlight/">Earlier HDR experiment</a></p></footer>
</body></html>"""
    page = page.replace('href="logo-flat-color.png">solid-color alternative',
                        'href="../logo-final-2026-11-06/assets/logo-flat-color.png">current solid-color alternative')
    page = page.replace("__CARDS__", "\n".join(cards)).replace("__STANDARD_REVISION__", selection["sha256"][:12])
    (OUTPUT / "index.html").write_text(page, encoding="utf-8")
    assert source.read_bytes() == original, "Original ivory version changed"
    for path, previous in previous_versions.items():
        assert path.read_bytes() == previous, f"Existing version changed: {path.name}"
    print(f"Corrected standard: sides 44px; sampled corners {min(corner_thickness):.3f}-{max(corner_thickness):.3f}px.")
    print(f"Opaque border now has {len(border_colors)} RGB color: {border_colors[0].tolist()}. Previously: {len(shaded_colors)} colors.")
    print("Ivory artwork, geometry, and transparency unchanged; previous border archived.")
    print(f"Flat-color alternative verified: {palette}; one RGB color per fully covered region.")
    print(f"Photo-inspired glyph alternative verified: {photo_checks}; all existing PNGs preserved byte-for-byte.")
    print(f"Corrected photo proportions verified against collage guides: {guide_measurements}")
    print(f"Bold alternative verified: {stroke_measurements}; glyph scale {PHOTO_BOLD_SCALE}, group drop {PHOTO_BOLD_DROP * OUTPUT_SCALE}px.")
    print(f"Manual layout regenerated from original geometry and verified: {manual_layout}")
    print(f"Saved native proportional layout regenerated and verified: {proportional_layout}")
    print(f"Equal-stroke alternative verified: {equal_checks}")
    print(f"Intermediate-weight alternative verified: {balanced_checks}")
    print(f"Accepted standard saved as logo.png; previous standard archived as {BASE_STANDARD_ASSET}.")
    print(f"Created {OUTPUT / 'index.html'}")


if __name__ == "__main__":
    main()
