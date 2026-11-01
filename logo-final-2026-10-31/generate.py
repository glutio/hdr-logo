import argparse
import hashlib
import io
import json
import platform
from pathlib import Path

import imagecodecs
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "logo.json").read_text(encoding="utf-8"))
WIDTH, HEIGHT = CONFIG["canvas"]
DRAW_SCALE, OUTPUT_SCALE = CONFIG["draw_scale"], CONFIG["output_scale"]
SIZE = (WIDTH * OUTPUT_SCALE, HEIGHT * OUTPUT_SCALE)
INK = tuple(CONFIG["ink"])
ASSETS = (
    "logo.png",
    "logo-flat-color.png",
    "logo-inner-rim-hdr.jpg",
    "logo-inner-rim-symbols-hdr.jpg",
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def points(coords):
    return [(round(x * DRAW_SCALE), round(y * DRAW_SCALE)) for x, y in coords]


def box(coords):
    return tuple(round(value * DRAW_SCALE) for value in coords)


def mask(draw_shape):
    image = Image.new("L", (WIDTH * DRAW_SCALE, HEIGHT * DRAW_SCALE))
    draw_shape(ImageDraw.Draw(image))
    return image.resize(SIZE, Image.Resampling.LANCZOS)


def rounded(coords, radius):
    return mask(lambda draw: draw.rounded_rectangle(box(coords), radius * DRAW_SCALE, fill=255))


def fold_mask(inner):
    t = np.linspace(0, 1, 160)[:, None]
    bottom = ((1 - t) ** 3 * [394, 460] + 3 * (1 - t) ** 2 * t * [434, 458]
              + 3 * (1 - t) * t ** 2 * [441, 473] + t ** 3 * [459, 491])
    top = ((1 - t) ** 3 * [394, 34] + 3 * (1 - t) ** 2 * t * [430, 34]
           + 3 * (1 - t) * t ** 2 * [452, 34] + t ** 3 * [475, 28])
    fold = mask(lambda draw: draw.polygon(
        points([*top.tolist(), (539, 28), (539, 571), (494, 532), *bottom[::-1].tolist()]),
        fill=255,
    ))
    return Image.fromarray(np.minimum(np.asarray(fold), np.asarray(inner)))


def distance(x, y, bounds, radius):
    left, top, right, bottom = bounds
    qx = np.abs(x - (left + right) / 2) - (right - left) / 2 + radius
    qy = np.abs(y - (top + bottom) / 2) - (bottom - top) / 2 + radius
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - radius


def texture(color, shading):
    rgb = np.clip(np.array(color) + shading[..., None], 0, 255)
    return Image.fromarray(np.round(rgb).astype(np.uint8)).convert("RGBA")


def draw_four(draw):
    scale, offset = np.array(CONFIG["four"]["scale"]), np.array(CONFIG["four"]["offset"])
    width = CONFIG["four"]["stroke_width_output_pixels"] / OUTPUT_SCALE
    left, top = np.array([151, 81]) * scale + offset
    right, bottom = np.array([254, 217]) * scale + offset
    stem_right = 235 * scale[0] + offset[0]
    stem_left = stem_right - width
    center = 178.5 * scale[1] + offset[1]
    bar_top, bar_bottom = center - width / 2, center + width / 2
    slope = (bar_top - top) / (stem_left - left)
    counter_edge = bar_top + slope * left + width * np.hypot(1, slope)
    outer = [
        (left, bar_top), (stem_left, top), (stem_right, top),
        (stem_right, bar_top), (right, bar_top), (right, bar_bottom),
        (stem_right, bar_bottom), (stem_right, bottom), (stem_left, bottom),
        (stem_left, bar_bottom), (left, bar_bottom),
    ]
    counter = [
        ((counter_edge - bar_top) / slope, bar_top),
        (stem_left, counter_edge - slope * stem_left), (stem_left, bar_top),
    ]
    draw.polygon(points(outer), fill=255)
    draw.polygon(points(counter), fill=0)


def draw_currency(draw):
    scale, offset = np.array(CONFIG["currency"]["scale"]), np.array(CONFIG["currency"]["offset"])

    def transform(coords):
        return np.asarray(coords) * scale + offset

    outer, inner = transform([(145, 274), (275, 404)]), transform([(160, 289), (260, 389)])
    draw.ellipse(box(outer.ravel()), fill=255)
    for start, end in [((150, 284), (270, 395)), ((269, 284), (150, 395))]:
        draw.line(points(transform([start, end])), fill=255, width=round(15 * DRAW_SCALE * np.mean(scale)))
        for px, py in (start, end):
            cap = transform([(px - 7.5, py - 7.5), (px + 7.5, py + 7.5)])
            draw.ellipse(box(cap.ravel()), fill=255)
    draw.ellipse(box(inner.ravel()), fill=0)


def draw_symbols(draw):
    draw_four(draw)
    draw_currency(draw)


def layers():
    shoulder = rounded((23, 23, 515, 548), 74)
    return [
        ("border", rounded((1, 1, 537, 570), 96)),
        ("shoulder", shoulder),
        ("fold", fold_mask(shoulder)),
        ("rim", rounded((65, 34, 479, 460), 85)),
        ("face", rounded((81, 51, 463, 444), 69)),
        ("symbols", mask(draw_symbols)),
    ]


def render_standard(regions):
    y, x = np.mgrid[:SIZE[1], :SIZE[0]].astype(np.float32) / OUTPUT_SCALE
    nx, ny = x / WIDTH, y / HEIGHT
    original_shoulder = rounded((24, 23, 515, 548), 69)
    edge = distance(x, y, (24, 23, 515, 548), 69)  # Preserve the approved material's lighting field.
    dy, dx = np.gradient(edge)
    direction = -0.65 * dx - 0.75 * dy
    bevel = direction * np.exp(-np.abs(edge) / 24)
    light = np.exp(-((nx - 0.28) / 0.65) ** 2 - ((ny - 0.20) / 0.70) ** 2)
    grain = np.random.default_rng(4).normal(0, 0.17, edge.shape).astype(np.float32)
    face_roll = np.exp(-np.abs(distance(x, y, (81, 51, 463, 444), 69)) / 16)
    frame_shading = 4 * light - 3 * ny + 4 * bevel
    shoulder_shading = 1.25 * (18 * light - 21 * ny + 24 * bevel) + grain
    fold_shading = 1.25 * (12 * light - 13 * ny + 12 * bevel) + grain
    rim_shading = 1.25 * (8 * light - 8 * ny + 3 * direction)
    face_shading = 1.25 * (14 * light - 10 * ny - 7 * face_roll) + grain
    image = Image.new("RGBA", SIZE, (0, 0, 0, 0))
    image.paste(texture(INK, frame_shading), mask=regions["border"])
    image.paste(texture((195, 176, 138), shoulder_shading), mask=original_shoulder)
    image.paste(texture((226, 211, 178), fold_shading), mask=fold_mask(original_shoulder))
    image.paste(texture((239, 230, 207), rim_shading), mask=regions["rim"])
    image.paste(texture((229, 215, 185), face_shading), mask=regions["face"])
    pixels = np.asarray(image).copy()
    old = np.asarray(original_shoulder, dtype=np.float32) / 255
    shoulder = np.asarray(regions["shoulder"], dtype=np.float32) / 255
    fold = np.asarray(regions["fold"], dtype=np.float32) / 255
    fold = np.clip(fold / np.maximum(shoulder, 1e-6), 0, 1)
    shoulder_finish = np.array([195, 176, 138]) + shoulder_shading[..., None]
    fold_finish = np.array([226, 211, 178]) + fold_shading[..., None]
    finish = shoulder_finish + fold[..., None] * (fold_finish - shoulder_finish)
    frame_finish = np.array(INK) + frame_shading[..., None]
    changed = old != shoulder
    perimeter = frame_finish + shoulder[..., None] * (finish - frame_finish)
    pixels[..., :3][changed] = np.round(np.clip(perimeter[changed], 0, 255)).astype(np.uint8)
    border = (np.asarray(regions["border"]) > 0) & (shoulder < 1)
    perimeter = np.array(INK) + shoulder[..., None] * (finish - np.array(INK))
    pixels[..., :3][border] = np.round(np.clip(perimeter[border], 0, 255)).astype(np.uint8)
    image = Image.fromarray(pixels)
    symbols = np.asarray(regions["symbols"])
    assert np.all(np.asarray(regions["face"])[symbols > 0] == 255), "Symbols extend outside the face"
    image.paste((*INK, 255), mask=regions["symbols"])
    return image


def render_flat(standard, region_layers):
    pixels = np.asarray(standard).copy()
    pixels[..., :3][pixels[..., 3] > 0] = INK
    image = Image.fromarray(pixels)
    palette = {}
    coverages = [np.asarray(layer) for _, layer in region_layers]
    for index, (name, layer) in enumerate(region_layers):
        core = coverages[index] == 255
        for higher in coverages[index + 1:]:
            core &= higher == 0
        assert np.any(core), f"No solid region for {name}"
        color = INK if name in ("border", "symbols") else tuple(
            np.round(np.median(np.asarray(standard)[..., :3][core], axis=0)).astype(int).tolist()
        )
        palette[name] = list(color)
        if index:
            image.paste((*color, 255), mask=layer)
    return image, palette


def to_linear(rgb):
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def blur(coverage, radius):
    image = Image.fromarray(np.round(coverage * 255).astype(np.uint8))
    return np.asarray(image.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32) / 255


def sdr_rgb(standard):
    background = Image.new("RGBA", SIZE, (*CONFIG["hdr"]["background"], 255))
    return np.asarray(Image.alpha_composite(background, standard).convert("RGB"))


def hdr_masks(regions, symbols_lit):
    rim = np.clip(
        np.asarray(regions["rim"], dtype=np.float32) / 255
        - np.asarray(regions["face"], dtype=np.float32) / 255, 0, 1
    )
    symbols = np.asarray(regions["symbols"], dtype=np.float32) / 255
    lit_symbols = symbols if symbols_lit else np.zeros_like(symbols)
    luminous = np.maximum(rim, lit_symbols)
    cap = np.asarray(regions["shoulder"], dtype=np.float32) / 255
    halo = (0.16 * blur(lit_symbols, 5) + 0.14 * blur(rim, 8)) * (1 - luminous) * cap
    if not symbols_lit:
        halo *= 1 - symbols
    return rim, lit_symbols, luminous, halo


def encode_hdr(standard, regions, symbols_lit):
    settings = CONFIG["hdr"]
    rgb8 = sdr_rgb(standard)
    linear = to_linear(rgb8.astype(np.float32) / 255)
    rim, symbols, luminous, halo = hdr_masks(regions, symbols_lit)
    warmth = np.array(settings["linear_warmth"], dtype=np.float32)
    hdr_rgb = linear * (1 - luminous[..., None])
    hdr_rgb += (settings["rim_linear_intensity"] * rim
                + settings["symbols_linear_intensity"] * symbols)[..., None] * warmth
    hdr_rgb += halo[..., None] * warmth
    hdr = np.ones((*rgb8.shape[:2], 4), dtype=np.float16)
    hdr[..., :3] = hdr_rgb.astype(np.float16)
    sdr = np.full(hdr.shape, 255, dtype=np.uint8)
    sdr[..., :3] = rgb8
    return imagecodecs.ultrahdr_encode(
        hdr, level=100, sdr=sdr, gamut=imagecodecs.ULTRAHDR.CG.BT_709,
        transfer=imagecodecs.ULTRAHDR.CT.LINEAR, scale=1, gainmap_level=100,
        boostmin=1.0, boostmax=512.0 if symbols_lit else 5.0,
        nits=settings["target_peak_nits"],
    )


def validate(output, region_layers, palette):
    assert sha256((output / ASSETS[0]).read_bytes()) == CONFIG["standard_sha256"], "Approved PNG changed"
    standard = np.asarray(Image.open(output / ASSETS[0]).convert("RGBA"))
    flat = np.asarray(Image.open(output / ASSETS[1]).convert("RGBA"))
    assert standard.shape == (SIZE[1], SIZE[0], 4)
    assert flat.shape == standard.shape and np.array_equal(flat[..., 3], standard[..., 3])
    coverages = [np.asarray(layer) for _, layer in region_layers]
    for index, (name, _) in enumerate(region_layers):
        core = coverages[index] == 255
        for higher in coverages[index + 1:]:
            core &= higher == 0
        assert np.array_equal(np.unique(flat[..., :3][core], axis=0), np.array(palette[name])[None, :]), name
    symbols = coverages[-1]
    assert np.array_equal(flat[symbols == 255], standard[symbols == 255]), "Flat ink changed"
    assert np.array_equal(flat[coverages[1] == 0], standard[coverages[1] == 0]), "Outer border changed"
    for name, draw in [("four", draw_four), ("currency", draw_currency)]:
        ys, xs = np.nonzero(np.asarray(mask(draw)) >= 128)
        assert [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] == CONFIG[name]["bounds"], name
    outline, shoulder = coverages[0], coverages[1]
    for outer, inner in [
        (outline[SIZE[1] // 2], shoulder[SIZE[1] // 2]),
        (outline[SIZE[1] // 2][::-1], shoulder[SIZE[1] // 2][::-1]),
        (outline[:, SIZE[0] // 2], shoulder[:, SIZE[0] // 2]),
        (outline[:, SIZE[0] // 2][::-1], shoulder[:, SIZE[0] // 2][::-1]),
    ]:
        assert np.flatnonzero(inner >= 128)[0] - np.flatnonzero(outer >= 128)[0] == 44
    regions = dict(region_layers)
    rim, _, _, _ = hdr_masks(regions, False)
    border_core = (outline == 255) & (shoulder == 0)
    y, x = np.mgrid[:SIZE[1], :SIZE[0]]
    plain_face = (x > SIZE[0] * 0.56) & (x < SIZE[0] * 0.74) & (y > SIZE[1] * 0.25) & (y < SIZE[1] * 0.65)
    fallback_rgb = sdr_rgb(Image.fromarray(standard))
    reports = {}
    for filename, symbols_lit in [(ASSETS[2], False), (ASSETS[3], True)]:
        saved = (output / filename).read_bytes()
        assert saved.startswith(b"\xff\xd8"), f"Not a JPEG container: {filename}"
        assert imagecodecs.ultrahdr_check(saved), f"No valid gain map: {filename}"
        with Image.open(io.BytesIO(saved)) as fallback:
            assert fallback.format in ("JPEG", "MPO") and fallback.size == SIZE
            decoded_rgb = np.asarray(fallback.convert("RGB"))
        error = np.abs(decoded_rgb.astype(float) - fallback_rgb.astype(float))
        assert float(np.mean(error)) < 1.5, f"SDR fallback differs from standard: {filename}"
        readings = []
        for headroom in [1, 1.5, 2, 3, 5]:
            decoded = imagecodecs.ultrahdr_decode(saved, dtype=np.float16, boost=headroom)
            assert decoded.shape == standard.shape and np.isfinite(decoded).all()
            luminance = decoded[..., :3].astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722])
            readings.append({
                "headroom": headroom,
                "rim": float(np.median(luminance[rim > 0.99])),
                "symbols": float(np.median(luminance[symbols == 255])),
                "face": float(np.median(luminance[plain_face])),
                "outer_border": float(np.median(luminance[border_core])),
            })
        first, last = readings[0], readings[-1]
        assert first["symbols"] < 0.03, "SDR symbols are no longer dark"
        assert last["rim"] > 2.5, f"Inner rim did not illuminate: {filename}"
        assert all(a["rim"] <= b["rim"] for a, b in zip(readings, readings[1:]))
        for reading in readings:
            assert abs(reading["face"] / first["face"] - 1) < 0.03, "Plain face changed brightness"
            assert abs(reading["outer_border"] / first["outer_border"] - 1) < 0.05, "Outer outline lit up"
        if symbols_lit:
            assert last["symbols"] > 2.5, "Symbols did not illuminate"
            assert all(a["symbols"] <= b["symbols"] for a, b in zip(readings, readings[1:]))
        else:
            assert abs(last["symbols"] / first["symbols"] - 1) < 0.05, "Rim-only mode lit the symbols"
        reports[filename] = {"sdr_mean_absolute_error": float(np.mean(error)), "readings": readings}
    return reports


def main():
    parser = argparse.ArgumentParser(description="Generate and verify the four final ivory logo assets.")
    parser.add_argument("--output", type=Path, default=ROOT / "assets")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    region_layers = layers()
    regions = dict(region_layers)
    standard = render_standard(regions)
    flat, palette = render_flat(standard, region_layers)
    if not args.validate_only:
        args.output.mkdir(parents=True, exist_ok=True)
        buffer = io.BytesIO()
        standard.save(buffer, format="PNG")
        assert sha256(buffer.getvalue()) == CONFIG["standard_sha256"], "Procedural render differs from the approved standard"
        (args.output / ASSETS[0]).write_bytes(buffer.getvalue())
        flat.save(args.output / ASSETS[1])
        for filename, symbols_lit in [(ASSETS[2], False), (ASSETS[3], True)]:
            (args.output / filename).write_bytes(encode_hdr(standard, regions, symbols_lit))
    reports = validate(args.output, region_layers, palette)
    if not args.validate_only:
        manifest = {
            "snapshot_date": CONFIG["snapshot_date"],
            "dimensions": list(SIZE),
            "standard_sha256": CONFIG["standard_sha256"],
            "environment": {
                "python": platform.python_version(), "numpy": np.__version__,
                "pillow": PIL.__version__, "imagecodecs": imagecodecs.__version__,
                "ultrahdr": imagecodecs.ultrahdr_version(),
            },
            "flat_palette_rgb": palette,
            "hdr": {"settings": CONFIG["hdr"], "decoded_validation": reports},
            "assets_sha256": {name: sha256((args.output / name).read_bytes()) for name in ASSETS},
            "source_sha256": {
                name: sha256((ROOT / name).read_bytes())
                for name in ["generate.py", "logo.json", "requirements.txt", "AGENTS.md"]
            },
        }
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    else:
        manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
        for name, expected in manifest["assets_sha256"].items():
            assert sha256((args.output / name).read_bytes()) == expected, f"Asset checksum mismatch: {name}"
        for name, expected in manifest["source_sha256"].items():
            assert sha256((ROOT / name).read_bytes()) == expected, f"Source checksum mismatch: {name}"
    print(f"Verified approved standard, flat regions and geometry, and both native HDR JPEGs: {args.output}")
    print(json.dumps({"flat_palette_rgb": palette, "hdr_decoded_validation": reports}, indent=2))


if __name__ == "__main__":
    main()
