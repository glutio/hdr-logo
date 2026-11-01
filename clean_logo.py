import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


WIDTH, HEIGHT = 539, 571
DRAW_SCALE, OUTPUT_SCALE = 8, 2


def mask(draw_shape):
    image = Image.new("L", (WIDTH * DRAW_SCALE, HEIGHT * DRAW_SCALE))
    draw_shape(ImageDraw.Draw(image))
    return image.resize((WIDTH * OUTPUT_SCALE, HEIGHT * OUTPUT_SCALE), Image.Resampling.LANCZOS)


def box(coords):
    return tuple(round(value * DRAW_SCALE) for value in coords)


def rounded(coords, radius):
    return mask(lambda draw: draw.rounded_rectangle(box(coords), radius * DRAW_SCALE, fill=255))


def points(coords):
    return [(round(x * DRAW_SCALE), round(y * DRAW_SCALE)) for x, y in coords]


def fitted_finish(source, samples, nx, ny):
    sy, sx = np.mgrid[:HEIGHT, :WIDTH]
    sx, sy = sx / WIDTH, sy / HEIGHT
    basis = np.stack([np.ones_like(sx), sx, sy, sx * sy, sx * sx, sy * sy], axis=-1)
    coefficients = np.linalg.lstsq(basis[samples], source[samples], rcond=None)[0]
    target = np.stack([np.ones_like(nx), nx, ny, nx * ny, nx * nx, ny * ny], axis=-1)
    return np.clip(target @ coefficients, 0, 255)


def fold_mask(inner_gold):
    t = np.linspace(0, 1, 160)[:, None]
    curve = ((1 - t) ** 3 * [394, 460] + 3 * (1 - t) ** 2 * t * [434, 458]
             + 3 * (1 - t) * t ** 2 * [441, 473] + t ** 3 * [459, 491])
    top_curve = ((1 - t) ** 3 * [394, 34] + 3 * (1 - t) ** 2 * t * [430, 34]
                 + 3 * (1 - t) * t ** 2 * [452, 34] + t ** 3 * [475, 28])
    fold = mask(lambda draw: draw.polygon(
        points([*top_curve.tolist(), (539, 28), (539, 571), (494, 532), *curve[::-1].tolist()]),
        fill=255,
    ))
    return Image.fromarray(np.minimum(np.asarray(fold), np.asarray(inner_gold)))


def draw_number(draw, transform=None):
    number = [(151, 170), (217, 81), (235, 81), (235, 170), (254, 170),
              (254, 187), (235, 187), (235, 217), (217, 217),
              (217, 187), (151, 187)]
    counter = [(175, 170), (217, 112), (217, 170)]
    if transform is not None:
        number, counter = transform(number), transform(counter)
    draw.polygon(points(number), fill=255)
    draw.polygon(points(counter), fill=0)


def draw_currency(draw, transform=None, width_scale=1):
    outer, inner = [(145, 274), (275, 404)], [(160, 289), (260, 389)]
    if transform is not None:
        outer, inner = transform(outer), transform(inner)
    draw.ellipse(box((*outer[0], *outer[1])), fill=255)
    for start, end in [((150, 284), (270, 395)), ((269, 284), (150, 395))]:
        line = [start, end] if transform is None else transform([start, end])
        draw.line(points(line), fill=255, width=round(15 * DRAW_SCALE * width_scale))
        for px, py in (start, end):
            cap = [(px - 7.5, py - 7.5), (px + 7.5, py + 7.5)]
            if transform is not None:
                cap = transform(cap)
            draw.ellipse(box((*cap[0], *cap[1])), fill=255)
    draw.ellipse(box((*inner[0], *inner[1])), fill=0)


def draw_symbols(draw):
    draw_number(draw)
    draw_currency(draw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    source = np.asarray(Image.open(args.source).convert("RGB"), dtype=np.float32)
    assert source.shape == (HEIGHT, WIDTH, 3), "This cleanup traces the original 539 x 571 artwork"
    y, x = np.mgrid[:HEIGHT * OUTPUT_SCALE, :WIDTH * OUTPUT_SCALE]
    nx, ny = x / (WIDTH * OUTPUT_SCALE), y / (HEIGHT * OUTPUT_SCALE)
    sy, sx = np.mgrid[:HEIGHT, :WIDTH]
    bright_gold = (source[..., 0] > 180) & (source[..., 2] < 200)
    face_samples = bright_gold & (sx > 95) & (sx < 450) & (sy > 65) & (sy < 430)
    outer_samples = bright_gold & (
        ((sx > 30) & (sx < 58) & (sy > 120) & (sy < 440))
        | ((sx > 100) & (sx < 400) & (sy > 477) & (sy < 525))
        | ((sx > 130) & (sx < 415) & (sy > 25) & (sy < 31))
    )
    gold = fitted_finish(source, outer_samples, nx, ny)
    face = fitted_finish(source, face_samples, nx, ny)
    gold_image = Image.fromarray(np.round(gold).astype(np.uint8)).convert("RGBA")
    face_image = Image.fromarray(np.round(face).astype(np.uint8)).convert("RGBA")
    black = (25, 27, 20, 255)
    ivory = (247, 238, 221, 255)
    output = Image.new("RGBA", gold_image.size, (0, 0, 0, 0))
    outer = rounded((1, 1, 537, 570), 96)
    inner_gold = rounded((24, 23, 515, 548), 69)
    tube_outer = rounded((65, 34, 479, 460), 85)
    tube_inner = rounded((81, 51, 463, 444), 69)
    output.paste(black, mask=outer)
    output.paste(gold_image, mask=inner_gold)

    # A cubic curve follows the original folded corner into its diagonal crease.
    fold = fold_mask(inner_gold)
    output.paste((234, 220, 185, 255), mask=fold)
    output.paste(ivory, mask=tube_outer)
    output.paste(face_image, mask=tube_inner)

    output.paste(black, mask=mask(draw_symbols))
    destination = Path(__file__).resolve().parent / "start_logo_clean.png"
    output.save(destination)
    tube = np.asarray(tube_outer).astype(np.float32) - np.asarray(tube_inner).astype(np.float32)
    Image.fromarray(np.clip(tube, 0, 255).astype(np.uint8)).save(destination.with_name("start_logo_tube_mask.png"))
    assert output.size == (1078, 1142)
    assert output.getpixel((0, 0))[3] == 0
    assert output.getpixel((400, 80))[:3] == ivory[:3]
    assert output.getpixel((450, 260))[:3] == black[:3]
    assert output.getpixel((440 * OUTPUT_SCALE, 40 * OUTPUT_SCALE))[:3] == (234, 220, 185)
    assert np.count_nonzero((np.asarray(outer) > 0) & (np.asarray(outer) < 255)) > 100
    print(f"Created {destination}: 1078 x 1142, transparent background, supersampled edges")


if __name__ == "__main__":
    main()
