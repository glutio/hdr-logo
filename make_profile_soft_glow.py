from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).parent
OUTPUT = ROOT / "profile-soft-glow"


def main():
    OUTPUT.mkdir(exist_ok=True)
    source = Image.open(ROOT / "profile-contours" / "source.png").convert("RGB")
    base = np.asarray(source)
    rgb = base.astype(float) / 255
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    hard = Image.open(ROOT / "profile-contours" / "eyeball-mask-v2.png").convert("L")
    # Inset before feathering so the bright core stays inside the lids.
    inner = hard.filter(ImageFilter.MinFilter(5))
    soft = inner.filter(ImageFilter.GaussianBlur(2.0))
    eyeball = np.asarray(soft, dtype=float) / 255
    assert np.count_nonzero((eyeball > 0) & (eyeball < 1)) > 500
    soft.save(OUTPUT / "feathered-mask.png")
    iris_image = Image.open(ROOT / "profile-contours" / "iris-mask-v2.png").convert("L")
    iris_image = iris_image.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(1.7))
    iris = np.minimum(np.asarray(iris_image, dtype=float) / 255, eyeball)
    pupils = Image.new("L", (source.width * 4, source.height * 4))
    draw = ImageDraw.Draw(pupils)
    for ox, oy, cx, cy, rx, ry in [
        (340, 545, 435, 231, 15, 20),
        (690, 490, 389, 274, 21, 21),
    ]:
        draw.ellipse(
            (ox * 4 + cx - rx, oy * 4 + cy - ry, ox * 4 + cx + rx, oy * 4 + cy + ry),
            fill=255,
        )
    pupils = pupils.resize(source.size, Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(1.0))
    pupil = np.asarray(pupils, dtype=float) / 255
    styles = [
        ("iris-only", iris * (1 - pupil)),
        ("iris-and-pupil", iris),
        ("entire-eyeball", eyeball),
    ]
    for name, mask in styles:
        alpha = (mask * 0.99)[..., None]
        target = 0.98 + 0.02 * linear
        preview = linear * (1 - alpha) + target * alpha
        encoded = np.where(
            preview <= 0.0031308, preview * 12.92,
            1.055 * preview ** (1 / 2.4) - 0.055,
        )
        pixels = np.round(np.clip(encoded, 0, 1) * 255).astype(np.uint8)
        assert np.array_equal(pixels[mask == 0], base[mask == 0])
        image = Image.fromarray(pixels)
        Image.fromarray(np.round(mask * 255).astype(np.uint8)).save(OUTPUT / f"{name}-mask.png")
        image.save(OUTPUT / f"{name}.png")
        image.resize((360, 360), Image.Resampling.LANCZOS).save(OUTPUT / f"{name}-profile.png")
        image.crop((350, 520, 860, 645)).resize((1020, 250)).save(OUTPUT / f"{name}-eyes.png")
        for left, top, right, bottom in [(340, 545, 535, 665), (690, 490, 875, 615)]:
            assert pixels[top:bottom, left:right].mean() > base[top:bottom, left:right].mean()
    source.crop((350, 520, 860, 645)).resize((1020, 250)).save(OUTPUT / "original-eyes.png")
    assert np.max(styles[0][1][pupil > 0.99]) < 0.01
    assert np.median(iris[pupil > 0.99]) > 0.95
    print("Created bright-white iris-only, iris-with-pupil, and full-eyeball SDR simulations.")
    print("Validated fractional edge coverage and unchanged pixels outside the feathered mask.")


if __name__ == "__main__":
    main()
