import argparse
from pathlib import Path

import imagecodecs
import numpy as np
from PIL import Image, ImageFilter


def to_linear(rgb):
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def to_srgb(rgb):
    return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)


def blur(mask, radius):
    image = Image.fromarray(np.round(mask * 255).astype(np.uint8))
    return np.asarray(image.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32) / 255


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path.cwd())
    parser.add_argument("--tube-mask", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    source = Image.open(args.source).convert("RGBA")
    backdrop = Image.new("RGBA", source.size, (0, 0, 0, 255))
    rgb8 = np.asarray(Image.alpha_composite(backdrop, source).convert("RGB"))
    rgb = rgb8.astype(np.float32) / 255
    linear = to_linear(rgb)
    height, width = rgb.shape[:2]
    y, x = np.mgrid[:height, :width]
    nx, ny = x / width, y / height

    # The existing ivory pixels isolate the tube without redrawing the artwork.
    ivory = np.clip((rgb8[..., 2].astype(np.float32) - 195) / 23, 0, 1)
    ivory *= np.clip((rgb8[..., 1].astype(np.float32) - 224) / 12, 0, 1)
    ring = ivory * ((nx > 0.10) & (nx < 0.91) & (ny > 0.045) & (ny < 0.81))
    if args.tube_mask:
        tube_mask = Image.open(args.tube_mask).convert("L")
        assert tube_mask.size == source.size, "Tube mask dimensions must match the logo"
        ring = np.asarray(tube_mask, dtype=np.float32) / 255
    glow = blur(ring, 8 * width / 539)
    gold_face = np.clip((rgb.mean(axis=2) - 0.2) / 0.4, 0, 1)
    warmth = np.array([1.0, 0.77, 0.43], dtype=np.float32)
    base_linear = np.clip(linear + 0.055 * glow[..., None] * gold_face[..., None] * warmth, 0, 1)
    base8 = np.round(np.clip(to_srgb(base_linear), 0, 1) * 255).astype(np.uint8)
    base_linear = to_linear(base8.astype(np.float32) / 255)

    crease_distance = np.abs(ny - (1.16 * nx - 0.126))
    crease = np.exp(-(crease_distance / 0.004) ** 2)
    crease *= (nx > 0.80) & (nx < 0.925) & (ny > 0.79) & (ny < 0.935)
    crease *= gold_face
    gain = 1 + 3.3 * ring + 0.30 * glow * gold_face + 0.55 * crease
    hdr = np.ones((height, width, 4), dtype=np.float16)
    hdr[..., :3] = (base_linear * gain[..., None]).astype(np.float16)
    sdr = np.full((height, width, 4), 255, dtype=np.uint8)
    sdr[..., :3] = base8

    encoded = imagecodecs.ultrahdr_encode(
        hdr, level=98, sdr=sdr, gamut=imagecodecs.ULTRAHDR.CG.BT_709,
        transfer=imagecodecs.ULTRAHDR.CT.LINEAR, scale=1, gainmap_level=100,
        boostmin=1.0, boostmax=5.0, nits=1015,
    )
    output = args.output / "start_logo_hdr.jpg"
    output.write_bytes(encoded)
    Image.fromarray(base8).save(args.output / "start_logo_sdr_preview.png")

    # Decode the saved file through the native HDR codec, not just the base JPEG.
    saved = output.read_bytes()
    assert imagecodecs.ultrahdr_check(saved), "Missing valid Ultra HDR gain map"
    decoded_hdr = imagecodecs.ultrahdr_decode(saved, dtype=np.float16, boost=5)
    decoded_sdr = imagecodecs.ultrahdr_decode(saved, dtype=np.float16, boost=1)
    assert decoded_hdr.shape == hdr.shape
    assert np.isfinite(decoded_hdr).all()
    weights = np.array([0.2627, 0.6780, 0.0593])
    luminance_hdr = decoded_hdr[..., :3].astype(np.float32) @ weights
    luminance_sdr = decoded_sdr[..., :3].astype(np.float32) @ weights
    ratio = luminance_hdr / np.maximum(luminance_sdr, 1e-5)
    tube_boost = float(np.median(ratio[ring > 0.95]))
    face = (nx > 0.56) & (nx < 0.74) & (ny > 0.25) & (ny < 0.65)
    face_boost = float(np.median(ratio[face]))
    assert tube_boost > 3, f"Tube did not light up: {tube_boost}"
    assert abs(face_boost - 1) < 0.05, f"Gold face brightness changed: {face_boost}"
    with Image.open(output) as fallback:
        fallback.load()
        assert fallback.size == source.size
    print(f"Created {output} ({len(saved):,} bytes, {width} x {height})")
    print(f"Decoded tube boost: {tube_boost:.2f}x; gold face: {face_boost:.2f}x SDR")
    print("SDR fallback: start_logo_sdr_preview.png")
    print("JPEG is opaque; transparent source pixels were composited onto black.")


if __name__ == "__main__":
    main()
