import json
from pathlib import Path

import imagecodecs
import numpy as np
from PIL import Image

from clean_logo import draw_symbols, mask, rounded
from make_hdr_logo import blur, to_linear, to_srgb


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "backlight"


def main():
    OUTPUT.mkdir(exist_ok=True)
    source = Image.open(ROOT / "variants" / "ivory-defined-bevel.png").convert("RGBA")
    background = Image.new("RGBA", source.size, (0, 0, 0, 255))
    rgb8 = np.asarray(Image.alpha_composite(background, source).convert("RGB"))
    linear = to_linear(rgb8.astype(np.float32) / 255)
    symbols = np.asarray(mask(draw_symbols), dtype=np.float32) / 255
    shoulder = np.asarray(rounded((61, 36, 483, 462), 84), dtype=np.float32) / 255
    face = np.asarray(rounded((75, 48, 469, 447), 72), dtype=np.float32) / 255
    rim = np.clip(shoulder - face, 0, 1)
    cap = np.asarray(rounded((16, 16, 523, 555), 84), dtype=np.float32) / 255
    luminous = np.maximum(symbols, rim)
    warmth = np.array([1.0, 0.90, 0.72], dtype=np.float32)
    halo = (0.16 * blur(symbols, 5) + 0.14 * blur(rim, 8)) * (1 - luminous) * cap
    hdr_rgb = linear * (1 - luminous[..., None])
    hdr_rgb += (4.4 * symbols + 3.8 * rim)[..., None] * warmth
    hdr_rgb += halo[..., None] * warmth
    hdr = np.ones((*rgb8.shape[:2], 4), dtype=np.float16)
    hdr[..., :3] = hdr_rgb.astype(np.float16)
    sdr = np.full(hdr.shape, 255, dtype=np.uint8)
    sdr[..., :3] = rgb8

    encoded = imagecodecs.ultrahdr_encode(
        hdr, level=100, sdr=sdr, gamut=imagecodecs.ULTRAHDR.CG.BT_709,
        transfer=imagecodecs.ULTRAHDR.CT.LINEAR, scale=1, gainmap_level=100,
        boostmin=1.0, boostmax=512.0, nits=1015,
    )
    destination = OUTPUT / "ivory-backlit-hdr.jpg"
    destination.write_bytes(encoded)
    Image.fromarray(rgb8).save(OUTPUT / "ivory-unlit-sdr.png")
    Image.fromarray(np.round(luminous * 255).astype(np.uint8)).save(OUTPUT / "backlight-mask.png")

    # This is an SDR illustration of the lit design, not an HDR export.
    preview = hdr_rgb / (1 + hdr_rgb)
    preview = np.round(np.clip(to_srgb(preview), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(preview).save(OUTPUT / "ivory-lit-sdr-illustration.png")

    saved = destination.read_bytes()
    assert imagecodecs.ultrahdr_check(saved), "Invalid Ultra HDR JPEG"
    weights = np.array([0.2627, 0.6780, 0.0593])
    ink = symbols > 0.99
    rim_core = rim > 0.99
    height, width = rgb8.shape[:2]
    y, x = np.mgrid[:height, :width]
    plain_face = (x > width * 0.56) & (x < width * 0.74) & (y > height * 0.25) & (y < height * 0.65)
    readings = []
    reference = None
    for headroom in [1, 1.5, 2, 3, 5]:
        decoded = imagecodecs.ultrahdr_decode(saved, dtype=np.float16, boost=headroom)
        assert decoded.shape == hdr.shape and np.isfinite(decoded).all()
        luminance = decoded[..., :3].astype(np.float32) @ weights
        reading = {
            "headroom": headroom,
            "symbols": float(np.median(luminance[ink])),
            "rim": float(np.median(luminance[rim_core])),
            "face": float(np.median(luminance[plain_face])),
        }
        if headroom == 1:
            reference = reading
            assert reading["symbols"] < 0.03, "SDR symbols are no longer black"
        assert abs(reading["face"] / reference["face"] - 1) < 0.03, "Face brightness changed"
        readings.append(reading)
    assert readings[-1]["symbols"] > 2.5, f"Black symbols did not illuminate: {readings}"
    assert readings[-1]["rim"] > 2.5, f"Rim did not illuminate: {readings}"
    assert all(a["symbols"] <= b["symbols"] for a, b in zip(readings, readings[1:]))
    with Image.open(destination) as fallback:
        fallback.load()
        assert fallback.size == source.size
    report = {
        "codec": imagecodecs.ultrahdr_version(),
        "size": list(source.size),
        "luminance_unit": "linear SDR reference white",
        "readings": readings,
    }
    (OUTPUT / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Created and decoded {destination} ({len(saved):,} bytes)")


if __name__ == "__main__":
    main()
