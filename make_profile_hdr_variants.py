import json
from pathlib import Path

import imagecodecs
import numpy as np
from PIL import Image


ROOT = Path(__file__).parent
OUTPUT = ROOT / "profile-hdr"
WEIGHTS = np.array([0.2126, 0.7152, 0.0722])
DISPLAY_BOOST = 1000 / 203


def main():
    OUTPUT.mkdir(exist_ok=True)
    base = np.asarray(Image.open(ROOT / "profile-contours" / "source.png").convert("RGB"))
    rgb = base.astype(np.float32) / 255
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    height, width = base.shape[:2]
    sdr = np.full((height, width, 4), 255, dtype=np.uint8)
    sdr[..., :3] = base
    fallback_pixels = None
    report = {}
    for region in ("iris-and-pupil", "entire-eyeball"):
        mask = np.asarray(
            Image.open(ROOT / "profile-soft-glow" / f"{region}-mask.png"), dtype=np.float32,
        ) / 255
        assert mask.shape == (height, width)
        alpha = (mask * 0.995)[..., None]
        for color, target in [("white", [4.0, 4.0, 4.0]), ("red", [12.0, 0.035, 0.018])]:
            name = f"{region}-{color}"
            hdr_rgb = linear * (1 - alpha) + np.array(target, dtype=np.float32) * alpha
            hdr = np.ones((height, width, 4), dtype=np.float16)
            hdr[..., :3] = hdr_rgb
            assert np.array_equal(hdr[..., :3][mask == 0], linear.astype(np.float16)[mask == 0])
            encoded = imagecodecs.ultrahdr_encode(
                hdr, level=100, sdr=sdr, scale=1, multichannel=True,
                gamut=imagecodecs.ULTRAHDR.CG.BT_709,
                transfer=imagecodecs.ULTRAHDR.CT.LINEAR,
                gainmap_level=100, boostmin=0.01, boostmax=1024, nits=1000,
            )
            path = OUTPUT / f"{name}.jpg"
            path.write_bytes(encoded)
            saved = path.read_bytes()
            assert imagecodecs.ultrahdr_check(saved)
            decoded = imagecodecs.ultrahdr_decode(saved, dtype=np.float16, boost=DISPLAY_BOOST)
            baseline = imagecodecs.ultrahdr_decode(saved, dtype=np.float16, boost=1)
            assert decoded.shape == hdr.shape and np.isfinite(decoded).all()
            with Image.open(path) as fallback:
                pixels = np.asarray(fallback.convert("RGB"))
            if fallback_pixels is None:
                fallback_pixels = pixels.copy()
            assert np.array_equal(pixels, fallback_pixels), "SDR fallback differs between variants"
            assert np.abs(pixels.astype(float) - base).mean() < 2
            lit = decoded[..., :3].astype(np.float32)
            normal = baseline[..., :3].astype(np.float32)
            outside = mask == 0
            outside_error = np.abs(lit[outside] - normal[outside])
            assert np.quantile(outside_error, 0.999) < 0.03
            medians = []
            for left, top, right, bottom in [(340, 545, 535, 665), (690, 490, 875, 615)]:
                eye_mask = mask[top:bottom, left:right] > 0.95
                eye = lit[top:bottom, left:right][eye_mask]
                luminance = eye @ WEIGHTS
                median = float(np.median(luminance))
                medians.append(median)
                assert median > 1.0, f"{name}: eye not above SDR white: {median}"
                if color == "red":
                    assert np.median(eye[:, 0]) > 8 * np.median(eye[:, 1:].max(axis=1))
                else:
                    assert np.median(eye.max(axis=1) - eye.min(axis=1)) < 0.5
            report[name] = {
                "size": list(base.shape[:2][::-1]),
                "bytes": len(saved),
                "median_eye_luminance_relative_to_sdr_white": medians,
                "decode_display_boost": DISPLAY_BOOST,
                "sdr_fallback_mean_error_8bit": float(np.abs(pixels.astype(float) - base).mean()),
                "outside_mask_hdr_error_p999_relative_to_sdr_white": float(np.quantile(outside_error, 0.999)),
            }
            print(f"{name}: {len(saved):,} bytes; eye luminance {medians}; valid RGB HDR gain map")
    (OUTPUT / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    print("All four variants retain pixel-identical normal SDR fallback images.")


if __name__ == "__main__":
    main()
