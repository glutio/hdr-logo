import json
import io
import struct
from pathlib import Path

import imagecodecs
import numpy as np
from PIL import Image


ROOT = Path(__file__).parent
OUTPUT = ROOT / "profile-hdr"
WEIGHTS = np.array([0.2126, 0.7152, 0.0722])
DISPLAY_BOOST = 1000 / 203
OFFSET = 1 / 16


def encode_even_gainmap(template, color, alpha):
    split = template.index(b"\xff\xd9\xff\xd8") + 2
    primary = bytearray(template[:split])
    base = imagecodecs.ultrahdr_decode(template, dtype=np.float16, boost=1)[..., :3].astype(np.float32)
    target = base * (1 - alpha) + np.asarray(color, dtype=np.float32) * alpha
    # Author against the stored JPEG, with offsets to illuminate even zero-valued pixels.
    stops = np.log2((target + OFFSET) / (base + OFFSET))
    assert stops.min() >= -4 and stops.max() <= 8
    gain = np.round((stops + 4) / 12 * 255).astype(np.uint8)
    stream = io.BytesIO()
    Image.fromarray(gain).save(stream, format="JPEG", quality=100, subsampling=0)
    # ISO 21496-1: base color space, shared RGB parameters, SDR headroom zero.
    metadata = b"urn:iso:std:iso:ts:21496:-1\x00"
    metadata += struct.pack(">HHB", 0, 0, 0x40)
    metadata += struct.pack(">IIII", 0, 1, round(np.log2(DISPLAY_BOOST) * 1000000), 1000000)
    metadata += struct.pack(">iIiIIIiIiI", -4, 1, 8, 1, 1, 1, 1, 16, 1, 16)
    segment = b"\xff\xe2" + struct.pack(">H", len(metadata) + 2) + metadata
    gain_jpeg = stream.getvalue()
    secondary = gain_jpeg[:2] + segment + gain_jpeg[2:]
    mpf = primary.index(b"MPF\x00") + 4
    assert primary[mpf:mpf + 4] == b"MM\x00*"
    directory = mpf + struct.unpack_from(">I", primary, mpf + 4)[0]
    count = struct.unpack_from(">H", primary, directory)[0]
    entries = None
    for i in range(count):
        pos = directory + 2 + i * 12
        if struct.unpack_from(">H", primary, pos)[0] == 0xB002:
            entries = mpf + struct.unpack_from(">I", primary, pos + 8)[0]
    assert entries is not None
    struct.pack_into(">I", primary, entries + 4, len(primary))
    struct.pack_into(">II", primary, entries + 20, len(secondary), len(primary) - mpf)
    return bytes(primary) + secondary


def main():
    OUTPUT.mkdir(exist_ok=True)
    base = np.asarray(Image.open(OUTPUT / "source-light-background-v3.png").convert("RGB"))
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
        for color, target in [("white", [4.0, 4.0, 4.0]), ("red", [10.0, 0.30, 0.015])]:
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
            encoded = encode_even_gainmap(encoded, target, alpha)
            path = OUTPUT / f"{name}-v3.jpg"
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
                eye_mask = mask[top:bottom, left:right] > 0.98
                eye = lit[top:bottom, left:right][eye_mask]
                luminance = eye @ WEIGHTS
                median = float(np.median(luminance))
                medians.append(median)
                assert median > 1.0, f"{name}: eye not above SDR white: {median}"
                assert luminance.min() > median * 0.80, f"{name}: remaining dark patch"
                for channel in range(3):
                    if target[channel] > 0.1:
                        assert np.max(np.abs(eye[:, channel] / target[channel] - 1)) < 0.20
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
            preview = np.clip(lit / 4, 0, 1)
            preview = np.where(preview <= 0.0031308, preview * 12.92, 1.055 * preview ** (1 / 2.4) - 0.055)
            preview = Image.fromarray(np.round(preview * 255).astype(np.uint8))
            preview.crop((350, 520, 860, 645)).resize((1020, 250)).save(OUTPUT / f"{name}-v3-decoded.png")
            print(f"{name}: {len(saved):,} bytes; eye luminance {medians}; valid RGB HDR gain map")
    (OUTPUT / "validation-v3.json").write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    print("All four variants retain pixel-identical normal SDR fallback images.")


if __name__ == "__main__":
    main()
