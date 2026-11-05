import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).parent / "profile-contours"
SOURCE = ROOT / "source.png"

# Control points are traced in the enlarged inspection crops, then mapped to source pixels.
TRACES = [
    {
        "name": "left",
        "crop": [340, 545, 535, 665],
        "eyeball": [
            [174, 273],
            [[207, 240], [262, 196], [302, 171]],
            [[351, 137], [415, 122], [468, 131]],
            [[536, 140], [615, 187], [680, 241]],
            [[651, 275], [592, 308], [537, 327]],
            [[482, 347], [410, 351], [354, 342]],
            [[285, 332], [220, 309], [174, 273]],
        ],
        "iris": [
            [347, 141],
            [[386, 126], [446, 122], [489, 136]],
            [[525, 153], [547, 187], [547, 227]],
            [[551, 270], [539, 306], [511, 325]],
            [[483, 342], [437, 346], [395, 346]],
            [[351, 348], [321, 330], [307, 303]],
            [[287, 269], [287, 224], [302, 190]],
            [[312, 166], [327, 151], [347, 141]],
        ],
    },
    {
        "name": "right",
        "crop": [690, 490, 875, 615],
        "eyeball": [
            [165, 345],
            [[185, 294], [240, 240], [282, 211]],
            [[328, 175], [369, 157], [412, 158]],
            [[468, 158], [516, 192], [550, 216]],
            [[589, 240], [620, 263], [640, 284]],
            [[619, 311], [568, 343], [518, 363]],
            [[466, 383], [404, 387], [351, 382]],
            [[280, 376], [216, 366], [165, 345]],
        ],
        "iris": [
            [320, 184],
            [[348, 166], [380, 156], [410, 158]],
            [[451, 159], [479, 175], [499, 204]],
            [[524, 238], [533, 277], [519, 312]],
            [[506, 346], [479, 368], [441, 376]],
            [[399, 383], [354, 383], [322, 365]],
            [[289, 344], [276, 313], [277, 275]],
            [[275, 236], [291, 204], [320, 184]],
        ],
    },
]


def points(trace, crop):
    offset = np.array(crop[:2], dtype=float)
    start = np.array(trace[0], dtype=float)
    result = []
    for controls in trace[1:]:
        a, b, end = np.array(controls, dtype=float)
        for t in np.linspace(0, 1, 81, endpoint=False):
            point = (1 - t) ** 3 * start + 3 * (1 - t) ** 2 * t * a
            point += 3 * (1 - t) * t ** 2 * b + t ** 3 * end
            result.append(tuple(offset + point / 4))
        start = end
    result.append(result[0])
    return result


def main():
    source = Image.open(SOURCE).convert("RGB")
    assert source.size == (1254, 1254)
    scale = 4
    masks = {kind: Image.new("L", (source.width * scale, source.height * scale)) for kind in ("eyeball", "iris")}
    overlay = Image.new("RGBA", (source.width * scale, source.height * scale))
    draw = ImageDraw.Draw(overlay)
    for trace in TRACES:
        for kind, color in [("eyeball", "#6dff73"), ("iris", "#36dcff")]:
            path = [(round(px * scale), round(py * scale)) for px, py in points(trace[kind], trace["crop"])]
            ImageDraw.Draw(masks[kind]).polygon(path, fill=255)
            draw.line(path, fill=color, width=3, joint="curve")
    for kind, mask in masks.items():
        masks[kind] = mask.resize(source.size, Image.Resampling.LANCZOS)
    masks["iris"] = Image.fromarray(
        np.minimum(np.asarray(masks["iris"]), np.asarray(masks["eyeball"]))
    )
    for kind, mask in masks.items():
        mask.save(ROOT / f"{kind}-mask.png")
        pixels = np.asarray(mask)
        assert np.count_nonzero(pixels > 128) > 800
        for trace in TRACES:
            left, top, right, bottom = trace["crop"]
            assert np.count_nonzero(pixels[top:bottom, left:right] > 128) > 400
    assert np.all(np.asarray(masks["iris"]) <= np.asarray(masks["eyeball"]))
    overlay = overlay.resize(source.size, Image.Resampling.LANCZOS)
    overlay.save(ROOT / "outlines.png")
    outlined = Image.alpha_composite(source.convert("RGBA"), overlay).convert("RGB")
    outlined.save(ROOT / "outlined.png")
    for trace in TRACES:
        box = trace["crop"]
        size = ((box[2] - box[0]) * 4, (box[3] - box[1]) * 4)
        source.crop(box).resize(size, Image.Resampling.LANCZOS).save(ROOT / f"{trace['name']}-original.png")
        outlined.crop(box).resize(size, Image.Resampling.LANCZOS).save(ROOT / f"{trace['name']}-outlined.png")
    (ROOT / "contours.json").write_text(json.dumps(TRACES, indent=2) + "\n", encoding="ascii")
    print("Created high-resolution iris and visible-eyeball masks, outline overlay, and enlarged inspection crops.")
    print("Validated mask coverage for both eyes and complete iris containment inside the eyeball mask.")


if __name__ == "__main__":
    main()
