import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter


ROOT = Path(__file__).parent / "profile-contours"
SOURCE = ROOT / "source.png"

TRACES = [
    {"name": "left", "crop": [340, 545, 535, 665]},
    {"name": "right", "crop": [690, 490, 875, 615]},
]

LIDS = {
    "left": [
        [174, 273], [194, 257], [213, 239], [232, 219], [254, 203], [275, 187],
        [298, 172], [321, 157], [346, 142], [373, 133], [402, 125], [431, 121],
        [459, 124], [485, 130], [511, 140], [535, 153], [560, 167], [583, 182],
        [605, 197], [626, 209], [648, 223], [666, 234], [680, 241],
        [662, 259], [639, 276], [614, 289], [587, 304], [561, 315], [533, 326],
        [505, 334], [477, 340], [449, 344], [421, 348], [395, 349], [369, 348],
        [344, 343], [319, 338], [294, 331], [270, 323], [246, 314], [223, 301],
        [202, 289], [184, 279],
    ],
    "right": [
        [165, 345], [180, 312], [200, 282], [221, 256], [242, 235], [259, 222],
        [272, 215], [282, 204], [292, 199], [305, 185], [320, 179], [337, 169],
        [353, 167], [365, 158], [382, 158], [396, 154], [413, 157], [429, 159],
        [444, 166], [459, 172], [478, 184], [495, 187], [514, 194], [535, 199],
        [553, 203], [570, 211], [589, 224], [608, 235], [626, 247], [644, 263],
        [630, 282], [609, 300], [586, 316], [562, 332], [537, 343], [511, 354],
        [485, 362], [457, 369], [426, 374], [397, 376], [367, 375], [339, 373],
        [310, 370], [280, 368], [251, 365], [222, 360], [194, 355],
    ],
}


def interpolate_boundary(anchors, crop):
    anchors = np.asarray(anchors, dtype=float)
    result = []
    for i, current in enumerate(anchors):
        previous = anchors[(i - 1) % len(anchors)]
        following = anchors[(i + 1) % len(anchors)]
        after = anchors[(i + 2) % len(anchors)]
        for t in np.linspace(0, 1, 16, endpoint=False):
            point = 0.5 * (
                2 * current + (-previous + following) * t
                + (2 * previous - 5 * current + 4 * following - after) * t ** 2
                + (-previous + 3 * current - 3 * following + after) * t ** 3
            )
            result.append(tuple(np.asarray(crop[:2]) + point / 4))
    return result


def sample(image, px, py):
    x0 = np.floor(px).astype(int)
    y0 = np.floor(py).astype(int)
    dx, dy = px - x0, py - y0
    return (
        image[y0, x0] * (1 - dx) * (1 - dy)
        + image[y0, x0 + 1] * dx * (1 - dy)
        + image[y0 + 1, x0] * (1 - dx) * dy
        + image[y0 + 1, x0 + 1] * dx * dy
    )


def iris_boundary(source, name, crop):
    gray = np.asarray(source.convert("L"), dtype=float) / 255
    if name == "left":
        cx, cy, rx, ry = 420, 229, 124, 110
    else:
        cx, cy, rx, ry = 400, 270, 124, 109
    center = np.asarray(crop[:2]) + np.array([cx, cy]) / 4
    rx, ry = rx / 4, ry / 4
    angles = np.linspace(-np.pi / 2, 3 * np.pi / 2, 361)
    cosine, sine = np.cos(angles), np.sin(angles)
    seed = 1 / np.sqrt((cosine / rx) ** 2 + (sine / ry) ** 2)
    offsets = np.linspace(-4, 4, 33)
    radii = seed[:, None] + offsets
    outward = sample(
        gray, center[0] + (radii + 0.6) * cosine[:, None],
        center[1] + (radii + 0.6) * sine[:, None],
    )
    inward = sample(
        gray, center[0] + (radii - 0.6) * cosine[:, None],
        center[1] + (radii - 0.6) * sine[:, None],
    )
    cost = -2 * (outward - inward) + 0.008 * offsets[None, :] ** 2
    # The upper iris disappears behind the lid: do not snap to eyelashes.
    cost[sine < -0.55] = 100 * offsets ** 2
    transition = 0.04 * (offsets[:, None] - offsets[None, :]) ** 2
    scores = cost[0].copy()
    back = np.zeros(cost.shape, dtype=int)
    for i in range(1, len(angles)):
        candidates = scores[:, None] + transition
        back[i] = candidates.argmin(axis=0)
        scores = cost[i] + candidates.min(axis=0)
    path = np.zeros(len(angles), dtype=int)
    path[-1] = scores.argmin()
    for i in range(len(angles) - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    radius = seed + offsets[path]
    return list(zip(center[0] + radius * cosine, center[1] + radius * sine))


def main():
    source = Image.open(SOURCE).convert("RGB")
    assert source.size == (1254, 1254)
    scale = 4
    masks = {kind: Image.new("L", (source.width * scale, source.height * scale)) for kind in ("eyeball", "iris")}
    overlay = Image.new("RGBA", (source.width * scale, source.height * scale))
    traced = []
    for trace in TRACES:
        refined = {
            "eyeball": interpolate_boundary(LIDS[trace["name"]], trace["crop"]),
            "iris": iris_boundary(source, trace["name"], trace["crop"]),
        }
        traced.append({"name": trace["name"], "crop": trace["crop"], "paths": refined})
        for kind in ("eyeball", "iris"):
            path = [(round(px * scale), round(py * scale)) for px, py in refined[kind]]
            ImageDraw.Draw(masks[kind]).polygon(path, fill=255)
    masks["iris"] = Image.fromarray(
        np.minimum(np.asarray(masks["iris"]), np.asarray(masks["eyeball"]))
    )
    for kind, color in [("eyeball", "#6dff73"), ("iris", "#36dcff")]:
        boundary = ImageChops.subtract(
            masks[kind].filter(ImageFilter.MaxFilter(5)),
            masks[kind].filter(ImageFilter.MinFilter(3)),
        )
        ink = Image.new("RGBA", overlay.size, color)
        ink.putalpha(boundary)
        overlay = Image.alpha_composite(overlay, ink)
    for kind, mask in masks.items():
        masks[kind] = mask.resize(source.size, Image.Resampling.LANCZOS)
    masks["iris"] = Image.fromarray(
        np.minimum(np.asarray(masks["iris"]), np.asarray(masks["eyeball"]))
    )
    for kind, mask in masks.items():
        mask.save(ROOT / f"{kind}-mask-v2.png")
        pixels = np.asarray(mask)
        assert np.count_nonzero(pixels > 128) > 800
        for trace in TRACES:
            left, top, right, bottom = trace["crop"]
            assert np.count_nonzero(pixels[top:bottom, left:right] > 128) > 400
    assert np.all(np.asarray(masks["iris"]) <= np.asarray(masks["eyeball"]))
    overlay = overlay.resize(source.size, Image.Resampling.LANCZOS)
    overlay.save(ROOT / "outlines-v2.png")
    outlined = Image.alpha_composite(source.convert("RGBA"), overlay).convert("RGB")
    outlined.save(ROOT / "outlined-v2.png")
    for trace in TRACES:
        box = trace["crop"]
        size = ((box[2] - box[0]) * 4, (box[3] - box[1]) * 4)
        source.crop(box).resize(size, Image.Resampling.LANCZOS).save(ROOT / f"{trace['name']}-original.png")
        outlined.crop(box).resize(size, Image.Resampling.LANCZOS).save(ROOT / f"{trace['name']}-outlined-v2.png")
    (ROOT / "contours-v2.json").write_text(json.dumps(traced, indent=2) + "\n", encoding="ascii")
    print("Created high-resolution iris and visible-eyeball masks, outline overlay, and enlarged inspection crops.")
    print("Validated mask coverage for both eyes and complete iris containment inside the eyeball mask.")


if __name__ == "__main__":
    main()
