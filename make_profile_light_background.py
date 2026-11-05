from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).parent


def main():
    cv2.setNumThreads(2)
    source = Image.open(ROOT / "profile-contours" / "source.png").convert("RGB")
    size = 627
    small = source.resize((size, size), Image.Resampling.LANCZOS)
    seed = Image.new("L", (size, size))
    outline = [
        (0, 322), (25, 322), (43, 310), (69, 307), (93, 300), (103, 288),
        (83, 268), (69, 235), (64, 221), (59, 209), (59, 183), (62, 172),
        (63, 136), (65, 102), (75, 73), (96, 49), (132, 32), (173, 25),
        (218, 33), (250, 54), (272, 82), (281, 119), (285, 152), (289, 171),
        (295, 186), (295, 211), (288, 229), (280, 253), (267, 278),
        (263, 294), (286, 307), (314, 314), (360, 324), (360, 360), (0, 360),
    ]
    ImageDraw.Draw(seed).polygon([(round(x * size / 360), round(y * size / 360)) for x, y in outline], fill=255)
    binary = np.asarray(seed)
    inner = cv2.erode(binary, np.ones((23, 23), np.uint8)) > 0
    outer = cv2.dilate(binary, np.ones((23, 23), np.uint8)) > 0
    labels = np.where(binary > 0, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    labels[inner] = cv2.GC_FGD
    labels[~outer] = cv2.GC_BGD
    cv2.grabCut(
        np.asarray(small)[..., ::-1].copy(), labels, None,
        np.zeros((1, 65)), np.zeros((1, 65)), 4, cv2.GC_INIT_WITH_MASK,
    )
    background = np.isin(labels, [cv2.GC_BGD, cv2.GC_PR_BGD]).astype(np.uint8) * 255
    mask = Image.fromarray(background).resize(source.size, Image.Resampling.LANCZOS)
    mask = mask.filter(ImageFilter.GaussianBlur(1.0))
    alpha = np.asarray(mask, dtype=float) / 255
    pixels = np.asarray(source)
    # Keep the existing background gradient, lifting it toward the original warm beige.
    lighter = pixels.astype(float) * 0.35 + np.array([233, 225, 214]) * 0.65
    result = np.round(pixels * (1 - alpha[..., None]) + lighter * alpha[..., None]).astype(np.uint8)
    assert np.array_equal(result[alpha == 0], pixels[alpha == 0])
    assert np.all(alpha[420:970, 360:900] == 0), "Background mask overlaps the central face"
    assert np.mean(result[alpha > 0.99]) > np.mean(pixels[alpha > 0.99]) + 25
    mask.save(ROOT / "profile-hdr" / "background-mask-v3.png")
    image = Image.fromarray(result)
    image.save(ROOT / "profile-hdr" / "source-light-background-v3.png")
    image.resize((360, 360), Image.Resampling.LANCZOS).save(ROOT / "profile-hdr" / "light-background-preview-v3.png")
    print("Lightened only the background; validated original subject pixels remain unchanged.")


if __name__ == "__main__":
    main()
