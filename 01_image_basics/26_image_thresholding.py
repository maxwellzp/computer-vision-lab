from pathlib import Path
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

# black = 0
# white = 255

# 255 / 2 ≈ 127

# THRESH_BINARY
# >127 -> 255
# ≤127 -> 0
# if pixel > 127:
#     pixel = 255
# else:
#     pixel = 0

# THRESH_BINARY_INV
# >127 -> 0
# ≤127 -> 255
# if pixel > 127:
#     pixel = 0
# else:
#     pixel = 255

# THRESH_TRUNC
# pixel = min(pixel, 127)

# THRESH_TOZERO
# if pixel < 127:
#     pixel = 0

# THRESH_TOZERO_INV
# if pixel > 127:
#     pixel = 0


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from filesystem in BGR using OpenCV."""
    img = cv.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    return img


def thresholding(img: np.ndarray):
    thresholds = [
        ("binary", cv.THRESH_BINARY),
        ("binary inv", cv.THRESH_BINARY_INV),
        ("tozero", cv.THRESH_TOZERO),
        ("tozero inv", cv.THRESH_TOZERO_INV),
        ("trunc", cv.THRESH_TRUNC),
    ]

    plt.figure()
    plt.subplot(231)
    plt.title("Original")
    plt.imshow(img, cmap="gray")

    for i, (image_title, threshold) in enumerate(thresholds):
        plt.subplot(2, 3, i + 2)
        plt.title(image_title)
        # 255 / 2 ≈ 127
        # 127 is default value
        _, processed_img = cv.threshold(src=img, thresh=127, maxval=255, type=threshold)
        plt.imshow(processed_img, cmap="gray")

    plt.show()


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo_7.png"

    bgr_image = read_image(image_path)
    gray_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2GRAY)

    thresholding(gray_image)
