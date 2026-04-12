from pathlib import Path
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

# blockSize (3, 5, 7, 11, …)
# 3-7 -> local
# 9-15 -> good
# 21+ -> better

# C
# threshold = mean - C
# 0-5 normal range


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from filesystem in BGR using OpenCV."""
    img = cv.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    return img


def adaptive_thresholding(img: np.ndarray, max_value: float, block_size: int, C: float):
    _, global_thresh = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

    adaptive_mean = cv.adaptiveThreshold(
        img, max_value, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, block_size, C
    )

    adaptive_gaussian = cv.adaptiveThreshold(
        img,
        max_value,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY,
        block_size,
        C,
    )

    plt.figure()

    plt.subplot(221)
    plt.title("Original")
    plt.imshow(img, cmap="gray")

    plt.subplot(222)
    plt.title("Global Threshold")
    plt.imshow(global_thresh, cmap="gray")

    plt.subplot(223)
    plt.title("Adaptive mean")
    plt.imshow(adaptive_mean, cmap="gray")

    plt.subplot(224)
    plt.title("Adaptive Gaussian")
    plt.imshow(adaptive_gaussian, cmap="gray")

    plt.show()


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo_7.png"

    bgr_image = read_image(image_path)
    gray_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2GRAY)

    max_value = 255
    block_size = 11
    C = 2
    adaptive_thresholding(gray_image, max_value, block_size, C)
