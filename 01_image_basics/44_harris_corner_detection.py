import numpy as np
import cv2 as cv
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from filesystem in BGR using OpenCV."""
    img = cv.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    return img


def harris_corner_detection(img: np.ndarray):
    rgb_image = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    gray_float = np.float32(gray_img)

    block_size = 5
    sobel_size = 3
    k = 0.04

    harris = cv.cornerHarris(gray_float, block_size, sobel_size, k)

    harris = cv.dilate(harris, None)

    result = rgb_image.copy()
    result[harris > 0.01 * harris.max()] = [255, 0, 0]

    plt.figure()

    plt.subplot(131)
    plt.title("Gray")
    plt.imshow(gray_img, cmap="gray")

    plt.subplot(132)
    plt.title("Harris")
    plt.imshow(harris, cmap="jet")

    plt.subplot(133)
    plt.title("Corners")
    plt.imshow(result)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo_8.png"
    img = read_image(image_path)

    harris_corner_detection(img)
