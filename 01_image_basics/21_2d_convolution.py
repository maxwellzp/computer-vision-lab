import cv2 as cv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from filesystem in BGR using OpenCV."""
    img = cv.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    return img


def create_kernel(n: int):
    """Create a normalized averaging kernel of size (n x n)."""

    return np.ones((n, n), np.float32) / (n * n)


def convolution_2d(img: np.ndarray):
    """Apply 2D convolution to an image using an averaging kernel."""

    kernel = create_kernel(7)
    # Image patch:
    # [10 20 30]
    # [40 50 60]
    # [70 80 90]

    # Kernel:
    # [1 1 1]
    # [1 1 1]
    # [1 1 1] / 9
    img_filtered = cv.filter2D(img, -1, kernel)
    display_original_and_filtered(img, img_filtered)


def display_original_and_filtered(img: np.ndarray, img_filtered: np.ndarray):
    """Display original and filtered images side by side."""

    plt.figure()

    plt.subplot(121)
    plt.imshow(img)

    plt.subplot(122)
    plt.imshow(img_filtered)

    plt.show()


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo.png"
    bgr_image = read_image(image_path)
    rgb_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)
    convolution_2d(rgb_image)
