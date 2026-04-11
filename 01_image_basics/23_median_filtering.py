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


def filter_image(img: np.ndarray, ksize: int) -> np.ndarray:
    """Apply median filtering to remove noise from an image."""
    if ksize % 2 == 0:
        ksize += 1
    filtered_img = cv.medianBlur(img, 7)
    return filtered_img


if __name__ == "__main__":
    # an image with salt & pepper noise
    image_path = BASE_DIR / "data" / "images" / "photo_6.png"
    bgr_image = read_image(image_path)

    filtered_img = filter_image(bgr_image, 7)

    plt.figure()
    rgb_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)
    plt.subplot(121)
    plt.title("Original image")
    plt.imshow(rgb_image)

    plt.subplot(122)
    plt.title("Median Filter")
    filtered_rgb = cv.cvtColor(filtered_img, cv.COLOR_BGR2RGB)
    plt.imshow(filtered_rgb)

    plt.show()
