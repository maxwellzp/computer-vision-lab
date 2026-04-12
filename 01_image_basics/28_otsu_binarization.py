from pathlib import Path
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

# Correct pipeline: gray → blur → Otsu


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from filesystem in BGR using OpenCV."""
    img = cv.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    return img


def otsu_binarization(img: np.ndarray):
    _, manual = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

    placeholder = 0
    t, thresh = cv.threshold(img, placeholder, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    plt.figure()

    plt.subplot(131)
    plt.title("Original image")
    plt.imshow(img, cmap="gray")

    plt.subplot(132)
    plt.title("Manual thresh: 127")
    plt.imshow(manual, cmap="gray")

    plt.subplot(133)
    plt.title(f"Otsu thresh: {t}")
    plt.imshow(thresh, cmap="gray")

    plt.show()


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo_7.png"

    bgr_image = read_image(image_path)
    gray_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2GRAY)

    otsu_binarization(gray_image)
