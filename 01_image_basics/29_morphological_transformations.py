from pathlib import Path
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# blur -> threshold -> morphology

BASE_DIR = Path(__file__).resolve().parent.parent


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from filesystem in BGR using OpenCV."""
    img = cv.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    return img


def morph_transfromations(img: np.ndarray):
    blured_image = cv.GaussianBlur(img, (7, 7), 0)

    threshold = cv.adaptiveThreshold(
        blured_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2
    )

    kernel = np.ones((5, 5), np.uint8)

    eroded_image = cv.erode(threshold, kernel, iterations=1)

    dilated_image = cv.dilate(threshold, kernel, iterations=1)

    plt.figure(figsize=(14, 6))

    plt.subplot(251)
    plt.title("Original image")
    plt.imshow(img, cmap="gray")

    plt.subplot(252)
    plt.title("Threshold")
    plt.imshow(threshold, cmap="gray")

    plt.subplot(253)
    plt.title("Eroded image")
    plt.imshow(eroded_image, cmap="gray")

    plt.subplot(254)
    plt.title("Dilated image")
    plt.imshow(dilated_image, cmap="gray")

    morph_types = [
        ("Open", cv.MORPH_OPEN),
        ("Close", cv.MORPH_CLOSE),
        ("Gradient", cv.MORPH_GRADIENT),
        ("Tophat", cv.MORPH_TOPHAT),
        ("Blackhat", cv.MORPH_BLACKHAT),
    ]

    for i, (title, morph_type) in enumerate(morph_types):
        morphed_image = cv.morphologyEx(threshold, morph_type, kernel)
        plt.subplot(2, 5, i + 6)
        plt.title(title)
        plt.imshow(morphed_image, cmap="gray")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo_5.png"

    bgr_image = read_image(image_path)
    gray_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2GRAY)

    morph_transfromations(gray_image)
