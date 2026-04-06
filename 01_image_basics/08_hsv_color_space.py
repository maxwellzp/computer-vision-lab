import cv2 as cv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from file using OpenCV."""
    img = cv.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    return img


def display_image(img: np.ndarray, title: str):
    plt.figure()
    plt.imshow(img)
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo.png"
    img = read_image(image_path).copy()

    # convert BGR to HSV
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # convert BGR to HSV
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    display_image(rgb_img, "RGB image")
    display_image(hsv_img, "HSV image")

    # Split an image into 3 channels
    h, s, v = cv.split(hsv_img)

    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.imshow(h, cmap="gray")
    plt.title("Hue")

    plt.subplot(132)
    plt.imshow(s, cmap="gray")
    plt.title("Saturation")

    plt.subplot(133)
    plt.imshow(v, cmap="gray")
    plt.title("Value")

    plt.show()
