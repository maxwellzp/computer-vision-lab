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
    image_path = BASE_DIR / "data" / "images" / "photo_3.png"
    img = read_image(image_path).copy()

    # convert BGR to HSV
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # lower = (Hmin, Smin, Vmin)
    # upper = (Hmax, Smax, Vmax)

    # HSV Hue ranges:

    # 0   red
    # 20  orange
    # 30  yellow
    # 60  green
    # 90  cyan
    # 120 blue
    # 150 purple

    # Blue: 100–140

    lower = (100, 150, 0)
    upper = (140, 255, 255)

    # maks is a binary image
    mask = cv.inRange(hsv_img, lower, upper)

    print("Shape:", mask.shape)
    # Shape: (628, 1200)

    plt.imshow(mask, cmap="gray")

    plt.title("Blue mask")

    plt.show()
