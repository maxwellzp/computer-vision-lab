import cv2 as cv
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def read_image_as_gray(img_path: Path):
    """Read an image from filesystem as grayscale using OpenCV."""
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    cv.imshow("Gray image 1", img)
    cv.waitKey(0)


def display_grayscale_image(img_path: Path):
    """Read an image from filesystem then convert it to grayscale using OpenCV."""
    img = cv.imread(img_path)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    cv.imshow("Gray image 2", img_gray)
    cv.waitKey(0)


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo.png"
    read_image_as_gray(image_path)
    display_grayscale_image(image_path)
