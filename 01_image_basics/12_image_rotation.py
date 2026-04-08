import cv2 as cv
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from filesystem in BGR using OpenCV."""
    img = cv.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    return img


def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate an image by a certain angle"""
    height, width = img.shape[:2]
    T = cv.getRotationMatrix2D((width // 2, height // 2), angle, 1)
    rotated = cv.warpAffine(img, T, (width, height), flags=cv.INTER_LINEAR)
    return rotated


def display_image(img: np.ndarray) -> None:
    cv.imshow("Rotated image", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo.png"
    img = read_image(image_path)
    rotated_image = rotate_image(img, 45)
    display_image(rotated_image)
