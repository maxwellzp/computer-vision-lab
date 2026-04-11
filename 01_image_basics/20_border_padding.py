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


def display_image_with_border(img: np.ndarray, padding: int):
    border_data = [
        ("reflect_101", cv.BORDER_REFLECT_101),
        ("constant", cv.BORDER_CONSTANT),
        ("reflect", cv.BORDER_REFLECT),
        ("replicate", cv.BORDER_REPLICATE),
        ("wrap", cv.BORDER_WRAP),
    ]

    plt.figure()
    plt.subplot(231)
    plt.imshow(img)
    plt.title("Original image")
    plt.axis("off")

    for i, (border_title, border_type) in enumerate(border_data):
        plt.subplot(2, 3, i + 2)
        plt.imshow(
            cv.copyMakeBorder(img, padding, padding, padding, padding, border_type)
        )
        plt.title(border_title)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo_3.png"
    bgr_image = read_image(image_path)
    rgb_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)
    display_image_with_border(rgb_image, 100)
