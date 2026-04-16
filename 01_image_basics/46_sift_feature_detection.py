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


def good_corner_detection(img: np.ndarray):
    rgb_image = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT.create()

    keypoints = sift.detect(gray_img, None)

    result = cv.drawKeypoints(
        img, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    rgb = cv.cvtColor(result, cv.COLOR_BGR2RGB)

    plt.figure()
    plt.title("SIFT Keypoints")
    plt.imshow(rgb)
    plt.show()


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo_8.png"
    img = read_image(image_path)

    good_corner_detection(img)
