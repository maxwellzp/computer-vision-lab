import numpy as np
import cv2 as cv
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent

# Good Corner Detection (Shi–Tomasi)


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from filesystem in BGR using OpenCV."""
    img = cv.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    return img


def good_corner_detection(img: np.ndarray):
    rgb_image = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    max_corners = 200
    quality = 0.01
    min_distance = 15

    corners = cv.goodFeaturesToTrack(gray_img, max_corners, quality, min_distance)

    if corners is None:
        print("No corners found")
        return

    corners = np.int32(corners)

    result = rgb_image.copy()

    for x, y in corners.reshape(-1, 2):
        cv.circle(result, (x, y), 3, (255, 0, 0), -1)

    plt.figure()
    plt.title("Good Features to Track")
    plt.imshow(result)
    plt.show()


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo_8.png"
    img = read_image(image_path)

    good_corner_detection(img)
