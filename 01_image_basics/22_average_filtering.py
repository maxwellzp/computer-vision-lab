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


def callback(value: int) -> None:
    """Trackbar callback required by OpenCV."""
    pass


def average_filtering(img: np.ndarray, window_name: str):
    """Apply average filtering."""
    cv.namedWindow(window_name)
    cv.createTrackbar("filter", window_name, 3, 100, callback)

    while True:
        if cv.waitKey(1) == ord("q"):
            break

        n = cv.getTrackbarPos("filter", window_name)

        n = max(1, n)
        if n % 2 == 0:
            n += 1

        filtered_img = cv.blur(img, (n, n))  # average filtering (mean filter)
        cv.imshow(window_name, filtered_img)

    cv.destroyAllWindows()


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo.png"
    bgr_image = read_image(image_path)
    window_name = "Average Filtering"
    average_filtering(bgr_image, window_name)
