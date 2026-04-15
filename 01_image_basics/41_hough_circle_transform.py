import numpy as np
import cv2 as cv
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent

# gray -> blur -> HoughCircles -> draw


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from filesystem in BGR using OpenCV."""
    img = cv.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    return img


def hough_circle_transform(img: np.ndarray):
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    blurred = cv.GaussianBlur(gray, (9, 9), 1.5)

    circles = cv.HoughCircles(
        blurred,
        cv.HOUGH_GRADIENT,
        dp=1.2,
        minDist=150,
        param1=100,
        param2=40,
        minRadius=150,
        maxRadius=300,
    )

    if circles is None:
        print("No circles detected")
        return

    circles = np.uint16(np.around(circles))

    for x, y, r in circles[0]:
        cv.circle(rgb_img, (x, y), r, (255, 0, 0), 2)
        cv.circle(rgb_img, (x, y), 2, (0, 255, 0), 3)

    plt.imshow(rgb_img, cmap="gray")
    plt.title("Detected circles")

    plt.show()


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo_20.png"
    img = read_image(image_path)

    hough_circle_transform(img)
