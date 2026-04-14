from pathlib import Path
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from filesystem in BGR using OpenCV."""
    img = cv.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    return img


def display_center(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if not contours:
        return

    cnt = max(contours, key=cv.contourArea)

    M = cv.moments(cnt)

    if M["m00"] == 0:
        return

    # m00 - (Area): Sum of pixel intensities, representing the area of a binary shape.
    # m10 - (Horizontal Centroid Moment): Sum of x-coordinates, used for x-centroid.
    # m01 - (Vertical Centroid Moment): Sum of y-coordinates, used for y-centroid.

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    draw = img.copy()

    cv.circle(draw, (cx, cy), 5, (0, 0, 255), -1)

    rgb_image = cv.cvtColor(draw, cv.COLOR_BGR2RGB)
    plt.title("Center of contours")
    plt.imshow(rgb_image)
    plt.show()


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo_12.png"
    img = read_image(image_path)

    display_center(img)
