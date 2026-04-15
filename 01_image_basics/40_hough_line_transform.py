import numpy as np
import cv2 as cv
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent

# Grayscale -> Canny (edges) -> Hough Transform -> Draw lines


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from filesystem in BGR using OpenCV."""
    img = cv.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    return img


def hough_line_transform(img: np.ndarray):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred_img = cv.GaussianBlur(gray, (5, 5), 0)
    canny_edges = cv.Canny(blurred_img, 50, 150)

    plt.figure()

    plt.subplot(141)
    plt.title("Gray")
    plt.imshow(gray, cmap="gray")

    plt.subplot(142)
    plt.title("Blur")
    plt.imshow(blurred_img, cmap="gray")

    plt.subplot(143)
    plt.title("Canny edges")
    plt.imshow(canny_edges, cmap="gray")

    rho_res = 1
    theta_res = np.pi / 180
    threshold = 80

    lines = cv.HoughLines(canny_edges, rho_res, theta_res, threshold)

    draw = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

    if lines is not None:
        for cur_line in lines:
            rho, theta = cur_line[0]

            a = np.cos(theta)
            b = np.sin(theta)

            x0 = a * rho
            y0 = b * rho

            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv.line(draw, (x1, y1), (x2, y2), (0, 0, 255), 2)
    else:
        print("No lines detected.")

    plt.subplot(144)
    plt.title("Hough Lines")
    plt.imshow(cv.cvtColor(draw, cv.COLOR_BGR2RGB))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo_18.png"
    img = read_image(image_path)

    hough_line_transform(img)
