import cv2 as cv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from filesystem in BGR using OpenCV."""
    img = cv.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    return img


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo.png"
    bgr_img = read_image(image_path).copy()
    gray_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)

    plt.figure()
    plt.imshow(gray_img, cmap="gray")

    # x (pixel intensity or brightness) = 0 - 255
    # 0   = black
    # 64  = dark gray
    # 128 = medium gray
    # 192 = light gray
    # 255 = white

    # y (number of pixels)

    # dominant intensity = (x, y) = (64, 9630)
    # 9630 pixels have brightness 64

    hist = cv.calcHist([gray_img], [0], None, [256], [0, 256])
    hist = hist.flatten()

    plt.figure()
    plt.plot(hist)
    plt.xlabel("Pixel intensity")
    plt.ylabel("Number of pixels")
    plt.title("Grayscale histogram")

    plt.xlim([0, 256])

    print(np.argmax(hist))
    # 64
    plt.show()


# 1) dark image
#    underexposed
# |
# | ███
# | █████
# | ███████
# |____________
# 0          255
#
#
# 2) bright image
#    overexposed
# |
# |          ███
# |        █████
# |      ███████
# |____________
# 0          255
#
#
#
# 3) normal exposure
# |
# |     ████
# |   ███████
# |     ████
# |____________
# 0          255
