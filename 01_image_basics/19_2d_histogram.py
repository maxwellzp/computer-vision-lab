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


def histogram_2d(img: np.ndarray):
    rgb_image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    v_channel = hsv_image[:, :, 2]

    print("Min V:", v_channel.min())
    print("Max V:", v_channel.max())

    # 2D histogram: Hue + Saturation
    hist = cv.calcHist(
        [hsv_image],
        [0, 1],  # Hue and Saturation channels
        None,
        [180, 256],  # Hue: 0-180 and Saturation: 0-256
        [0, 180, 0, 256],
    )

    plt.figure()
    plt.subplot(131)
    plt.imshow(rgb_image)

    plt.subplot(132)
    plt.imshow(hist)
    plt.xlabel("Saturation")
    plt.ylabel("Hue")
    plt.colorbar()

    # X = Saturation
    # Y = Hue

    # (x,y) = 13.0, 4.3
    # (x,y) = 91.0, 22.1

    lower = np.array([13.0, 4.3, 50])
    upper = np.array([91.0, 22.1, 255])
    mask = cv.inRange(hsv_image, lower, upper)

    plt.subplot(133)
    plt.imshow(mask, cmap="gray")

    plt.show()


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo.png"
    bgr_img = read_image(image_path)
    histogram_2d(bgr_img)
