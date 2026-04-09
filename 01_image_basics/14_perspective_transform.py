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


def display_images_on_plot(img_1: np.ndarray, img_2: np.ndarray) -> None:
    plt.figure()

    plt.subplot(121)
    plt.title("Original")
    plt.imshow(img_1)

    plt.subplot(122)
    plt.title("Perspective Transform")
    plt.imshow(img_2)

    plt.show()


def transform(img: np.ndarray) -> np.ndarray:
    # Perspective:
    # 3x3 matrix
    #
    # Matrix:
    # [a b c]
    # [d e f]
    # [g h i]
    #
    # np.float32([
    #   [x1,y1],
    #   [x2,y2],
    #   [x3,y3],
    #   [x4,y4]
    # ])

    # top-left
    # top-right
    # bottom-left
    # bottom-right
    #

    height, width = img.shape[:2]
    print(f"Height: {height} | Width: {width}")
    # Height: 600 | Width: 800

    src = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])

    dst = np.float32(
        [[50, 50], [width - 100, 20], [20, height - 50], [width - 50, height - 20]]
    )

    M = cv.getPerspectiveTransform(src, dst)
    transformed_img = cv.warpPerspective(img, M, ((width, height)))
    return transformed_img


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo.png"
    bgr_img = read_image(image_path).copy()
    transformed_img = transform(bgr_img)

    rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
    rgb_transformed_img = cv.cvtColor(transformed_img, cv.COLOR_BGR2RGB)

    display_images_on_plot(rgb_img, rgb_transformed_img)
