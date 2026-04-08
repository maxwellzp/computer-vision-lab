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


def transform_image(img: np.ndarray):
    height, width = img.shape[:2]
    print(f"Height: {height} | Width: {width}")

    # Affine transform:
    # x' = ax + by + tx
    # y' = cx + dy + ty

    # Matrix:
    # [a b tx]
    # [c d ty]

    # Pixel coordinates:
    # np.array([
    #     [x1,y1],
    #     [x2,y2],
    #     [x3,y3]
    # ], dtype=np.float32)

    # top left
    # top right
    # bottom left

    # width/height - 1 because indexes start from zero
    # max x coordinate = width - 1
    # max y coordinate = height - 1
    src = np.float32(
        [
            [0, 0],  # top left
            [width - 1, 0],  # top right
            [0, height - 1],  # bottom left
        ]
    )

    dst = np.float32(
        [
            [100, 100],  # top left
            [width - 100, 20],  # top right
            [20, height - 50],  # bottom left
        ]
    )

    T = cv.getAffineTransform(src, dst)

    transformed_img = cv.warpAffine(img, T, (width, height), flags=cv.INTER_LINEAR)
    return transformed_img


def display_images_on_plot(img_1: np.ndarray, img_2: np.ndarray) -> None:
    plt.figure()

    plt.subplot(121)
    plt.title("Original")
    plt.imshow(img_1)

    plt.subplot(122)
    plt.title("Affine transform")
    plt.imshow(img_2)

    plt.show()


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo.png"
    bgr_img = read_image(image_path).copy()

    transformed_img = transform_image(bgr_img)

    rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
    rgb_transformed_img = cv.cvtColor(transformed_img, cv.COLOR_BGR2RGB)
    display_images_on_plot(rgb_img, rgb_transformed_img)
