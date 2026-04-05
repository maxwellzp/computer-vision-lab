import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from file using OpenCV."""
    img = cv.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    print_shape(img)

    return cv.cvtColor(img, cv.COLOR_BGR2RGB)


def display_image(rgb_image: np.ndarray) -> None:
    """Display RGB image using Matplotlib."""
    plt.figure()
    plt.imshow(rgb_image)
    plt.show()


def read_pixel_by_coordinates(rgb_image: np.ndarray, y: int, x: int) -> np.ndarray:
    """Read a pixel from image using OpenCV."""
    height, width = rgb_image.shape[:2]
    print(f"h={height} | w={width}")
    if not (0 <= y < height and 0 <= x < width):
        raise ValueError("Out of bounds")

    # Read a pixel img[y, x]
    one_pixel = rgb_image[y, x]
    print("Pixel value:", one_pixel)
    # pixel: [55 79 57]
    return one_pixel


def write_to_pixel_by_coordinates(
    rgb_image: np.ndarray, y: int, x: int, color: tuple
) -> None:
    """Write a pixel in an image using OpenCV."""
    rgb_image[y, x] = color


def print_shape(img: np.ndarray) -> None:
    """Print a shape of an image."""
    print(img.shape)
    # (height, width, channels)
    # (600   , 800  ,    3    )
    # 600 - Y
    # 800 - X
    # 3   - RGB/BGR


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo.png"
    rgb_image = read_image(image_path)
    display_image(rgb_image)

    y = 524
    x = 755
    read_pixel_by_coordinates(rgb_image, y, x)
    write_to_pixel_by_coordinates(rgb_image, y, x, (255, 0, 0))
    read_pixel_by_coordinates(rgb_image, y, x)
    display_image(rgb_image)
