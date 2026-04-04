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


def read_roi_by_coordinates(
    rgb_image: np.ndarray, y_start: int, y_end: int, x_start: int, x_end: int
) -> np.ndarray:
    """Read ROI from image using OpenCV."""

    height, width = rgb_image.shape[:2]

    if not (0 <= y_start < y_end <= height):
        raise ValueError("Invalid Y coordinates")

    if not (0 <= x_start < x_end <= width):
        raise ValueError("Invalid X coordinates")

    # array[start:end]
    # (y_start,x_start) ---- (y_start,x_end)
    #         |                     |
    #         |        ROI          |
    #         |                     |
    # (y_end,x_start) ---- (y_end,x_end)
    roi = rgb_image[y_start:y_end, x_start:x_end]

    print("Region Of Interest:", roi)
    #     Region Of Interest: [[[ 41  58  40]
    #   [ 41  58  40]
    #   [ 42  59  41]
    #   ...
    #   [ 61  86  57]
    #   [ 61  86  57]
    #   [ 61  86  57]]

    #  [[ 41  58  40]
    #   [ 41  58  40]
    #   [ 42  59  41]
    #   ...
    #   [ 61  86  57]
    #   [ 61  86  57]
    #   [ 61  86  57]]

    #  [[ 43  57  40]
    #   [ 44  58  41]
    #   [ 44  58  41]
    #   ...
    #   [ 62  87  58]
    #   [ 62  87  58]
    #   [ 62  87  58]]

    #  ...

    #  [[ 88  88  76]
    #   [ 88  88  76]
    #   [ 88  88  76]
    #   ...
    #   [ 83 107  75]
    #   [ 83 107  75]
    #   [ 83 107  75]]

    #  [[ 88  88  76]
    #   [ 88  88  76]
    #   [ 87  87  75]
    #   ...
    #   [ 83 107  75]
    #   [ 83 107  75]
    #   [ 83 107  75]]

    #  [[ 87  87  75]
    #   [ 87  87  75]
    #   [ 87  87  75]
    #   ...
    #   [ 83 107  75]
    #   [ 83 107  75]
    #   [ 83 107  75]]]

    return roi


def write_to_roi_by_coordinates(
    rgb_image: np.ndarray,
    y_start: int,
    y_end: int,
    x_start: int,
    x_end: int,
    color: tuple,
) -> None:
    """Write ROI to an image using OpenCV."""
    rgb_image[y_start:y_end, x_start:x_end] = color


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

    # Coordinates of a pixel region
    y_start = 0
    y_end = 100
    x_start = 100
    x_end = 300
    read_roi_by_coordinates(rgb_image, y_start, y_end, x_start, x_end)
    write_to_roi_by_coordinates(rgb_image, y_start, y_end, x_start, x_end, (255, 0, 0))
    read_roi_by_coordinates(rgb_image, y_start, y_end, x_start, x_end)
    display_image(rgb_image)
