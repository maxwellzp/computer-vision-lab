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


def bilateral_filtering(
    img: np.ndarray,
    d: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75,
) -> np.ndarray:
    # option 1
    # filtered_img = cv.bilateralFilter(img, 9, 75, 75)

    # option 2
    # filtered_img = cv.bilateralFilter(img, 15, 75, 75)

    # option 3
    filtered_img = cv.bilateralFilter(
        img, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space
    )

    return filtered_img


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo.png"

    bgr_image = read_image(image_path)
    rgb_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)

    filtered_img = bilateral_filtering(bgr_image, 5, 50, 50)
    rgb_filtered_img = cv.cvtColor(filtered_img, cv.COLOR_BGR2RGB)

    plt.figure()

    plt.subplot(121)
    plt.title("Original image")
    plt.imshow(rgb_image)

    plt.subplot(122)
    plt.title("Bilateral filtering")
    plt.imshow(rgb_filtered_img)

    plt.show()
