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


def image_gradient(img: np.ndarray):
    plt.figure()
    plt.subplot(221)
    plt.title("Original")
    plt.imshow(img, cmap="gray")

    laplacian = cv.Laplacian(img, cv.CV_64F, ksize=3)
    laplacian = cv.convertScaleAbs(laplacian)
    plt.subplot(222)
    plt.title("Laplacian")
    plt.imshow(laplacian, cmap="gray")

    kx, ky = cv.getDerivKernels(1, 0, 3)
    print(ky @ kx.T)
    #  [[-1.  0.  1.]
    #  [-2.  0.  2.]
    #  [-1.  0.  1.]]

    sobelX = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
    sobelX = cv.convertScaleAbs(sobelX)
    plt.subplot(223)
    plt.title("Sobel X")
    plt.imshow(sobelX, cmap="gray")

    sobelY = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
    sobelY = cv.convertScaleAbs(sobelY)
    plt.subplot(224)
    plt.title("Sobel Y")
    plt.imshow(sobelY, cmap="gray")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo_9.png"

    bgr_image = read_image(image_path)
    gray_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2GRAY)
    image_gradient(gray_image)
