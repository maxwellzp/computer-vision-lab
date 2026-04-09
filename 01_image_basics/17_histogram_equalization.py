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


def calc_hist(img: np.ndarray) -> np.ndarray:
    """Calculate histogram for specific image."""
    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    return hist.flatten()


def calc_hist_and_cdf(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate histogram and normalized CDF for an image."""
    hist = calc_hist(img)
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    return hist, cdf_normalized


def plot_histogram(
    position: int,
    title: str,
    hist: np.ndarray,
    cdf: np.ndarray,
    hist_label: str,
    cdf_label: str,
):
    plt.subplot(position)
    plt.title(title)
    plt.plot(hist, label=hist_label)
    plt.plot(cdf, label=cdf_label)
    plt.xlabel("Pixel intensity")
    plt.ylabel("Pixels")
    plt.legend()


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo_4.png"
    bgr_img = read_image(image_path)
    gray_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)

    hist, cdf_normalized = calc_hist_and_cdf(gray_img)

    plt.figure(figsize=(12, 8))

    # Display original image
    plt.subplot(221)
    plt.title("Original image")
    plt.imshow(gray_img, cmap="gray")
    plt.axis("off")

    # Display histogram of the original image
    plot_histogram(
        position=222,
        title="Histogram of original image",
        hist=hist,
        hist_label="Histogram",
        cdf=cdf_normalized,
        cdf_label="CDF",
    )

    equalized_image = cv.equalizeHist(gray_img)

    equalized_hist, equalized_cdf_normalized = calc_hist_and_cdf(equalized_image)

    # Display equalized image
    plt.subplot(223)
    plt.title("Equalized image")
    plt.imshow(equalized_image, cmap="gray")
    plt.axis("off")

    # Display histogram of the equalized image
    plot_histogram(
        position=224,
        title="Histogram of equalized image",
        hist=equalized_hist,
        hist_label="Histogram",
        cdf=equalized_cdf_normalized,
        cdf_label="CDF",
    )

    plt.tight_layout()
    plt.show()
