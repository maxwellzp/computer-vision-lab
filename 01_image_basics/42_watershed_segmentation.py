import numpy as np
import cv2 as cv
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent

# grayscale -> threshold -> noise removal (morphology) -> distance transform ->
# -> foreground -> background -> markers -> watershed


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from filesystem in BGR using OpenCV."""
    img = cv.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    return img


def watershed_segmentation(img: np.ndarray):
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 1. threshold (+ Otsu)
    _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # Create a kernel
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

    # 2. remove noise
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

    # 3. sure background
    sure_bg = cv.dilate(opening, kernel, iterations=3)

    # 4. distance transform
    dist = cv.distanceTransform(opening, cv.DIST_L2, 5)

    # 5. sure foreground
    _, sure_fg = cv.threshold(dist, 0.5 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # 6. unknown region
    unknown = cv.subtract(sure_bg, sure_fg)

    # 7. markers
    _, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # 8. watershed
    markers = cv.watershed(img, markers)

    # boundaries
    result = rgb_img.copy()
    result[markers == -1] = [255, 0, 0]

    plt.figure(figsize=(10, 6))

    plt.subplot(231)
    plt.title("Original")
    plt.imshow(rgb_img)

    plt.subplot(232)
    plt.title("Threshold")
    plt.imshow(thresh, cmap="gray")

    plt.subplot(233)
    plt.title("Opening")
    plt.imshow(opening, cmap="gray")

    plt.subplot(234)
    plt.title("Distance transform")
    plt.imshow(dist, cmap="jet")

    plt.subplot(235)
    plt.title("Markers")
    plt.imshow(markers, cmap="nipy_spectral")

    plt.subplot(236)
    plt.title("Result")
    plt.imshow(result)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo_21.png"
    img = read_image(image_path)

    watershed_segmentation(img)
