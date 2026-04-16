import numpy as np
import cv2 as cv
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from filesystem in BGR using OpenCV."""
    img = cv.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    return img


def graph_cut_segmentation(img: np.ndarray):
    rgb_image = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    rows, cols = img.shape[:2]

    mask = np.zeros((rows, cols), np.uint8)

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    x0 = 153
    y0 = 209
    x1 = 531
    y1 = 416
    rect = (x0, y0, x1 - x0, y1 - y0)

    cv.grabCut(img, mask, rect, bg_model, fg_model, 5, cv.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD), 1, 0).astype("uint8")

    result = rgb_image * mask2[:, :, np.newaxis]

    plt.figure(figsize=(10, 4))

    plt.subplot(131)
    plt.title("Original")
    plt.imshow(rgb_image)

    plt.subplot(132)
    plt.title("Mask")
    plt.imshow(mask2, cmap="gray")

    plt.subplot(133)
    plt.title("Segmented")
    plt.imshow(result)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo_8.png"
    img = read_image(image_path)

    graph_cut_segmentation(img)
