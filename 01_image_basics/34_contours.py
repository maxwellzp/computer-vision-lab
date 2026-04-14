from pathlib import Path
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

# Image -> Grayscale -> Blur -> Otsu Threshold -> Contours -> Filtering -> Bounding Boxes


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from filesystem in BGR using OpenCV."""
    img = cv.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    return img


def contours(img: np.ndarray):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    blur = cv.GaussianBlur(gray_img, (5, 5), 0)

    _, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv.contourArea, reverse=True)

    draw_img = img.copy()

    for cnt in contours:
        if cv.contourArea(cnt) < 500:
            continue

        cv.drawContours(draw_img, [cnt], -1, (0, 255, 0), 2)

        x, y, w, h = cv.boundingRect(cnt)
        cv.rectangle(draw_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    plt.figure(figsize=(10, 4))

    plt.subplot(221)
    plt.title("Original")
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

    plt.subplot(222)
    plt.title("Threshold")
    plt.imshow(thresh, cmap="gray")

    plt.subplot(223)
    plt.title("Contours")
    plt.imshow(cv.cvtColor(draw_img, cv.COLOR_BGR2RGB))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo_12.png"
    img = read_image(image_path)

    contours(img)
