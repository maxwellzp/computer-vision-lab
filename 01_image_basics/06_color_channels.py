import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def display_rgb_and_wb_colors():
    # Create a matrix filled with 0
    zeros = np.zeros((100, 100), dtype=np.uint8)

    max_val = np.full((100, 100), 255, dtype=np.uint8)

    # Merge color channels into a single 3-channel image.
    # Matplotlib expects RGB
    # (RED---GREEN---BLUE)
    r_img = cv.merge((max_val, zeros, zeros))
    g_img = cv.merge((zeros, max_val, zeros))
    b_img = cv.merge((zeros, zeros, max_val))

    black_img = np.zeros((100, 100, 3), dtype=np.uint8)
    white_img = np.full((100, 100, 3), 255, dtype=np.uint8)

    plt.figure(figsize=(12, 6))

    images = [r_img, g_img, b_img, black_img, white_img]
    titles = ["Red (RGB)", "Green (RGB)", "Blue (RGB)", "Black", "White"]

    for i in range(5):
        # R — rows
        # C — columns
        # I — index
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i])
        plt.title(titles[i])

    plt.tight_layout()
    plt.show()


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from file using OpenCV."""
    img = cv.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    return img


def display_image_in_different_channels():
    image_path = BASE_DIR / "data" / "images" / "photo.png"

    # Reading an image in BGR format
    img_bgr = read_image(image_path)

    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

    r, g, b = cv.split(img_rgb)
    # (600,800,3)
    # =>
    # b -> (600,800)
    # g -> (600,800)
    # r -> (600,800)

    zeros = np.zeros_like(r)

    # Assemble together 3-channel image
    r_viz = cv.merge((r, zeros, zeros))
    g_viz = cv.merge((zeros, g, zeros))
    b_viz = cv.merge((zeros, zeros, b))

    data = [(r_viz, "Red Channel"), (g_viz, "Green Channel"), (b_viz, "Blue Channel")]

    plt.figure(figsize=(15, 5))

    for i, (img, title) in enumerate(data):
        # R — rows
        # C — columns
        # I — index
        plt.subplot(1, 3, i + 1)
        plt.title(title)
        plt.imshow(img)

    plt.show()


if __name__ == "__main__":
    display_rgb_and_wb_colors()
    display_image_in_different_channels()
