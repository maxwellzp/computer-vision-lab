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


def calc_histogram(rgb_img: np.ndarray, channel_index: int) -> np.ndarray:
    """Calculate histogram for a specific RGB channel."""
    hist = cv.calcHist([rgb_img], [channel_index], None, [256], [0, 256])
    return hist.flatten()


def get_color_channels() -> list[tuple[int, str, str]]:
    return [
        (0, "r", "Red"),  # Red
        (1, "g", "Green"),  # Green
        (2, "b", "Blue"),  # Blue
    ]


def display_peak(name: str, hist) -> None:
    peak = np.argmax(hist)
    print(f"{name} peak intensity: {peak}")
    # Red peak intensity: 50
    # Green peak intensity: 73
    # Blue peak intensity: 48

    # blue channel - most pixels have intensity = 48
    # green channel - most pixels have intensity = 73
    # red channel - most pixels have intensity = 50

    # image contains mostly dark colors because:
    # 0-80 - dark colors
    # 80-170 - mid tones
    # 170-255 - bright


def display_histogram(rgb_img: np.ndarray) -> None:
    """Display histogram for RGB channels."""
    channels = get_color_channels()

    plt.figure()

    for channel_index, color, name in channels:
        hist = calc_histogram(rgb_img, channel_index)
        display_peak(name, hist)
        plt.plot(hist, color, label=name)

    plt.xlabel("Pixel intensity")
    plt.ylabel("Number of pixels")

    plt.title("Red-Green-Blue histogram")

    plt.legend()
    plt.xlim([0, 256])


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo.png"
    bgr_img = read_image(image_path)
    rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)

    plt.figure()
    plt.imshow(rgb_img)

    display_histogram(rgb_img)

    plt.show()
