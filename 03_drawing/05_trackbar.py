import cv2 as cv
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
RECT_START = (100, 150)
RECT_END = (400, 450)


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from filesystem in BGR using OpenCV."""
    img = cv.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    return img


def trackbar_callback(value) -> None:
    """Required OpenCV callback."""
    pass


def setup_callbacks(window_name: str) -> None:
    """Create three trackabars for each BGR channel."""
    cv.createTrackbar("B", window_name, 255, 255, trackbar_callback)
    cv.createTrackbar("G", window_name, 255, 255, trackbar_callback)
    cv.createTrackbar("R", window_name, 255, 255, trackbar_callback)


def get_color(window_name: str) -> tuple[int, int, int]:
    """Get a current value for each trackbar."""
    b = cv.getTrackbarPos("B", window_name)
    g = cv.getTrackbarPos("G", window_name)
    r = cv.getTrackbarPos("R", window_name)
    return (b, g, r)


def create_window(window_name: str) -> None:
    """Create a window with specified name."""
    cv.namedWindow(window_name)


def draw_image_and_trackbars(img: np.ndarray, window_name: str) -> None:
    """Draw an image with trackbars and apply all changes immediately."""
    create_window(window_name)
    setup_callbacks(window_name)

    while True:
        frame = img.copy()

        color = get_color(window_name)

        cv.rectangle(frame, RECT_START, RECT_END, color, -1)

        cv.imshow(window_name, frame)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cv.destroyAllWindows()


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo.png"
    img = read_image(image_path)
    window_name = "trackbar window"
    draw_image_and_trackbars(img, window_name)
