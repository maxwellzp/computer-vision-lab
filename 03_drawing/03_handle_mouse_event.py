import cv2 as cv
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from filesystem in BGR using OpenCV."""
    img = cv.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    return img


def draw_circle(event, x, y, flags, param):
    """Mouse callback. Draw a blue circle on left mouse button click."""
    img: np.ndarray = param
    if event == cv.EVENT_LBUTTONDOWN:
        # -------------------------Blue in BGR
        cv.circle(img, (x, y), 25, (255, 0, 0), -1)


def mouse_drawing(image_path: Path):
    """Setup mouse callback and allow drawing circles interactively."""
    img = read_image(image_path)
    window_name = "drawing circle"

    cv.namedWindow(window_name)
    cv.setMouseCallback(window_name, draw_circle, img)

    while True:
        cv.imshow(window_name, img)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cv.destroyAllWindows()


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo.png"
    mouse_drawing(image_path)
