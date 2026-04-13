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


def callback(input):
    pass


def canny_edge(img: np.ndarray):
    blur = cv.GaussianBlur(img, (5, 5), 0)

    window_name = "canny app"
    cv.namedWindow(window_name)

    cv.createTrackbar("min_thres", window_name, 50, 255, callback)
    cv.createTrackbar("max_thres", window_name, 150, 255, callback)

    while True:
        if cv.waitKey(1) == ord("q"):
            break

        min_thres = cv.getTrackbarPos("min_thres", window_name)
        max_thres = cv.getTrackbarPos("max_thres", window_name)

        if min_thres > max_thres:
            min_thres, max_thres = max_thres, min_thres

        edges = cv.Canny(blur, min_thres, max_thres)
        cv.imshow(window_name, edges)

    cv.destroyAllWindows()


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo_9.png"

    bgr_image = read_image(image_path)
    gray_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2GRAY)
    canny_edge(gray_image)
