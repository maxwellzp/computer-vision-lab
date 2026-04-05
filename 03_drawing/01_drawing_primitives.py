import cv2 as cv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from filesystem using OpenCV."""
    img = cv.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    return img


def display_image(img: np.ndarray) -> None:
    """Display an image using Matplotlib."""
    plt.figure()
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title("Image")
    plt.show()


def draw_line(
    img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: tuple
) -> None:
    """Draw a line using OpenCV."""
    # format = x1,y1,x2,y2
    cv.line(img=img, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=2)


def draw_rectangle(
    img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: tuple
) -> None:
    """Draw a rectangle using OpenCV."""
    # format = x1,y1,x2,y2
    cv.rectangle(img=img, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=-1)


def draw_circle(img: np.ndarray, x: int, y: int, radius: int, color: tuple) -> None:
    """Draw a circle using OpenCV."""
    cv.circle(img=img, center=(x, y), radius=radius, color=color, thickness=-1)


def draw_polygon(img: np.ndarray, points: np.ndarray, color: tuple) -> None:
    """Draw a polygon contour using OpenCV."""
    cv.polylines(img=img, pts=[points], isClosed=True, color=color, thickness=2)


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo_2.png"
    img = read_image(image_path).copy()
    display_image(img)

    # BGR
    contour_color = (0, 0, 255)

    draw_line(img=img, x1=428, y1=100, x2=680, y2=250, color=contour_color)

    draw_rectangle(img=img, x1=50, y1=100, x2=300, y2=250, color=contour_color)

    draw_circle(img=img, x=175, y=380, radius=100, color=contour_color)

    points = np.array([[568, 281], [702, 380], [568, 471], [436, 380]], dtype=np.int32)
    draw_polygon(img, points, contour_color)

    display_image(img)
