import cv2 as cv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent


def display_image(img: np.ndarray) -> None:
    """Display an image using Matplotlib."""
    plt.figure()
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title("Image")
    plt.show()


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from filesystem using OpenCV."""
    img = cv.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    return img


def write_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    font_scale: float = 1,
    font_face: int = cv.FONT_HERSHEY_SIMPLEX,
    line_type: int = cv.LINE_AA,
) -> None:
    """Write a text on an image using OpenCV."""
    cv.putText(
        img=img,
        text=text,
        org=org,
        fontFace=font_face,
        fontScale=font_scale,
        color=color,
        thickness=thickness,
        lineType=line_type,
    )


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo_2.png"
    img = read_image(image_path).copy()
    # org = (x, y)
    # org = bottom-left corner of text

    bgr_color = (0, 0, 255)
    write_text(
        img=img,
        text="Python + CV",
        org=(150, 250),
        color=bgr_color,
        thickness=2,
        font_scale=2.5,
        font_face=cv.FONT_HERSHEY_SIMPLEX,
        line_type=cv.LINE_AA,
    )
    display_image(img)
