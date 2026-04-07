import cv2 as cv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from filesystem in BGR using OpenCV."""
    img = cv.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    return img


def translation(img: np.ndarray, tx: int, ty: int):
    """Shift image by tx and ty."""
    height, width = img.shape[:2]
    print(f"Height:{height} | Width: {width}")

    # M = [
    #   [a  b  tx]
    #   [c  d  ty]
    # ]
    # x' = a*x + b*y + tx
    # y' = c*x + d*y + ty
    #

    # a = scale X
    # b = rotation/skew
    # c = rotation/skew
    # d = scale Y
    # tx = shift X
    # ty = shift Y

    M = np.float32([[1, 0, tx], [0, 1, ty]])

    shifted = cv.warpAffine(
        img,
        M,
        (width, height),
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=(255, 0, 0),
    )

    cv.imshow("Original", img)
    cv.imshow("Shifted", shifted)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo.png"
    img = read_image(image_path)

    tx = 100  # shift right
    ty = 150  # shift down
    translation(img, tx, ty)
