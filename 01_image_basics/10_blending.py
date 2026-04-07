import cv2 as cv
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

# dst = src1 * alpha + src2 * beta + gamma


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from filesystem in BGR using OpenCV."""
    img = cv.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    return img


def blend_two_images(img_1: np.ndarray, alpha: float, img_2: np.ndarray, beta: float):
    """Blending two images using OpenCV."""
    if img_1.shape != img_2.shape:
        print("Image sizes:", img_1.shape, img_2.shape)
        raise ValueError("Images must have same size")
    # alpha = 0.7
    # beta = 0.3
    # =>
    # 70% image1
    # 30% image2
    return cv.addWeighted(img_1, alpha, img_2, beta, 0)


def resize_image(img: np.ndarray, size: tuple[int, int]):
    return cv.resize(img, (width, height))


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo.png"
    img_1 = read_image(image_path).copy()
    image_path = BASE_DIR / "data" / "images" / "photo_3.png"
    img_2 = read_image(image_path).copy()

    print("Image #1", img_1.shape)
    # Image #1 (600, 800, 3)

    print("Image #2", img_2.shape)
    # Image #2 (628, 1200, 3)

    width, height = img_1.shape[1], img_1.shape[0]
    resized_img = resize_image(img_2, (width, height))

    alpha = 0.7
    beta = 0.3
    result = blend_two_images(img_1, alpha, resized_img, beta)
    cv.imshow("Blending result", result)
    cv.waitKey(0)
    cv.destroyAllWindows()
