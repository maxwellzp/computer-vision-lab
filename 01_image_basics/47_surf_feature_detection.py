import numpy as np
import cv2 as cv
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from filesystem in BGR using OpenCV."""
    img = cv.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    return img


def surf_feature_detection(img: np.ndarray):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    hessian_threshold = 400

    detector = cv.xfeatures2d.SURF_create(hessian_threshold)

    keypoints, descriptors = detector.detectAndCompute(gray, None)

    result = cv.drawKeypoints(
        img, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    rgb = cv.cvtColor(result, cv.COLOR_BGR2RGB)

    plt.figure()
    plt.title("SURF Keypoints")
    plt.imshow(rgb)
    plt.show()


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo_8.png"
    img = read_image(image_path)

    surf_feature_detection(img)
