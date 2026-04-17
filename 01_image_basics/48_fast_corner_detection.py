import numpy as np
import cv2 as cv
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent

# FAST (Features from Accelerated Segment Test)


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from filesystem in BGR using OpenCV."""
    img = cv.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    return img


def fast_detector(img: np.ndarray):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    fast = cv.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)

    keypoints = fast.detect(gray, None)

    result = cv.drawKeypoints(img, keypoints, None, color=(255, 0, 0))

    rgb = cv.cvtColor(result, cv.COLOR_BGR2RGB)

    print("Keypoints:", len(keypoints))
    print("Threshold:", fast.getThreshold())
    print("Nonmax:", fast.getNonmaxSuppression())

    # Keypoints: 2664
    # Threshold: 25
    # Nonmax: True

    plt.figure()
    plt.title("FAST Corner Detection")
    plt.imshow(rgb)
    plt.show()


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo_8.png"
    img = read_image(image_path)

    fast_detector(img)
