import numpy as np
import cv2 as cv
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent

# BRIEF (Binary Robust Independent Elementary Features)


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from filesystem in BGR using OpenCV."""
    img = cv.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    return img


def brief_descriptor(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 1. detector (FAST)
    fast = cv.FastFeatureDetector_create(threshold=25)
    keypoints = fast.detect(gray, None)

    # 2. descriptor (BRIEF)
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
    keypoints, descriptors = brief.compute(gray, keypoints)

    result = cv.drawKeypoints(img, keypoints, None, color=(255, 0, 0))
    rgb = cv.cvtColor(result, cv.COLOR_BGR2RGB)

    print("Keypoints:", len(keypoints))
    print("Descriptor shape:", descriptors.shape)

    plt.figure(figsize=(8, 6))
    plt.title("BRIEF Keypoints")
    plt.imshow(rgb)
    plt.show()


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo_8.png"
    img = read_image(image_path)

    brief_descriptor(img)
