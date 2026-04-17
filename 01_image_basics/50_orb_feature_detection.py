import numpy as np
import cv2 as cv
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent

# ORB (Oriented FAST and Rotated BRIEF)


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from filesystem in BGR using OpenCV."""
    img = cv.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    return img


def orb_feature_detection(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    orb = cv.ORB.create(nfeatures=1000)

    keypoints, descriptors = orb.detectAndCompute(gray, None)

    print("Keypoints:", len(keypoints))

    if descriptors is not None:
        print("Descriptor shape:", descriptors.shape)
    else:
        print("No descriptors found")

    result = cv.drawKeypoints(
        img, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    rgb_img = cv.cvtColor(result, cv.COLOR_BGR2RGB)

    plt.figure()
    plt.title("Result of ORB")
    plt.imshow(rgb_img)

    plt.show()


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo_8.png"
    img = read_image(image_path)

    orb_feature_detection(img)
