from pathlib import Path
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# # ROI → histogram → backprojection → blur → threshold → morphology → mask → result

BASE_DIR = Path(__file__).resolve().parent.parent


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from filesystem in BGR using OpenCV."""
    img = cv.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    return img


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo_9.png"

    bgr_image = read_image(image_path)
    rgb_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)
    hsv_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2HSV)

    # plt.imshow(rgb_image)
    # plt.show()

    # img[y1:y2, x1:x2]
    # roi = bgr_image[280:330, 185:290]
    roi = bgr_image[150:300, 200:350]
    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

    mask_roi = cv.inRange(hsv_roi, (0, 50, 50), (180, 255, 255))

    # [0, 1] = H и S channels
    roi_hist = cv.calcHist([hsv_roi], [0, 1], mask_roi, [30, 32], [0, 180, 0, 256])

    cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

    back_proj = cv.calcBackProject([hsv_image], [0, 1], roi_hist, [0, 180, 0, 256], 1)

    back_proj = cv.GaussianBlur(back_proj, (9, 9), 0)

    # Create a mask
    _, mask = cv.threshold(back_proj, 40, 255, cv.THRESH_BINARY)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    mask_3ch = cv.merge((mask, mask, mask))
    segmented_image = cv.bitwise_and(rgb_image, mask_3ch)

    plt.figure(figsize=(12, 6))

    plt.subplot(231)
    plt.title("Original")
    plt.imshow(rgb_image)

    plt.subplot(232)
    plt.title("ROI")
    plt.imshow(cv.cvtColor(roi, cv.COLOR_BGR2RGB))

    plt.subplot(233)
    plt.title("BackProjection")
    plt.imshow(back_proj, cmap="gray")

    plt.subplot(234)
    plt.title("Mask")
    plt.imshow(mask, cmap="gray")

    plt.subplot(235)
    plt.title("Segmented")
    plt.imshow(segmented_image)

    plt.tight_layout()
    plt.show()
