import numpy as np
import cv2 as cv
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent

# Image -> Template -> matchTemplate -> minMaxLoc -> rectangle


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from filesystem in BGR using OpenCV."""
    img = cv.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    return img


def template_matching(img: np.ndarray):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # ROI (template)
    # img[y1:y2, x1:x2]
    template = gray[280:350, 330:420]
    h, w = template.shape

    plt.figure()

    plt.subplot(121)
    plt.title("Gray")
    plt.imshow(gray, cmap="gray")

    plt.subplot(122)
    plt.title("Template")
    plt.imshow(template, cmap="gray")

    plt.figure()

    methods = [
        ("TM_CCOEFF_NORMED", cv.TM_CCOEFF_NORMED),
        ("TM_CCORR_NORMED", cv.TM_CCORR_NORMED),
        ("TM_SQDIFF_NORMED", cv.TM_SQDIFF_NORMED),
        # ("TM_CCOEFF", cv.TM_CCOEFF),
        # ("TM_SQDIFF", cv.TM_SQDIFF),
    ]

    for i, (title, method) in enumerate(methods):
        temp_img = img.copy()

        result = cv.matchTemplate(gray, template, method)

        _, _, min_loc, max_loc = cv.minMaxLoc(result)

        if method == cv.TM_SQDIFF_NORMED:
            top_left = min_loc
        else:
            top_left = max_loc

        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv.rectangle(temp_img, top_left, bottom_right, (255, 0, 0), 2)

        res_norm = cv.normalize(result, None, 0, 1, cv.NORM_MINMAX)

        # Heatmap
        plt.subplot(len(methods), 2, i * 2 + 1)
        plt.title(title + " Heatmap")
        plt.imshow(res_norm, cmap="hot")

        # Result
        plt.subplot(len(methods), 2, i * 2 + 2)
        plt.title(title + " Detection")
        plt.imshow(cv.cvtColor(temp_img, cv.COLOR_BGR2RGB))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo_5.png"
    img = read_image(image_path)

    template_matching(img)
