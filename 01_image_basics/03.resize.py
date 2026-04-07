import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from filesystem in BGR using OpenCV."""
    img = cv.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    return img


def resize_image(img: np.ndarray):
    """Resize an image using different methods in OpenCV."""
    # Method        CV Usage
    # NEAREST	    masks, labels
    # LINEAR	    default everywhere (90%)
    # CUBIC	        upscale images
    # AREA	        downscale dataset
    # LANCZOS4	    rare (slow)

    methods = [
        ("NEAREST", cv.INTER_NEAREST),
        ("LINEAR", cv.INTER_LINEAR),
        ("CUBIC", cv.INTER_CUBIC),
        ("AREA", cv.INTER_AREA),
        ("LANCZOS4", cv.INTER_LANCZOS4),
    ]

    print("Image shape:", img.shape[:2])
    # Image shape: (600, 800)

    # new_width  = original_width  * fx
    # new_height = original_height * fy

    scale_factor_X = 8
    scale_factor_Y = 8

    # img[y1:y2, x1:x2]
    # ---------Y1---Y2---X1--X2--
    crop = img[120:191, 480:566]
    for name, method in methods:
        resized = cv.resize(
            crop, None, fx=scale_factor_X, fy=scale_factor_Y, interpolation=method
        )

        cv.imshow(name, resized)

    cv.waitKey(0)

    cv.destroyAllWindows()


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo.png"
    img = read_image(image_path)
    # rgb_image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # plt.figure()
    # plt.imshow(rgb_image)
    # plt.show()
    resize_image(img)
