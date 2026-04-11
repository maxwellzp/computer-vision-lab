import numpy as np
from pathlib import Path
import cv2 as cv
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from filesystem in BGR using OpenCV."""
    img = cv.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    return img


def gaussian_filtering(img: np.ndarray) -> np.ndarray:
    blured_img = cv.GaussianBlur(img, (7, 7), 0)
    return blured_img


def callback(input) -> None:
    pass


# if __name__ == "__main__":
#     image_path = BASE_DIR / "data" / "images" / "photo.png"
#     bgr_image = read_image(image_path)
#     rgb_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)

#     blured_img = gaussian_filtering(bgr_image)
#     rgb_blured_img = cv.cvtColor(blured_img, cv.COLOR_BGR2RGB)

#     plt.figure()

#     plt.subplot(121)
#     plt.title("Original")
#     plt.imshow(rgb_image)

#     plt.subplot(122)
#     plt.title("Gaussian Blur")
#     plt.imshow(rgb_blured_img)

#     plt.show()

if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo.png"

    bgr_image = read_image(image_path)
    rgb_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)

    window_name = "Gaussian Blur"
    cv.namedWindow(window_name)
    cv.createTrackbar("kernel", window_name, 1, 50, callback)
    cv.createTrackbar("sigma", window_name, 0, 100, callback)
    while True:
        if cv.waitKey(1) == ord("q"):
            break

        kernel = cv.getTrackbarPos("kernel", window_name)
        sigma = cv.getTrackbarPos("sigma", window_name)
        print(f"Kernel: {kernel} | Sigma: {sigma}")

        if kernel % 2 == 0:
            kernel += 1
        kernel = max(1, kernel)

        image_filter = cv.GaussianBlur(bgr_image, (kernel, kernel), sigma)
        cv.imshow(window_name, image_filter)

    cv.destroyAllWindows()
