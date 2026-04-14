from pathlib import Path
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

# Image -> DFT -> Shift -> Mask (low-pass) -> Inverse DFT


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from filesystem in BGR using OpenCV."""
    img = cv.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    return img


def fourier_transform(img: np.ndarray):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    plt.figure(figsize=(10, 6))

    plt.subplot(231)
    plt.title("Original")
    plt.imshow(img, cmap="gray")

    # DFT
    img_dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)

    magnitude = cv.magnitude(img_dft[:, :, 0], img_dft[:, :, 1])
    img_dft_db = 20 * np.log(magnitude + 1)

    plt.subplot(232)
    plt.title("DFT")
    plt.imshow(img_dft_db, cmap="gray")

    # Shift
    img_dft_shift = np.fft.fftshift(img_dft)

    magnitude_shift = cv.magnitude(img_dft_shift[:, :, 0], img_dft_shift[:, :, 1])
    img_dft_shift_db = 20 * np.log(magnitude_shift + 1)

    plt.subplot(233)
    plt.title("Shifted DFT")
    plt.imshow(img_dft_shift_db, cmap="gray")

    # Mask
    r, c = img.shape
    mask = np.zeros((r, c, 2), np.float32)
    offset = 50

    mask[
        r // 2 - offset : r // 2 + offset,
        c // 2 - offset : c // 2 + offset,
    ] = 1

    plt.subplot(234)
    plt.title("Mask")
    plt.imshow(mask[:, :, 0], cmap="gray")

    # Apply mask
    img_dft_shift_lp = img_dft_shift * mask

    magnitude_lp = cv.magnitude(img_dft_shift_lp[:, :, 0], img_dft_shift_lp[:, :, 1])
    img_dft_lp_db = 20 * np.log(magnitude_lp + 1)

    plt.subplot(235)
    plt.title("Filtered Spectrum")
    plt.imshow(img_dft_lp_db, cmap="gray")

    # Inverse
    img_inv_shift = np.fft.ifftshift(img_dft_shift_lp)
    img_back = cv.idft(img_inv_shift)
    img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    img_back = cv.normalize(img_back, None, 0, 255, cv.NORM_MINMAX)

    plt.subplot(236)
    plt.title("Low-pass Result")
    plt.imshow(img_back, cmap="gray")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo_7.png"
    img = read_image(image_path)

    fourier_transform(img)
