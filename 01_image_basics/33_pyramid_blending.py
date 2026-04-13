from pathlib import Path
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from filesystem in BGR using OpenCV."""
    img = cv.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    return img


def pyramid_blending(img1: np.ndarray, img2: np.ndarray):
    img1 = cv.resize(img1, (512, 512))
    img2 = cv.resize(img2, (512, 512))

    # 1. Gaussian pyramids ---
    gp1 = [img1.copy()]
    gp2 = [img2.copy()]

    for i in range(6):
        img1 = cv.pyrDown(img1)
        img2 = cv.pyrDown(img2)
        gp1.append(img1)
        gp2.append(img2)

    # 2. Laplacian pyramids ---
    lp1 = [gp1[-1]]
    lp2 = [gp2[-1]]

    for i in range(len(gp1) - 1, 0, -1):
        size = (gp1[i - 1].shape[1], gp1[i - 1].shape[0])

        GE1 = cv.pyrUp(gp1[i])
        GE1 = cv.resize(GE1, size)

        L1 = cv.subtract(gp1[i - 1], GE1)
        lp1.append(L1)

        GE2 = cv.pyrUp(gp2[i])
        GE2 = cv.resize(GE2, size)

        L2 = cv.subtract(gp2[i - 1], GE2)
        lp2.append(L2)

    # 3. Combine pyramids ---
    LS = []
    for l1, l2 in zip(lp1, lp2):
        cols = l1.shape[1]
        ls = np.hstack((l1[:, : cols // 2], l2[:, cols // 2 :]))
        LS.append(ls)

    # --- 4. Reconstruction ---
    img_reconstruct = LS[0]

    for i in range(1, len(LS)):
        size = (LS[i].shape[1], LS[i].shape[0])
        img_reconstruct = cv.pyrUp(img_reconstruct)
        img_reconstruct = cv.resize(img_reconstruct, size)
        img_reconstruct = cv.add(img_reconstruct, LS[i])

    # 5. Display results
    plt.imshow(cv.cvtColor(img_reconstruct, cv.COLOR_BGR2RGB))
    plt.title("Blended Image")
    plt.show()


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo_9.png"
    img1 = read_image(image_path)

    image_path = BASE_DIR / "data" / "images" / "photo_10.png"
    img2 = read_image(image_path)

    pyramid_blending(img1, img2)
