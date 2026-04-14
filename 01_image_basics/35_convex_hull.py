from pathlib import Path
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

# Image -> Gray -> Otsu -> Contours -> Hull -> RGB image


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from filesystem in BGR using OpenCV."""
    img = cv.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    return img


def convex_hull_image(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    draw = img.copy()

    for cnt in contours:
        if cv.contourArea(cnt) < 500:
            continue

        hull = cv.convexHull(cnt)

        area_cnt = cv.contourArea(cnt)
        area_hull = cv.contourArea(hull)

        if area_hull == 0:
            continue

        solidity = area_cnt / area_hull

        # BGR -> Green
        cv.drawContours(draw, [cnt], -1, (0, 255, 0), 2)

        # BGR -> Blue
        cv.drawContours(draw, [hull], -1, (255, 0, 0), 2)

        x, y, _, _ = cv.boundingRect(cnt)
        cv.putText(
            draw,
            f"{solidity:.2f}",
            (x, y - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            2,
        )

        # print(solidity)
        # a star = 0.4425141274485716
        # a circle = 0.996609489782898

    rgb_image = cv.cvtColor(draw, cv.COLOR_BGR2RGB)
    plt.imshow(rgb_image)
    plt.title("Contours and Convex Hull")
    plt.show()


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo_13.png"
    img = read_image(image_path)

    convex_hull_image(img)
