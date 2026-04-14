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


def detect_shapes(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    draw = img.copy()

    for cnt in contours:
        if cv.contourArea(cnt) < 500:
            continue

        epsilon = 0.02 * cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, epsilon, True)

        area = cv.contourArea(cnt)
        perimeter = cv.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        cv.drawContours(draw, [approx], -1, (0, 255, 0), 2)

        corners = len(approx)
        print(f"Corners: {corners} | Circularity: {circularity}")
        # Star
        # Corners: 10 | Circularity: 0.21919877609441663

        # Corcle
        # Corners: 8  | Circularity: 0.8982907645026967

        # Triangle
        # Corners: 3  | Circularity: 0.5497045773633431

        # Rectangle
        # Corners: 4  | Circularity: 0.7634192058742236

        x, y, _, _ = cv.boundingRect(cnt)

        if corners == 3:
            shape = "Triangle"
        elif corners == 4:
            shape = "Rectangle"
        elif circularity > 0.8:
            shape = "Circle"
        else:
            shape = "Polygon"

        cv.putText(draw, shape, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    rgb_image = cv.cvtColor(draw, cv.COLOR_BGR2RGB)
    plt.title("Shape detection app")
    plt.imshow(rgb_image)
    plt.show()


if __name__ == "__main__":
    # a circle
    image_path = BASE_DIR / "data" / "images" / "photo_12.png"
    # a star
    # image_path = BASE_DIR / "data" / "images" / "photo_13.png"
    # a triangle
    # image_path = BASE_DIR / "data" / "images" / "photo_14.png"
    # a rectangle
    # image_path = BASE_DIR / "data" / "images" / "photo_15.png"
    img = read_image(image_path)

    detect_shapes(img)
