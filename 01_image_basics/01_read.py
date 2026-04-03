import cv2 as cv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def image_read(img_path: Path):
    """Read and display an image using OpenCV."""
    img = cv.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    cv.imshow("Image", img)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo.png"
    image_read(image_path)
