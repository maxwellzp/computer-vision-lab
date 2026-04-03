import cv2 as cv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def write_image(input_image_path: Path, output_path: Path):
    """Write an image using OpenCV."""
    img = cv.imread(str(input_image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {input_image_path}")

    success = cv.imwrite(str(output_image_path), img)
    if not success:
        raise RuntimeError(f"Failed to write image: {output_image_path}")

    print(f"Image saved to {output_path}")


if __name__ == "__main__":
    input_image_path = BASE_DIR / "data" / "images" / "photo.png"
    output_image_path = BASE_DIR / "data" / "images" / "new_photo.png"

    write_image(input_image_path, output_image_path)
