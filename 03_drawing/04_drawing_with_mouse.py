import cv2 as cv
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent


class DrawingApp:
    def __init__(self, image_path: Path):
        self.image_path = image_path

        self.img = self.read_image(self.image_path)

        self.start_x = 0
        self.start_y = 0

        self.is_drawing = False

    def draw_line(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.is_drawing = True

            self.start_x = x
            self.start_y = y

        elif event == cv.EVENT_MOUSEMOVE:
            if self.is_drawing:
                cv.line(
                    self.img, (self.start_x, self.start_y), (x, y), (255, 255, 255), 3
                )

                self.start_x = x
                self.start_y = y
        elif event == cv.EVENT_LBUTTONUP:
            if self.is_drawing:
                cv.line(
                    self.img, (self.start_x, self.start_y), (x, y), (255, 255, 255), 3
                )
            self.is_drawing = False

    def read_image(self, image_path: Path) -> np.ndarray:
        """Read an image from filesystem in BGR using OpenCV."""
        img = cv.imread(str(self.image_path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        return img

    def run(self):
        """Setup mouse callback and run interactive drawing application."""
        window_name = "paint"

        cv.namedWindow(window_name)
        cv.setMouseCallback(window_name, self.draw_line)

        while True:
            cv.imshow(window_name, self.img)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

        cv.destroyAllWindows()


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo.png"
    app = DrawingApp(image_path)
    app.run()
