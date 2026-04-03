import numpy as np
import cv2 as cv
import os
from pathlib import Path

# Video properties of video_4K.mp4 file:
# Frame Rate: 25 FPS

# 1 second = 1000 ms
# 25 frames = 1000 ms

# 1 frame = 1000 / 25 = 40ms:


BASE_DIR = Path(__file__).resolve().parent.parent


def read_video_from_file(video_path: Path):
    cap = cv.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv.CAP_PROP_FPS)
    if video_fps == 0:
        video_fps = 25
    print(f"FPS: {video_fps}")  # 25.0

    delay = int(1000 / video_fps)
    print(f"Delay: {delay}")  # 40

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("End of video")
            break

        cv.imshow(f"Video from {video_path.name}", frame)

        if cv.waitKey(delay) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    video_path = BASE_DIR / "data" / "videos" / "video_4K.mp4"
    read_video_from_file(video_path)
