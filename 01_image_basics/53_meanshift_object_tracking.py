import numpy as np
import cv2 as cv
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent


def meanshift_object_tracking(video_path: str):
    cap = cv.VideoCapture(str(video_path))

    ret, frame = cap.read()

    # plt.figure()
    # plt.imshow(frame)
    # plt.show()

    # return

    if not ret:
        print(f"Cannot read {video_path} video")
        return

    x, y, w, h = 287, 212, 96, 97
    track_window = (x, y, w, h)

    roi = frame[y : y + h, x : x + w]
    # plt.imshow(roi)
    # plt.show()
    # return
    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

    lower = np.array([0, 150, 50])
    upper = np.array([10, 255, 255])

    mask = cv.inRange(hsv_roi, lower, upper)

    roi_hist = cv.calcHist([hsv_roi], [0, 1], mask, [180, 256], [0, 180, 0, 256])

    cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

    term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

    while True:
        ret, video_frame = cap.read()

        if not ret:
            break

        hsv = cv.cvtColor(video_frame, cv.COLOR_BGR2HSV)

        back_proj = cv.calcBackProject([hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)

        _, track_window = cv.meanShift(back_proj, track_window, term_crit)

        x, y, w, h = track_window

        draw_frame = video_frame.copy()
        cv.rectangle(draw_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv.imshow("MeanShift Tracking", draw_frame)

        if cv.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    video_path = BASE_DIR / "data" / "videos" / "video3.mp4"

    meanshift_object_tracking(video_path)
