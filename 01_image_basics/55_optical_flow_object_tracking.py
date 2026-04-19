import cv2 as cv
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Optical Flow Object Tracking


def optical_flow_tracking(video_path):
    cap = cv.VideoCapture(video_path)

    ret, old_frame = cap.read()
    if not ret:
        print("Cannot read video")
        return

    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

    p0 = cv.goodFeaturesToTrack(
        old_gray,
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7,
    )

    mask = np.zeros_like(old_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Optical Flow
        p1, status, error = cv.calcOpticalFlowPyrLK(
            old_gray,
            frame_gray,
            p0,
            None,
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        good_new = p1[status == 1]
        good_old = p0[status == 1]

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()

            mask = cv.line(mask, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

        output = cv.add(frame, mask)

        cv.imshow("Optical Flow", output)

        if cv.waitKey(30) & 0xFF == 27:
            break

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    video_path = BASE_DIR / "data" / "videos" / "video2.mp4"

    optical_flow_tracking(video_path)
