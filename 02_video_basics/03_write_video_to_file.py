import cv2 as cv
from pathlib import Path
import time

BASE_DIR = Path(__file__).resolve().parent.parent

WIDTH = 1280
HEIGHT = 720
FPS = 30


def write_video_from_camera_to_file(camera_index: int):
    """Read frames from webcam and display them."""
    cap = cv.VideoCapture(camera_index)
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter.fourcc(*"MJPG"))
    cap.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv.CAP_PROP_FPS, FPS)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam: {camera_index}")

    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    print("Camera FPS:", fps)

    output_file_path = BASE_DIR / "data" / "videos" / "webcam-video.mp4"

    writer = cv.VideoWriter(
    str(output_file_path),
    cv.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height))

    frame_duration = 1.0 / fps
    while True:
        start = time.time()
        ret, frame = cap.read()

        if not ret:
            print("Failed to read a frame.")
            break

        writer.write(frame)
        cv.imshow("Webcam frame", frame)
        elapsed = time.time() - start
        sleep_time = frame_duration - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    write_video_from_camera_to_file(0)

# maksim@maksim-laptop:~$ ls /dev/video*
# /dev/video0  /dev/video1
# maksim@maksim-laptop:~$

# maksim@maksim-laptop:~$ lsusb
# Bus 001 Device 001: ID 1d6b:0002 Linux Foundation 2.0 root hub
# Bus 001 Device 002: ID 0db0:0d11 Micro Star International MSI GM11 Gaming Mouse
# Bus 001 Device 003: ID 5986:212b Bison Electronics Inc. Integrated Camera
# Bus 001 Device 004: ID 0bda:c123 Realtek Semiconductor Corp. Bluetooth Radio
# Bus 002 Device 001: ID 1d6b:0003 Linux Foundation 3.0 root hub
# Bus 003 Device 001: ID 1d6b:0002 Linux Foundation 2.0 root hub
# Bus 003 Device 002: ID 048d:c966 Integrated Technology Express, Inc. ITE Device(8176)
# Bus 004 Device 001: ID 1d6b:0003 Linux Foundation 3.0 root hub
# maksim@maksim-laptop:~$

# maksim@maksim-laptop:~$ sudo apt install v4l-utils
# maksim@maksim-laptop:~$ v4l2-ctl --list-devices
# Integrated Camera: Integrated C (usb-0000:05:00.3-3):
# 	/dev/video0
# 	/dev/video1
# 	/dev/media0

# maksim@maksim-laptop:~$

# maksim@maksim-laptop:~$ v4l2-ctl --list-formats-ext -d /dev/video0
# ioctl: VIDIOC_ENUM_FMT
# 	Type: Video Capture

# 	[0]: 'MJPG' (Motion-JPEG, compressed)
# 		Size: Discrete 1280x720
# 			Interval: Discrete 0.033s (30.000 fps)
# 		Size: Discrete 320x180
# 			Interval: Discrete 0.033s (30.000 fps)
# 		Size: Discrete 320x240
# 			Interval: Discrete 0.033s (30.000 fps)
# 		Size: Discrete 352x288
# 			Interval: Discrete 0.033s (30.000 fps)
# 		Size: Discrete 424x240
# 			Interval: Discrete 0.033s (30.000 fps)
# 		Size: Discrete 640x360
# 			Interval: Discrete 0.033s (30.000 fps)
# 		Size: Discrete 640x480
# 			Interval: Discrete 0.033s (30.000 fps)
# 		Size: Discrete 848x480
# 			Interval: Discrete 0.033s (30.000 fps)
# 		Size: Discrete 960x540
# 			Interval: Discrete 0.033s (30.000 fps)
# 	[1]: 'YUYV' (YUYV 4:2:2)
# 		Size: Discrete 1280x720
# 			Interval: Discrete 0.100s (10.000 fps)
# 		Size: Discrete 320x180
# 			Interval: Discrete 0.033s (30.000 fps)
# 		Size: Discrete 320x240
# 			Interval: Discrete 0.033s (30.000 fps)
# 		Size: Discrete 352x288
# 			Interval: Discrete 0.033s (30.000 fps)
# 		Size: Discrete 424x240
# 			Interval: Discrete 0.033s (30.000 fps)
# 		Size: Discrete 640x360
# 			Interval: Discrete 0.033s (30.000 fps)
# 		Size: Discrete 640x480
# 			Interval: Discrete 0.033s (30.000 fps)
# 		Size: Discrete 848x480
# 			Interval: Discrete 0.050s (20.000 fps)
# 		Size: Discrete 960x540
# 			Interval: Discrete 0.067s (15.000 fps)
# maksim@maksim-laptop:~$
