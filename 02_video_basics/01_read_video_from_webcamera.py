import cv2 as cv


def read_video_from_webcam(camera_index: int):
    """Read frames from webcam and display them."""
    cap = cv.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam: {camera_index}")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to read a frame.")
            break

        cv.imshow("Webcam frame", frame)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    read_video_from_webcam(0)

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
