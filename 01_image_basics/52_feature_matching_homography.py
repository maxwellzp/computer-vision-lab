import numpy as np
import cv2 as cv
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from filesystem in BGR using OpenCV."""
    img = cv.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    return img


def feature_matching_homography(img1: np.ndarray, img2: np.ndarray):
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT.create()

    keypoints1, descriptor1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptor2 = sift.detectAndCompute(gray2, None)

    if descriptor1 is None or descriptor2 is None:
        print("Descriptors not found")
        return

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(descriptor1, descriptor2, k=2)

    good_matches = []

    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    min_good_matches = 20

    if len(good_matches) > min_good_matches:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )

        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )

        error_threshold = 5
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, error_threshold)
        if M is None:
            print("Homography failed")
            return

        inlier_mask = mask.ravel().tolist()

        h, w = img1.shape[:2]
        corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(
            -1, 1, 2
        )

        projected = cv.perspectiveTransform(corners, M)

        img2_draw = img2.copy()
        cv.polylines(img2_draw, [np.int32(projected)], True, (0, 255, 0), 3)

    else:
        print("Not enough matches")
        return

    result = cv.drawMatches(
        img1,
        keypoints1,
        img2_draw,
        keypoints2,
        good_matches,
        None,
        matchesMask=inlier_mask,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    rgb = cv.cvtColor(result, cv.COLOR_BGR2RGB)

    print("Keypoints1:", len(keypoints1))
    print("Keypoints2:", len(keypoints2))
    print("Matches:", len(matches))

    plt.figure(figsize=(12, 6))
    plt.title("Feature Matching Homography")
    plt.imshow(rgb)
    plt.show()


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo_23.png"
    img1 = read_image(image_path)

    image_path = BASE_DIR / "data" / "images" / "photo_22.png"
    img2 = read_image(image_path)

    feature_matching_homography(img1, img2)
