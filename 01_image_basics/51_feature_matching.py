import numpy as np
import cv2 as cv
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent

# Feature Matching


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from filesystem in BGR using OpenCV."""
    img = cv.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    return img


def bf_matching(img1: np.ndarray, img2: np.ndarray):
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    orb = cv.ORB_create(nfeatures=1000)

    keypoints1, descriptor1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptor2 = orb.detectAndCompute(gray2, None)

    if descriptor1 is None or descriptor2 is None:
        print("Descriptors not found")
        return

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    matches = bf.match(descriptor1, descriptor2)

    matches = sorted(matches, key=lambda x: x.distance)

    n_matches = 20

    result = cv.drawMatches(
        img1,
        keypoints1,
        img2,
        keypoints2,
        matches[:n_matches],
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    rgb = cv.cvtColor(result, cv.COLOR_BGR2RGB)

    print("Keypoints1:", len(keypoints1))
    print("Keypoints2:", len(keypoints2))
    print("Matches:", len(matches))

    plt.figure(figsize=(12, 6))
    plt.title("Feature Matching (ORB)")
    plt.imshow(rgb)
    plt.show()


def knn_matching(img1: np.ndarray, img2: np.ndarray):
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT.create()

    keypoints1, descriptor1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptor2 = sift.detectAndCompute(gray2, None)

    if descriptor1 is None or descriptor2 is None:
        print("Descriptors not found")
        return

    bf = cv.BFMatcher(cv.NORM_HAMMING)

    matches = bf.knnMatch(descriptor1, descriptor2, k=2)

    good = []

    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

    print("Good matches:", len(good))

    result = cv.drawMatchesKnn(
        img1,
        keypoints1,
        img2,
        keypoints2,
        good,
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    rgb = cv.cvtColor(result, cv.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))
    plt.title("Feature Matching (KNN)")
    plt.imshow(rgb)
    plt.show()


def flann_method(img1: np.ndarray, img2: np.ndarray):
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT.create()

    keypoints1, descriptor1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptor2 = sift.detectAndCompute(gray2, None)

    if descriptor1 is None or descriptor2 is None:
        print("Descriptors not found")
        return

    FLANN_INDEX_KDTREE = 1
    n_kd_trees = 5
    n_leaf_checks = 50
    n_neighbors = 2
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=n_kd_trees)
    search_params = dict(checks=n_leaf_checks)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptor1, descriptor2, k=n_neighbors)
    matches_mask = [[0, 0] for i in range(len(matches))]
    test_ration = 0.75
    for i, (m, n) in enumerate(matches):
        if m.distance < test_ration * n.distance:
            matches_mask[i] = [1, 0]

    draw_params = dict(
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        matchesMask=matches_mask,
        flags=cv.DrawMatchesFlags_DEFAULT,
    )

    result = cv.drawMatchesKnn(
        img1,
        keypoints1,
        img2,
        keypoints2,
        matches,
        None,
        **draw_params,
    )

    rgb = cv.cvtColor(result, cv.COLOR_BGR2RGB)

    print("Keypoints1:", len(keypoints1))
    print("Keypoints2:", len(keypoints2))
    print("Matches:", len(matches))

    plt.figure(figsize=(12, 6))
    plt.title("Feature Matching (FLANN)")
    plt.imshow(rgb)
    plt.show()


if __name__ == "__main__":
    image_path = BASE_DIR / "data" / "images" / "photo_23.png"
    img1 = read_image(image_path)

    image_path = BASE_DIR / "data" / "images" / "photo_22.png"
    img2 = read_image(image_path)

    # bf_matching(img1, img2)
    # knn_matching(img1, img2)
    flann_method(img1, img2)
