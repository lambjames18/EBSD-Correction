"""
Uses SIFT algorithm to find keypoint between two images
Code for homography matrix comes from:
http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
For explanation of homography matrix (M) functionality, look at:
https://upload.wikimedia.org/wikipedia/commons/2/2c/2D_affine_transformation_matrix.svg
"""

import cv2
from matplotlib import pyplot as plt
import numpy as np


def get_transformation_matrix(reference, misaligned):
    # min number keypoint between images
    MIN_MATCH_COUNT = 5
    # sift = cv2.xfeatures2d.SIFT_create()
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(reference, None)
    kp2, des2 = sift.detectAndCompute(misaligned, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)

    # print(len(good))
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # h,w,d = reference.shape
        # pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # dst = cv2.perspectiveTransform(pts, M)
        # img3 = cv2.polylines(misaligned, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        return M

    else:
        raise RuntimeError(
            "Not enough matches found - %d/%d" % (len(good), MIN_MATCH_COUNT)
        )


def shift_image(img, shift, xy="x"):
    # shift image
    if xy == "x":
        img_shifted = np.roll(img, -shift, axis=0)
    elif xy == "y":
        img_shifted = np.roll(img, -shift, axis=1)
    else:
        print("xy must be x or y")
        return -1
    return img_shifted


if __name__ == "__main__":
    import os
    from skimage import io
    from skimage import transform, registration
    from tqdm import tqdm

    folder = "/Users/jameslamb/Documents/research/data/CoNi90-thin/IPF/"
    paths = sorted([f for f in os.listdir(folder) if f.endswith(".png")])
    imgs = np.array([io.imread(os.path.join(folder, path)) for path in paths])
    print(imgs.shape)

    fig, ax = plt.subplots(2, 2, figsize=(4, 6), sharex=True, sharey=True)
    ax[0, 0].imshow(imgs[:, :, 600])
    ax[0, 1].imshow(imgs[:, 600])

    all_shifts = []
    idx0 = np.arange(imgs.shape[0])[:-1][::-1]
    idx1 = np.arange(imgs.shape[0])[1:][::-1]
    print(idx0.shape, idx1.shape)

    # for i in tqdm(range(50)):
    for i in tqdm(range(idx0.shape[0])):
        ref = imgs[idx1[i]]
        misaligned = imgs[idx0[i]]
        shifts, error, phasediff = registration.phase_cross_correlation(
            ref, misaligned, normalization=None
        )
        all_shifts.append((idx0[i], idx1[i], shifts[1], shifts[0]))
        shifts = np.array((shifts[0], shifts[1], 0))
        imgs[idx0[i]] = np.roll(misaligned, shifts.astype(int), axis=(0, 1, 2))

    all_shifts = np.array(all_shifts, dtype=int)
    cum_shifts = np.cumsum(all_shifts[:, 2:4], axis=0)
    all_shifts = np.hstack((all_shifts, cum_shifts))
    np.savetxt(
        "/Users/jameslamb/Documents/research/data/CoNi90-thin/manual_shifts.txt",
        all_shifts,
        delimiter="\t",
        fmt="%d",
    )

    ax[1, 0].imshow(imgs[:, :, 600])
    ax[1, 1].imshow(imgs[:, 600])

    ratio = 1.61
    xleft, xright = ax[0, 0].get_xlim()
    ybottom, ytop = ax[0, 0].get_ylim()
    ax[0, 0].set_aspect(abs((xright - xleft) / (ybottom - ytop)) * ratio)
    ax[0, 1].set_aspect(abs((xright - xleft) / (ybottom - ytop)) * ratio)
    ax[1, 0].set_aspect(abs((xright - xleft) / (ybottom - ytop)) * ratio)
    ax[1, 1].set_aspect(abs((xright - xleft) / (ybottom - ytop)) * ratio)
    plt.tight_layout()
    plt.show()
