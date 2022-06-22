'''
Uses SIFT algorithm to find keypoint between two images
Code for homography matrix comes from:
http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
For explanation of homography matrix (M) functionality, look at:
https://upload.wikimedia.org/wikipedia/commons/2/2c/2D_affine_transformation_matrix.svg
'''

import cv2
from matplotlib import pyplot as plt
import numpy as np


def get_transformation_matrix(reference, misaligned):
    # min number keypoint between images
    MIN_MATCH_COUNT = 5
    sift = cv2.xfeatures2d.SIFT_create()

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
        raise RuntimeError("Not enough matches found - %d/%d" % (len(good), MIN_MATCH_COUNT))


def shift_image(img, shift, xy='x'):
    # shift image
    if xy == 'x':
        img_shifted = np.roll(img, -shift, axis=0)
    elif xy == 'y':
        img_shifted = np.roll(img, -shift, axis=1)
    else:
        print('xy must be x or y')
        return -1
    return img_shifted
