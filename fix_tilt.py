import sys
import os

sys.path.insert(0, "../paraview_analysis/")

import numpy as np
import matplotlib.pyplot as plt
from skimage import transform as tf
from skimage.io import imread

import support as sp


def evalBoundary(image):
    rowMax = np.argmin(image[:, -1])
    colMax = np.argmin(image[0, ::-1])
    return image[:rowMax, -colMax:]


root = "D:/Research/NiAlMo_APS/Data/EMSphInx_Output/img_TiltON/"

ref = "D:/Research/NiAlMo_APS/Data/EMSphInx_Output/img/EMSphInx_30.png"
ref_img = imread(ref)

names = []
imgs = []
for i in range(31, 63):
    names.append(f"EMSphInx_{i}.png")
    imgs.append(imread(f"{root}{names[-1]}"))

for i in range(len(imgs)):
    imgs[i] = np.where(imgs[i] == 0, 1, imgs[i])
    img_resized = tf.resize(imgs[i], (imgs[i].shape[0] - 82, imgs[i].shape[1]))
    img_resized = np.around(255 * img_resized/img_resized.max()).astype(np.uint8)
    # fig, ax = plt.subplots(1, 1)
    # ax.imshow(ref_img, cmap="gray")
    # ax.imshow(np.roll(img_resized, 8, axis=0), alpha=0.5, cmap="inferno")
    # ax.set_title("Overlay")
    # plt.show()
    sp.saveim(img_resized, name=f"{root}../img/{names[i]}", cmap="gray")
