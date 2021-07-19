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


root = "./../NiAlMo_APS/Data/EMSphInx_Output/img_TiltON/"
params = np.load("paramsEBSD.npy")

transform = tf._geometric.PolynomialTransform(params)

rnge = np.arange(31, 63, 1, dtype=int)
names = []
imgs = []
for i in range(len(rnge)):
    names.append(f"EMSphInx_{rnge[i]}.png")
    imgs.append(imread(f"{root}{names[i]}"))

for i in range(len(imgs)):
    imgs[i] = np.where(imgs[i] == 0, 1, imgs[i])
    featmap_align = tf.warp(
        imgs[i],
        transform,
        cval=0,  # new pixels are black pixel
        order=0,  # k-neighbour
        preserve_range=True,
    )
    featmap_align = evalBoundary(featmap_align)
    sp.saveim(featmap_align, name=f"{root}../img/{names[i]}", cmap="gray")
