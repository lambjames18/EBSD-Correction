# -*- coding: utf-8 -*-


"""
Created on Thu Jan 24 14:51:36 2019

@author: Arsenic
"""
import sys

import numpy as np
from PIL import Image
import skimage
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from skimage import transform as tf
from skimage.io import imread
import toolsDistortions as tools
import matplotlib.pyplot as plt

sys.path.insert(0, "../paraview_analysis/")
import support as sp


# parameters of the polynomial function
deg = 2


# Read 3d feats map map
slice_id = 170
featmap = imread(f"slice_{slice_id}_ebsd.png")
featmap = featmap.dot([0.3, 0.59, 0.11, 0])
# featmap = imread("ebsd_tiltOn.png")

# Read control points from files
coord_ebsd = np.loadtxt(open("ctr_pts_ebsd.txt", "rb"), delimiter=" ").astype(int)
coord_bse = np.loadtxt(open("ctr_pts_bse.txt", "rb"), delimiter=" ").astype(int)
nb_of_points = coord_ebsd.shape[0]
print(f"Given {nb_of_points} points to apply distortion correction.")


def swap_cols(arr, frm, to):  # swap columns to get from imageJ to Python coordinates system
    arr[:, [frm, to]] = arr[:, [to, frm]]


# swap_cols(coord_ebsd, 0, 1)
# swap_cols(coord_bse, 0, 1)

dist_ebsd = tools.successiveDistances(coord_ebsd)
dist_dic = tools.successiveDistances(coord_bse)
ratio = tools.findRatio(dist_dic, dist_ebsd)

# Convert control points coordinates into same reference frame
coord_bse_rescaled = coord_bse / ratio
src = coord_bse_rescaled  # targets
dst = coord_ebsd  # sources
# dst = coord_bse  # sources
# src = coord_ebsd  # targets


# Define the polynomial regression
model_i = Pipeline(
    [
        ("poly", PolynomialFeatures(degree=deg, include_bias=True)),
        ("linear", LinearRegression(fit_intercept=False, normalize=False)),
    ]
)

model_j = Pipeline(
    [
        ("poly", PolynomialFeatures(degree=deg, include_bias=True)),
        ("linear", LinearRegression(fit_intercept=False, normalize=False)),
    ]
)

# Solve the regression system
model_i.fit(src, dst[:, 0])
model_j.fit(src, dst[:, 1])

# Define the image transformation
params = np.stack(
    [model_i.named_steps["linear"].coef_, model_j.named_steps["linear"].coef_], axis=0
)
np.save("paramsDistortion", params)
# Distort image
buf = 1500
featmap_buffered = np.zeros(np.array(featmap.shape) + buf)
featmap_buffered[int(buf / 2) : -int(buf / 2), int(buf / 2) : -int(buf / 2)] = featmap
transform = tf._geometric.PolynomialTransform(params)
featmap_align = tf.warp(
    featmap_buffered,
    transform,
    cval=0,  # new pixels are black pixel
    order=0,  # k-neighbour
    preserve_range=True,
)
buf = buf //2
featmap_align = featmap_align[buf:, buf:]
sp.saveim(featmap_align, name=f"slice_{slice_id}_ebsd_aligned.tiff", cmap="gray")

# featmap_trans_im = Image.fromarray(featmap_align)
# featmap_trans_im.save("ebsd_aligned.tif")

# """
