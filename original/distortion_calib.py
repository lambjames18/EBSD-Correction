# -*- coding: utf-8 -*-


"""
Created on Thu Jan 24 14:51:36 2019

@author: Arsenic
"""
import sys

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from skimage import transform as tf
from skimage.io import imread
import toolsDistortions as tools
import matplotlib.pyplot as plt


referencePoints = "D:/Research/scripts/Alignment/Slice420_CoNi16/ctr_pts_bse.txt"
distortedPoints = "D:/Research/scripts/Alignment/Slice420_CoNi16/ctr_pts_ebsd.txt"

# parameters of the polynomial function
deg = 3

# Read 3d feats map map
featmap = imread(f"D:/Research/scripts/Alignment/Slice420_CoNi16/ebsd.tif")

# Read control points from files
coord_ebsd = np.loadtxt(open(distortedPoints, "rb"), delimiter=" ").astype(int)
coord_bse = np.loadtxt(open(referencePoints, "rb"), delimiter=" ").astype(int)
nb_of_points = coord_ebsd.shape[0]
print(f"Given {nb_of_points} points to apply distortion correction.")

dist_ebsd = tools.successiveDistances(coord_ebsd)
dist_dic = tools.successiveDistances(coord_bse)
ratio = tools.findRatio(dist_dic, dist_ebsd)

# Convert control points coordinates into same reference frame
coord_bse_rescaled = coord_bse / ratio
src = coord_bse_rescaled  # targets
dst = coord_ebsd  # sources

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
print(model_i)

# Solve the regression system
model_i.fit(src, dst[:, 0])
model_j.fit(src, dst[:, 1])

# Define the image transformation
params = np.stack(
    [model_i.named_steps["linear"].coef_, model_j.named_steps["linear"].coef_], axis=0
)
np.save("paramsDistortion", params)
# Distort image
transform = tf._geometric.PolynomialTransform(params)
featmap_align = tf.warp(
    featmap,
    transform,
    cval=0,  # new pixels are black pixel
    order=0,  # k-neighbour
    preserve_range=True,
)

plt.imshow(featmap_align)
plt.savefig("test.tif")
