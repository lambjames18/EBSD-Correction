# -*- coding: utf-8 -*-


"""
Created on Thu Jan 24 14:51:36 2019

@author: Arsenic
"""
import sys
import os

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from skimage import transform as tf
from skimage.io import imread
import distortion_tools_3d as tools
import matplotlib.pyplot as plt


folder = "/Users/jameslamb/Code/Python/Research/EBSD-Correction/CoNi16_3D/"

# parameters of the polynomial function
deg = 3

# Read 3d feats map map
# featmap = imread(f"D:/Research/scripts/Alignment/Slice420_CoNi16/ebsd.tif")

# Read control points from files
coord_ebsd_paths = sorted([x for x in os.listdir(folder) if "ebsd" in x])
coord_bse_paths = sorted([x for x in os.listdir(folder) if "bse" in x])

coord_ebsd = np.array([[0,0,0]])
coord_bse = np.array([0,0,0])
for i in range(len(coord_bse_paths)):
    path_bse = coord_bse_paths[i]
    path_ebsd = coord_ebsd_paths[i]
    slice_num = int(coord_bse_paths[i].split("_")[-2])
    bse_dat = np.loadtxt(open(os.path.join(folder, path_bse), "rb"), delimiter=" ").astype(int)
    ebsd_dat = np.loadtxt(open(os.path.join(folder, path_ebsd), "rb"), delimiter=" ").astype(int)
    z_val = np.ones(bse_dat.shape[0], dtype=int).reshape(-1, 1) * slice_num
    bse_dat = np.hstack((bse_dat, z_val))
    ebsd_dat = np.hstack((ebsd_dat, z_val))
    coord_ebsd = np.vstack((coord_ebsd, ebsd_dat))
    coord_bse = np.vstack((coord_bse, bse_dat))

coord_bse = coord_bse[1:]
coord_ebsd = coord_ebsd[1:]

nb_of_points = coord_ebsd.shape[0]
print(f"Given {nb_of_points} points to apply distortion correction.")

dist_ebsd = tools.successiveDistances3D(coord_ebsd)
dist_dic = tools.successiveDistances3D(coord_bse)
ratio = tools.findRatio(dist_dic, dist_ebsd)

# Convert control points coordinates into same reference frame
coord_bse_rescaled = coord_bse / ratio
src = coord_bse_rescaled  # targets
dst = coord_ebsd  # sources

# Define the polynomial regression
model_i = Pipeline(
    [
        ("poly", PolynomialFeatures(degree=deg, include_bias=True)),
        ("linear", LinearRegression(fit_intercept=False)),
    ]
)

model_j = Pipeline(
    [
        ("poly", PolynomialFeatures(degree=deg, include_bias=True)),
        ("linear", LinearRegression(fit_intercept=False)),
    ]
)

model_k = Pipeline(
    [
        ("poly", PolynomialFeatures(degree=deg, include_bias=True)),
        ("linear", LinearRegression(fit_intercept=False)),
    ]
)

# Solve the regression system
model_i.fit(src, dst[:, 0])
model_j.fit(src, dst[:, 1])
model_k.fit(src, dst[:, 2])

# Define the image transformation
params = np.stack([model_i.named_steps["linear"].coef_, model_j.named_steps["linear"].coef_, model_j.named_steps["linear"].coef_], axis=0)
np.save("paramsDistortion_3D", params)
exit()
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
