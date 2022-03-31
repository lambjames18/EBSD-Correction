# -*- coding: utf-8 -*-


"""
Created on Thu Jan 24 14:51:36 2019

@author: Arsenic
"""
import os
import sys

sys.path.insert(0, "D:/Research/scripts/paraview_analysis")

import h5py
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from skimage import transform as tf
import distortion_tools_3d as tools
import matplotlib.pyplot as plt

import gif


# folder = "/Users/jameslamb/Code/Python/Research/EBSD-Correction/CoNi16_3D/"
folder = "D:/Research/scripts/Alignment/CoNi16_3D/"

# Read h5 file
path = "D:/Research/CoNi_16/Data/3D/CoNi16.dream3d"
h5 = h5py.File(path, "r")
data = h5["DataContainers/ImageDataContainer/CellData/CI"][:, :, :, 0]
print(data.shape)

# parameters of the polynomial function
deg = 3

# Read control points from files
coord_ebsd_paths = sorted([x for x in os.listdir(folder) if "ebsd" in x])
coord_bse_paths = sorted([x for x in os.listdir(folder) if "bse" in x])

coord_ebsd = np.array([[0, 0, 0]])
coord_bse = np.array([0, 0, 0])
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
# model_i.fit(src[:, 0].reshape(-1, 1), dst[:, 0])
# model_j.fit(src[:, 1].reshape(-1, 1), dst[:, 1])
# model_k.fit(src[:, 2].reshape(-1, 1), dst[:, 2])

# Define the image transformation
# params = np.stack(
#     [
#         model_i.named_steps["linear"].coef_,
#         model_j.named_steps["linear"].coef_,
#         model_j.named_steps["linear"].coef_,
#     ],
#     axis=0,
# )
# np.save("paramsDistortion_3D", params)


# Create 3D transformation
t_matrix = tf._geometric._umeyama(src, dst, estimate_scale=False)

xr = np.arange(data.shape[2])
yr = np.arange(data.shape[1])
zr = np.arange(data.shape[0])
zz, yy, xx = np.meshgrid(zr, yr, xr, indexing="ij")
points = np.hstack((zz.reshape(-1, 1), yy.reshape(-1, 1), xx.reshape(-1, 1)))
new_points = np.around(t_matrix[:3, :3].dot(points.T).T, 0).astype(int)
z = new_points[:, 0].reshape(zz.shape)
y = new_points[:, 1].reshape(yy.shape)
x = new_points[:, 2].reshape(xx.shape)
transform = np.array([z, y, x])
np.save("transform_3D.npy", transform)
print(transform.shape)

# data = np.swapaxes(data, 0, 1)
# Distort image
data_align = tf.warp(
    data,
    transform,
    cval=0,  # new pixels are black pixel
    order=0,  # k-neighbour
    preserve_range=True,
)
# print(data_align.shape)
# data_align = np.swapaxes(data_align, 0, 1)
print(data_align.shape)

gif.saveMedia(imgs=data_align, name="aligned".format(i), path="./")

"""
fig = plt.figure(figsize=(12, 6))
axes = []
ims = [im0, im1]
titles = ["Original", "Aligned"]
for i in range(2):
    axes.append(fig.add_subplot(1, 2, i + 1))
    axes[i].imshow(ims[i])
    axes[i].set_title(titles[i])
plt.show()
plt.close("all")
"""
