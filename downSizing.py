import os
import sys

sys.path.insert(0, "D://Research//paraview_analysis//")

import numpy as np
import skimage.transform as tf
import skimage.io as io
import matplotlib.pyplot as plt

import support as sp

path = "D:/Research/Alignment/"
bse_name = "385_bse.tif"
ebsd_name = "385_ebsd.tif"
bse = io.imread(os.path.join(path, bse_name))
ebsd = io.imread(os.path.join(path, ebsd_name))
bse_resized = tf.resize(bse, ebsd.shape, anti_aliasing=True)
sp.saveim(im=bse_resized, name="385_bse_resized.tif", cmap="gray")

bse_points = np.loadtxt(os.path.join(path, "ctr_pts_bse.txt")).astype(np.int32)
ebsd_points = np.loadtxt(os.path.join(path, "ctr_pts_ebsd.txt")).astype(np.int32)
# print(ebsd_points.shape)

bse_points_im = np.zeros(bse.shape)
for point in bse_points:
    slice_row = slice(point[1] - 5, point[1] + 6, None)
    slice_col = slice(point[0] - 6, point[0] + 7, None)
    bse_points_im[(slice_row, slice_col)] = 1

# bse_points_im_resized = tf.resize(bse_points_im, ebsd.shape, anti_aliasing=True)
# bse_points_resized = np.array(np.where(bse_points_im_resized == 1))
# np.savetxt("ctr_pts_bse_resized", bse_points_resized, delimiter=" ", fmt="%.2f")
bse_points_resized = np.loadtxt("ctr_pts_bse_resized.txt")
# print(bse_points_resized.shape)

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

ax1.imshow(bse)
ax1.set_title("Raw BSE")
ax1.scatter(bse_points[:, 0], bse_points[:, 1], s=50, marker="+", color="r")
ax2.imshow(bse_resized)
ax2.set_title("Resized BSE")
ax2.scatter(bse_points_resized[:, 0], bse_points_resized[:, 1], s=50, marker="+", color="r")
ax3.imshow(ebsd)
ax3.scatter(ebsd_points[:, 0], ebsd_points[:, 1], s=50, marker="+", color="r")
ax3.set_title("EBSD")

plt.tight_layout()
plt.show()
