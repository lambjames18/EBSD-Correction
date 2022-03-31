import os
import sys

import h5py
import numpy as np
from skimage import transform, io
import matplotlib

path = "D:/Research/CoNi_16/Data/3D/CoNi16_aligned.dream3d"
h5 = h5py.File(path, "r")
dataset = h5["DataContainers/ImageDataContainer/CellData/CI"][:, :, :, 0]

full_res_path = "D:/Research/CoNi_16/Data/3D/BSE/aligned/"
small_res_path = "D:/Research/CoNi_16/Data/3D/BSE/small/"

bse_slice_x = slice(476, 2696)
bse_slice_y = slice(175, 1840)
paths = sorted(
    [path for path in os.listdir(full_res_path) if ".tif" in path],
    key=lambda x: int(x.replace("aligned_", "").replace(".tif", "")),
)
bse_imgs_raw = np.array([io.imread(full_res_path + path, as_gray=True) for path in paths])
# Correct dtype and aspect ratio
bse_slice_x = slice(476, 2696)
bse_slice_y = slice(175, 1840)
bse_imgs_raw = np.float32(bse_imgs_raw)
bse_imgs = bse_imgs_raw[:, bse_slice_y, bse_slice_x]
# Transform to correct size
bse_stack = []
for i in range(bse_imgs.shape[0]):
    out = transform.resize(bse_imgs[i], dataset[0].shape, anti_aliasing=False)
    bse_stack.append(out)
bse_stack = np.array(bse_stack)[::-1]

print(bse_stack.shape)
for i, img in enumerate(bse_stack):
    matplotlib.image.imsave(
        os.path.join(small_res_path, "{}.tiff".format(i)), img, cmap="gray", dpi=1
    )
