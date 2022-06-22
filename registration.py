import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from skimage import io, morphology

import SIFT

def get_shift(reference, misaligned):
    M = SIFT.get_transformation_matrix(reference, misaligned)
    transform_mats.append(M)
    shift = (np.array([1, 1]) - np.around(M.dot(np.array([1, 1, 1]).T).T[:2])).astype(int)
    return shift

# Get bse data
bse_folder = "D:/Research/CoNi_16/Data/3D/BSE/small/"
paths = sorted([path for path in os.listdir(bse_folder) if ".tiff" in path], key=lambda x: int(x.replace(".tiff", "")))
bse_imgs_small = np.array([io.imread(bse_folder + path, as_gray=True) for path in paths])
# Created filled mask
mask = np.where(bse_imgs_small > bse_imgs_small.mean() * 0.85, True, False)
mask_filled = np.zeros(mask.shape)
for i in range(mask.shape[0]):
    im = morphology.remove_small_holes(mask[i], area_threshold=10000)
    im = morphology.remove_small_objects(im, min_size=200000)
    mask_filled[i] = im
mask_filled = mask_filled

# Get ebsd data
h5 = h5py.File("D:/Research/CoNi_16/Data/3D/CoNi16_basic.dream3d", 'r')
s = "DataContainers/ImageDataContainer/CellData/"
ebsd_slice_y = slice(None, -1)
ebsd_slice_x = slice(10, None)
ebsd = h5[s + "IPFColor_001"][:, ebsd_slice_y, ebsd_slice_y]  # EBSD crop taken care of in h5 already 
print(ebsd.shape, ebsd.dtype)
if ebsd.dtype != np.uint8:
    ebsd = np.around(255 * ebsd/ebsd.max(), 0).astype(np.uint8)

# Get SIFT keypoints and shift mask for all slices in ebsd
transform_mats = []
shifts = []
ebsd_new = np.zeros(ebsd.shape, dtype=np.uint8)
ebsd_new[-1] = ebsd[-1]
num = ebsd.shape[0] - 1
for i in range(num):
    # 1 is y, 0 is x
    new = ebsd[i + 1]
    total = 100
    count = 0
    while total > 2 and count < 100:
        shift0 = get_shift(ebsd[i, 100:300, 100:300], new[100:300, 100:300])
        new = SIFT.shift_image(ebsd[i], shift0[0], "x")
        shift1 = get_shift(ebsd[i, 100:300, 100:300], new[100:300, 100:300])
        new = SIFT.shift_image(new, shift1[1], "y")
        shift = get_shift(ebsd[i, 100:300, 100:300], new[100:300, 100:300])
        total = sum(shift)
        count += 1
    ebsd_new[i] = new
    print(i, count)

fig, ax = plt.subplots(2, 2, figsize=(10, 5))
ax[0, 0].imshow(ebsd[:num, ebsd.shape[1] // 2, :min(int(num * 1.5), ebsd.shape[0])])
ax[0, 0].set_title("Before registration (y)")
ax[0, 1].imshow(ebsd_new[:num, ebsd_new.shape[1] // 2, :min(int(num * 1.5), ebsd.shape[0])])
ax[0, 1].set_title("After registration (y)")
ax[1, 0].imshow(ebsd[:num, :min(int(num * 1.5), ebsd.shape[0]), ebsd.shape[2] // 2])
ax[1, 0].set_title("Before registration (x)")
ax[1, 1].imshow(ebsd_new[:num, :min(int(num * 1.5), ebsd.shape[0]), ebsd_new.shape[2] // 2]) 
ax[1, 1].set_title("After registration (x)")
plt.tight_layout()
plt.show()
