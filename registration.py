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

# Get ebsd data
# h5 = h5py.File("D:/Research/CoNi_16/Data/3D/CoNi16_basic.dream3d", 'r')
h5 = h5py.File("D:/Research/R2_Sample9-SHot5/Data/3D/R2S9S5.dream3d", 'r')
s = "DataContainers/ImageDataContainer/CellData/"
ebsd = h5[s + "Image Quality"][:]  # EBSD crop taken care of in h5 already 
print(ebsd.shape, ebsd.dtype)
if ebsd.dtype != np.uint8:
    ebsd = np.around(255 * ebsd / ebsd.max(), 0).astype(np.uint8)

# Get SIFT keypoints and shift mask for all slices in ebsd
transform_mats = []
shifts = []
ebsd_new = np.zeros(ebsd.shape, dtype=np.uint8)
ebsd_new[-1] = ebsd[-1]
num = ebsd.shape[0] - 1
for i in range(num):
    # 1 is y, 0 is x
    new = ebsd[i + 1]
    shift0 = get_shift(ebsd[i], new)
    new = SIFT.shift_image(ebsd[i], shift0[0], "x")
    shift1 = get_shift(ebsd[i], new)
    new = SIFT.shift_image(new, shift1[1], "y")
    shift = get_shift(ebsd[i], new)
    ebsd_new[i] = new

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
