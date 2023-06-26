import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from skimage import io, morphology

# Get ebsd data
# h5 = h5py.File("D:/Research/CoNi_16/Data/3D/CoNi16_basic.dream3d", 'r')
# h5 = h5py.File("D:/Research/R2_Sample9-SHot5/Data/3D/R2S9S5.dream3d", 'r')
h5 = h5py.File("D:/Research/NiAlMo_APS/Data/3D/NiAlMo_Basic.dream3d", 'r')
s = "DataContainers/ImageDataContainer/CellData/"
ebsd = np.squeeze(h5[s + "Image Quality"][:])  # EBSD crop taken care of in h5 already 
print(ebsd.shape, ebsd.dtype)
col = 800
row = 400

data = np.where(ebsd > 0.22, 1, 0)
data = morphology.remove_small_holes(data, area_threshold=10000)
data = morphology.binary_erosion(data, np.ones((5, 5, 5)))
data = morphology.remove_small_objects(data, min_size=500000)
data = morphology.binary_dilation(data, np.ones((5, 5, 5)))

data_new = np.zeros(data.shape, dtype=data.dtype)
for i in range(data.shape[0]):
    im = data[i]
    row_index = np.argmax(im[:, col])
    col_index = np.argmax(im[row, ::-1])
    diff_row = 50 - row_index
    diff_col = 10 - col_index
    data_new[i] = np.roll(np.roll(im, diff_row, axis=0), -diff_col, axis=1)
    # data_new[i] = np.roll(im, diff_row, axis=0)
    # data_new[i] = np.roll(im, -diff_col, axis=1)

fig, ax = plt.subplots(2, 2, figsize=(10, 5))
ax[0, 0].imshow(data[:, :, col])
ax[0, 0].set_title("Before registration (y)")
ax[0, 1].imshow(data_new[:, :, col])
ax[0, 1].set_title("After registration (y)")
ax[1, 0].imshow(data[:, row, :])
ax[1, 0].set_title("Before registration (x)")
ax[1, 1].imshow(data_new[:, row, :])
ax[1, 1].set_title("After registration (x)")
plt.tight_layout()
plt.show()
