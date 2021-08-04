import os
import sys

sys.path.insert(0, "D://Research//paraview_analysis//")

import numpy as np
from skimage import io, filters
import matplotlib.pyplot as plt

import support as sp

bse_folder = "D:/Research/NiAlMo_APS/Data/ALL_BSE_resized/"
paths = os.listdir(bse_folder)
bse_paths = sorted(paths, key=lambda x: int(x[8:-4]))
for i in range(len(paths)):
    bse_paths[i] = os.path.join(bse_folder, bse_paths[i])

im = io.imread(bse_paths[0])[:, :, 0]
sobel = filters.sobel(im)
im_max = np.amax(sobel)

plt.figure(0)
plt.imshow(sobel)
plt.title("Sobel")

plt.figure(1)
plt.imshow(np.where(sobel > im_max * 0.2, 1, 0))
plt.title("0.2")

plt.figure(2)
plt.imshow(np.where(sobel > im_max * 0.1, 1, 0))
plt.title("0.1")

plt.figure(3)
plt.imshow(np.where((im_max * 0.2 > sobel) & (sobel > im_max * 0.1), 1, 0))
plt.title("(0.1, 0.2)")

plt.figure(4)
plt.imshow(np.where(im > np.amax(im) * 0.5, 1, 0))
plt.title("raw")

plt.show()
