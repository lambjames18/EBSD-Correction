import os
import sys

sys.path.insert(0, "D://Research//paraview_analysis//")

import numpy as np
import h5py
from skimage import io, filters
import matplotlib.pyplot as plt

# import support as sp


def pipeline(im, fig_id=0):
    im_max = np.amax(im)
    im_mean = np.mean(im)
    sobel = filters.sobel(im)
    
    fig = plt.figure(fig_id, figsize=(12,6))
    ax1 = plt.subplot(141)
    ax2 = plt.subplot(142)
    ax3 = plt.subplot(143)
    ax4 = plt.subplot(144)
    
    ax1.imshow(im)
    ax1.set_title("Original")
    
    ax2.imshow(np.where(sobel > im_max * 0.2, 1, 0))
    ax2.set_title("0.2")

    ax3.imshow(np.where(sobel > im_max * 0.1, 1, 0))
    ax3.set_title("0.1")

    ax4.imshow(np.where((im_max * 0.2 > sobel) & (sobel > im_max * 0.1), 1, 0))
    ax4.set_title("(0.1, 0.2)")
    
    plt.tight_layout()
    

bse_folder = "D:/Research/NiAlMo_APS/Data/ALL_BSE_resized/"
ebsd = h5py.File("D:/Research/NiAlMo_APS/Data/3D/NiAlMo.dream3d", 'r')
paths = os.listdir(bse_folder)
bse_paths = sorted(paths, key=lambda x: int(x[8:-4]))
bse_imgs = np.zeros((391, 781, 1101))
for i in range(len(bse_paths)):
    bse_paths[i] = os.path.join(bse_folder, bse_paths[i])
    bse_imgs[i] = io.imread(bse_paths[i])

ebsd_imgs = ebsd['DataContainers/ImageDataContainer/CellData/Confidence Index'][...]

pipeline(bse_imgs[0], 0)
pipeline(ebsd_imgs[0], 1)
plt.show()