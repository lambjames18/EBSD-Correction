import os
import sys

sys.path.insert(0, "D://Research//paraview_analysis//")

import numpy as np
import h5py
from skimage import io, filters, morphology
import matplotlib.pyplot as plt

# import support as sp


def view(ar, t):
    plt.imshow(ar)
    plt.title(t)
    plt.show()
    plt.close()


def create_ped_mask(bse):
    if type(bse) == str:
        im = io.imread(bse)
    else:
        im = bse
    if len(im.shape) > 2:
        im = im[:, :, 0]
    mask = np.where(im > 0.9 * np.mean(im), 1, 0).astype(bool)
    # Remove islands of 1's outside the pedestal
    mask = morphology.remove_small_objects(mask, min_size=200000)
    # Remove noisy holes wihtin pedestal
    mask = np.where(mask == True, False, True)
    mask = morphology.remove_small_objects(mask, min_size=9)
    mask = np.where(mask == True, False, True)
    # Created filled mask
    filled = morphology.binary_closing(mask, selem=np.ones((10, 10)))
    filled = np.where(mask == True, False, True)
    filled = morphology.remove_small_objects(filled, min_size=100000)
    filled = np.where(filled == True, False, True)
    return (mask, filled)


"""
im = "coni16_459.tif"
im = io.imread(im, as_gray=True)
im = np.where(im > np.mean(im), 1, 0)

mask = morphology.remove_small_objects(im.astype(bool), min_size=4000)

mask_filled = np.where(mask == True, False, True)
mask_filled = morphology.remove_small_objects(mask_filled, min_size=10000)
mask_filled = np.where(mask_filled == False, True, False)

selem = np.ones((5,5))
final = morphology.opening(mask_filled, selem)
final = morphology.closing(final, selem)

fig = plt.figure(figsize=(11,9))
ax1 = fig.add_subplot(221)
ax1.imshow(im)
ax2 = fig.add_subplot(222)
ax2.imshow(mask)
ax3 = fig.add_subplot(223)
ax3.imshow(mask_filled)
ax4 = fig.add_subplot(224)
ax4.imshow(final)
plt.tight_layout()
plt.show()
"""
