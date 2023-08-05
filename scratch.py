import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import imageio
from skimage import transform, io
import core


### Apply stuff for visualizing
def _get_corrected_centroid(im):
    rows = np.any(im, axis=1)
    cols = np.any(im, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return (rmin + rmax) // 2, (cmin + cmax) // 2

def _get_cropping_slice(centroid, target_shape, current_shape):
    """Returns a slice object that can be used to crop an image"""
    rstart, rend = centroid[0] - target_shape[0] // 2, centroid[0] + target_shape[0] // 2 + 1
    if rstart < 0:
        print('rstart < 0')
        r_slice = slice(0, target_shape[0])
    elif rend > current_shape[0]:
        print('rend > current_shape[0]')
        r_slice = slice(current_shape[0] - target_shape[0], current_shape[0])
    else:
        print('else')
        r_slice = slice(rstart, rend)

    cstart, cend = centroid[1] - target_shape[1] // 2, centroid[1] + target_shape[1] // 2 + 1
    if cstart < 0:
        print('cstart < 0')
        c_slice = slice(0, target_shape[1])
    elif cend > current_shape[1]:
        print('cend > current_shape[1]')
        c_slice = slice(current_shape[1] - target_shape[1], current_shape[1])
    else:
        print('else')
        c_slice = slice(cstart, cend)
    return r_slice, c_slice

rc, cc = (69, 431)
target = (501, 501)
current = (540, 630)
im = np.zeros(current)
rslc, cslc = _get_cropping_slice((rc, cc), target, current)

print(im.shape, im[rslc, cslc].shape)