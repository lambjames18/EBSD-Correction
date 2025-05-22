import numpy as np
import h5py
from skimage import io
import os
import matplotlib.pyplot as plt
import InteractiveView


def _check_sizes(im1, im2):
    """im1: ebsd (distorted), im2: ebsd (corrected)"""
    if im1.shape[1] > im2.shape[1]:
        im2_temp = np.zeros((im2.shape[0], im1.shape[1], im2.shape[2]))
        im2_temp[:, :im2.shape[1], :] = im2
        im2 = im2_temp
    if im1.shape[2] > im2.shape[2]:
        im2_temp = np.zeros((im2.shape[0], im2.shape[1], im1.shape[2]))
        im2_temp[:, :, :im2.shape[2]] = im2
        im2 = im2_temp
    return im2

def put_BSE_in_h5(h5_path, BSE_path, BSE_name="", BSE_ext='.tif', h5_name='BSE', h5_string="DataContainers/ImageDataContainer/CellData/", correct_phases=True):
    bse_paths = [os.path.join(BSE_path, f) for f in os.listdir(BSE_path) if f.endswith(BSE_ext)]
    bse_paths = sorted(bse_paths, key=lambda x: int(x.replace(BSE_path, '').replace(BSE_ext, '').replace(BSE_name, '')))
    bse_images = [io.imread(p, as_gray=True) for p in bse_paths]
    bse_images = np.array(bse_images)
    # Correct shape
    h5 = h5py.File(h5_path, 'r+')
    bse_d3d = h5[h5_string + "Confidence Index"][..., 0]
    bse_images = _check_sizes(bse_d3d, bse_images)
    bse_images = bse_images[:, 20:504, 0:654]
    bse_images = bse_images.reshape(bse_images.shape + (1,))
    size_diff = np.array(bse_images.shape) - np.array(h5[h5_string + h5_name].shape)
    if size_diff[1] > 0:
        start = size_diff[1] // 2
        end = -(size_diff[1] - start)
        bse_images = bse_images[:, start: end, :]
    elif size_diff[1] == 0:
        print("Dimensions are the same")
    else:
        raise RuntimeError("Something went wrong while aligning the dataset." + f"{size_diff[1]=}")
    if size_diff[2] > 0:
        start = size_diff[2] // 2
        end = -(size_diff[2] - start)
        bse_images = bse_images[:, :, start: end]
    elif size_diff[2] == 0:
        print("Dimensions are the same")
    else:
        raise RuntimeError("Something went wrong while aligning the dataset." + f"{size_diff[2]=}")
    print("Shapes:", bse_images.shape, h5[h5_string + h5_name].shape)
    # convert to uint16
    bse_images = bse_images - np.min(bse_images)
    bse_images = np.around(bse_images / bse_images.max() * 65535, 0).astype(np.uint16)
    print("Dtypes:", bse_images.dtype, h5[h5_string + h5_name].dtype)
    h5[h5_string + h5_name][...] = bse_images
    if correct_phases:
        phases = h5['DataContainers/ImageDataContainer/CellData/Phases'][...]
        phases = np.ones_like(phases)
        h5["DataContainers/ImageDataContainer/CellData/Phases"][...] = phases
    h5.close()
    print('BSE images added to h5 file')


if __name__ == '__main__':
    h5_path = "D:/Research/CoNi_16/Data/3D/new/CoNi16.dream3d"
    BSE_path = "D:/Research/CoNi_16/Data/3D/BSE/Selected/"
    BSE_ext = ".png"
    put_BSE_in_h5(h5_path, BSE_path, BSE_ext=BSE_ext)
