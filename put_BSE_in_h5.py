import numpy as np
import h5py
from skimage import io
import os
import matplotlib.pyplot as plt
import core

def put_BSE_in_h5(h5_path, BSE_path, BSE_name="", BSE_ext='.tif', h5_name='BSE', h5_string="DataContainers/ImageDataContainer/CellData/", correct_phases=True):
    bse_paths = [os.path.join(BSE_path, f) for f in os.listdir(BSE_path) if f.endswith(BSE_ext)]
    bse_paths = sorted(bse_paths, key=lambda x: int(x.replace(BSE_path, '').replace(BSE_ext, '').replace(BSE_name, '')))
    bse_images = [io.imread(p, as_gray=True) for p in bse_paths]
    bse_images = np.array(bse_images)
    # convert to uint16
    bse_images = bse_images - np.min(bse_images)
    bse_images = np.around(bse_images / bse_images.max() * 65535, 0).astype(np.uint16)
    h5 = h5py.File(h5_path, 'r+')
    # convert to the right size
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
    print("Dtypes:", bse_images.dtype, h5[h5_string + h5_name].dtype)
    print("Shapes:", bse_images.shape, h5[h5_string + h5_name].shape)
    h5[h5_string + h5_name][...] = bse_images
    if correct_phases:
        phases = h5['DataContainers/ImageDataContainer/CellData/Phases'][...]
        phases = np.ones_like(phases)
        h5["DataContainers/ImageDataContainer/CellData/Phases"][...] = phases
    h5.close()
    print('BSE images added to h5 file')


if __name__ == '__main__':
    h5_path = "D:/Research/Ta/Data/3D/AMSpall/TaAMS_Stripped_corrected.dream3d"
    BSE_path = "D:/Research/Ta/Data/3D/AMSpall/BSE/small/"
    BSE_ext = ".tiff"
    put_BSE_in_h5(h5_path, BSE_path, BSE_ext=BSE_ext)
