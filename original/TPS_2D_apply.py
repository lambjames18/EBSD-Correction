import os

import numpy as np
import imageio
import h5py
from rich.progress import track
from rich import print


def correct_slice(array, sol):
    xgtId = np.around(sol[0]).astype(int).flatten()  # round to nearest pixel
    ygtId = np.around(sol[1]).astype(int).flatten()  # round to nearest pixel
    validX = (xgtId < array.shape[1]) * (xgtId > 0)
    validY = (ygtId < array.shape[0]) * (ygtId > 0)
    valid = validX * validY

    c = array[validY * ygtId, validX * xgtId]
    c = c * valid

    return np.reshape(c, (ny, nx))


fname = "tfs_data.dream3d"
folder = "D:\\Research\\Co_TFS\\Data\\"
path = os.path.join(folder, fname)
h5 = h5py.File(path, "r+")
sol = np.load("pointWise_mapping.npy")
dims = h5["DataContainers/ImageDataContainer/CellData/Confidence Index"].shape

nx = dims[2]
ny = dims[1]
nz = dims[0]

for key in h5["DataContainers/ImageDataContainer/CellData"].keys():
    data = h5[f"DataContainers/ImageDataContainer/CellData/{key}"]
    track_str = f"Aligning {key}..."
    for i in track(range(nz), track_str + " " * (30 - len(track_str))):
        data_slice = data[i]
        for j in range(data.shape[-1]):
            data_slice_value = data_slice[:, :, j]
            output = correct_slice(data_slice_value, sol)
            h5[f"DataContainers/ImageDataContainer/CellData/{key}"][i, :, :, j] = output
