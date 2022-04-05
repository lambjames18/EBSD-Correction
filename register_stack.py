import os
import sys

import numpy as np
import h5py
import matplotlib.pyplot as plt
from skimage import io, transform
from rich.progress import track
from pystackreg import StackReg

sys.path.insert(0, "D:/Research/scripts/paraview_analysis/")
import support as sp
import gif

folder = "D:/Research/Ta_AM-Spalled/Data/3D/"


def register_vol(paths, ref="previous", bound=10, dirs="xy", crop=slice(None)):
    # Set range of shifts to check
    delta = np.arange(-bound, bound + 1, 1)
    delta = np.array(sorted(delta, key=lambda x: np.abs(x)))
    # Create logging lists
    diff_volume_lists = [[0]]
    diff_volume_mins = [0]
    diff_volume_shifts = [[0, 0]]
    # Loop over volume
    for i in track(range(1, len(paths)), "Performing x-correlation..."):
        # Define params and logging arrays
        diff_layer_means = []
        delta_pair = []
        # Get current and previous slices
        layer_current = io.imread(paths[i + 1])
        if ref == "previous":
            layer_previous = io.imread(paths[i])
        elif ref == "first":
            layer_previous = io.imread(paths[0])
        # Get differences for each delta
        if "x" in dirs and "y" in dirs:
            for drow in delta:
                for dcol in delta:
                    delta_pair.append([drow, dcol])
                    slice_current, slice_previous = get_slice(drow, dcol)
                    diff_layer_mean = np.mean(
                        np.abs(
                            layer_current[slice_current][crop]
                            - layer_previous[slice_previous][crop]
                        )
                    )
                    diff_layer_means.append(diff_layer_mean)
        elif "x" in dirs and "y" not in dirs:
            for dcol in delta:
                delta_pair.append([0, dcol])
                slice_current, slice_previous = get_slice(0, dcol)
                diff_layer_mean = np.mean(
                    np.abs(
                        layer_current[slice_current][crop] - layer_previous[slice_previous][crop]
                    )
                )
                diff_layer_means.append(diff_layer_mean)
        elif "x" not in dirs and "y" in dirs:
            for drow in delta:
                delta_pair.append([drow, 0])
                slice_current, slice_previous = get_slice(drow, 0)
                diff_layer_mean = np.mean(
                    np.abs(
                        layer_current[slice_current][crop] - layer_previous[slice_previous][crop]
                    )
                )
                diff_layer_means.append(diff_layer_mean)
        else:
            raise ValueError("dirs must be x, y, or xy")
        # Convert lists to arrays
        delta_pair = np.array(delta_pair)
        diff_layer_means = np.array(diff_layer_means)
        # Find minimum
        minimum_layer = diff_layer_means.min()
        min_index = np.argmin(diff_layer_means)
        min_delta = delta_pair[min_index]
        diff_volume_lists.append(diff_layer_means)
        diff_volume_mins.append(minimum_layer)
        diff_volume_shifts.append(min_delta)
        colors, map = sp.mapColors(diff_layer_means, cmap="plasma", return_map=True)
        plt.scatter(delta_pair[:, 0], delta_pair[:, 1], s=100, marker="s", color=colors)
        plt.scatter(
            min_delta[0],
            min_delta[1],
            50,
            marker="2",
            color="cyan",
            linewidth=2,
        )
        plt.title(f"Slice {i} ({min_delta[0]},{min_delta[1]})")
        plt.xlabel("Row shift")
        plt.ylabel("Column shift")
        plt.colorbar(map)
        # plt.show()
        plt.savefig(f"D:/Research/scripts/paraview_analysis/gif_imgs2/{i}.png")
        plt.close()
    # plt.show()
    # return np.array(diff_volume_shifts)


def get_slice(dx, dy):
    # get slice for x axis
    if dx > 0:
        sliceROW_c = slice(0, -dx, None)
        sliceROW_p = slice(dx, None, None)
    elif dx == 0:
        sliceROW_c = slice(None, None, None)
        sliceROW_p = slice(None, None, None)
    else:
        sliceROW_c = slice(-dx, None, None)
        sliceROW_p = slice(0, dx, None)
    # get slice for y axis
    if dy > 0:
        sliceCOL_c = slice(0, -dy, None)
        sliceCOL_p = slice(dy, None, None)
    elif dy == 0:
        sliceCOL_c = slice(None, None, None)
        sliceCOL_p = slice(None, None, None)
    else:
        sliceCOL_c = slice(-dy, None, None)
        sliceCOL_p = slice(0, dy, None)
    # combine x and y slices and return
    current = (
        sliceROW_c,
        sliceCOL_c,
    )
    previous = (
        sliceROW_p,
        sliceCOL_p,
    )
    return (current, previous)


def register(stack, tmats=None, sr=StackReg.TRANSLATION, reference="first", save=True):
    if tmats is None:
        tmats = sr.register_stack(stack, axis=0, reference=reference, verbose=True)
        if save:
            np.save(f"{reference}_transMatrix.npy", tmats)
    elif type(tmats) == str:
        tmats = np.load(tmats)
    reg = np.copy(stack)
    for i in range(stack.shape[0]):
        tform = transform.AffineTransform(matrix=tmats[i, :, :])
        reg[i, :, :] = transform.warp(reg[i, :, :], tform)
    return reg


# Get ebsd
# h5 = h5py.File(os.path.join(folder, "Ta_AM-Spalled_aligned.dream3d"))
# ebsd = h5["DataContainers/ImageDataContainer/CellData/Image Quality"][...][:, :675, 7:, 0]
# ratio = ebsd.shape[2] / ebsd.shape[1]

# Get bse data
bse_folder = os.path.join(folder, "BSE/")
key_f = lambda x: int(x.replace(".tif", "").replace("Slice", "").replace(bse_folder, ""))
bse_paths = os.listdir(bse_folder)
paths = sorted([os.path.join(bse_folder, path) for path in bse_paths if ".tif" in path], key=key_f)
# bse = np.array([io.imread(path, as_gray=True) for path in paths])
bse = io.imread(paths[0], as_gray=True)
dy = 4050
dx = 4212
slice_y = slice(int(bse.shape[0] / 2 - dy / 2) - 23, int(bse.shape[0] / 2 + dy / 2) - 23)
slice_x = slice(int(bse.shape[1] / 2 - dx / 2) + 100, int(bse.shape[1] / 2 + dx / 2) + 100)

im0 = io.imread(paths[211])
im = io.imread(paths[212])
im1 = np.zeros(im.shape)
im1[:, 100:] = im[:, :-100]

plt.imshow(im0[:1000, :1000])
plt.imshow(im1[:1000, :1000], alpha=0.5, cmap="bone")
plt.title(f"{212} over {211}")
plt.show()
