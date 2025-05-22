import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage import transform, registration, morphology, filters
from tqdm import tqdm

import Inputs
import Outputs


def shift_image(img, shift):
    shift = np.array(shift)
    # Pad the image before rolling
    row_pad = (abs(shift[0]), abs(shift[0]))
    col_pad = (abs(shift[1]), abs(shift[1]))
    if img.ndim == 2:
        pad = (row_pad, col_pad)
    else:
        pad = (row_pad, col_pad, (0, 0))
    img = np.pad(img, pad, "constant")
    # Shift the image
    img = np.roll(img, shift, axis=(0, 1))
    # Crop the image back to original size
    img = img[
        row_pad[0] : img.shape[0] - row_pad[1],
        col_pad[0] : img.shape[1] - col_pad[1],
    ]
    return img


def get_key_string(h5):
    # Get the top key
    top_keys = list(h5.keys())
    if len(top_keys) == 1:
        top_key = top_keys[0]
    else:
        if "Pipeline" in top_keys:
            top_keys.pop(top_keys.index("Pipeline"))
            top_key = top_keys[0]
        else:
            top_keys_lower = [k.lower() for k in top_keys]
            top_key = [k for k in top_keys_lower if "data" in k][0]

    # Get the second key, there should only be one
    second_key = list(h5[top_key].keys())[0]

    # Find the last key, it should only have the words "Cell" and "Data" in it
    last_keys = list(h5[top_key][second_key].keys())
    last_keys_lower = [
        k for k in last_keys if "cell" in k.lower() and "data" in k.lower()
    ]
    last_keys_stripped = [
        k.lower().replace("cell", "").replace("data", "").strip()
        for k in last_keys_lower
    ]
    last_key = last_keys[last_keys_stripped.index("")]

    out = f"{top_key}/{second_key}/{last_key}/"
    return out


if __name__ == "__main__":
    ang0 = "/Users/jameslamb/Documents/research/data/CoNi90-thin/ANG/slice_0114_001_Rescan_Mod.ang"
    ang1 = "/Users/jameslamb/Documents/research/data/CoNi90-thin/ANG/slice_0114_002_Rescan_Mod.ang"

    # Get the images for alignment
    ang0 = Inputs.read_ang(ang0)[0]
    ang1 = Inputs.read_ang(ang1)[0]

    imgs = np.array([ang0["CI"][0], ang1["CI"][0]])
    # shifts, error, phasediff = registration.phase_cross_correlation(
    # imgs[0][400:-400, 400:-400], imgs[1][400:-400, 400:-400], normalization=None
    # )
    # shifts = (int(shifts[1]), int(shifts[0]))
    shifts = (-11, 6)
    for key in ang1.keys():
        ang1[key][0] = shift_image(ang1[key][0], shifts)

    shifts = (3, 2)
    for key in ang1.keys():
        ang1[key][0][:200, :400] = shift_image(ang1[key][0][:200, :400], shifts)

    slc = ((slice(220, 360), slice(469, 543)), (slice(0, 122), slice(187, 252)))

    # fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    # ax[0].imshow(ang0["CI"][0], cmap="gray")
    # ax[1].imshow(ang1["CI"][0], cmap="gray")
    # plt.show()

    mask_full = np.zeros_like(ang0["IQ"][0], dtype=bool)
    for i, s in enumerate(slc):
        im0 = ang0["IQ"][0][s]
        mask = ang0["IQ"][0][s] < 450
        mask = morphology.binary_opening(mask, footprint=morphology.disk(1))
        mask = morphology.binary_closing(mask, footprint=morphology.disk(1))
        mask = morphology.remove_small_objects(mask, min_size=mask.sum() / 10)
        mask = morphology.binary_dilation(mask, footprint=morphology.disk(10))
        mask = filters.gaussian(mask.astype(float), sigma=2) > 0.5
        mask_full[s] = mask

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    ax[0].imshow(ang0["CI"][0], cmap="gray")

    for key in ang1.keys():
        if ang1[key].ndim == 4:
            _mask = mask_full & (ang1[key][0, ..., 0] != 0)
        else:
            _mask = mask_full & (ang1[key][0] != 0)
        ang0[key][0][_mask] = ang1[key][0][_mask]

    ax[1].imshow(ang0["CI"][0], cmap="gray")

    ax[0].axis("off")
    ax[1].axis("off")
    plt.tight_layout()
    plt.show()

    Outputs.write_ang_file(
        "/Users/jameslamb/Documents/research/data/CoNi90-thin/ANG/slice_0114_001_Rescan_Mod.ang",
        "/Users/jameslamb/Documents/research/data/CoNi90-thin/ANG/slice_0114_Rescan_Mod.ang",
        ang0,
    )
