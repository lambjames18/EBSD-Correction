import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage import transform, registration
from tqdm import tqdm

from InteractiveView import Interactive3D_Alignment


def shift_image(img, shift):
    shift = np.array(shift)
    # Pad the image before rolling
    row_pad = (abs(shift[0]), abs(shift[0]))
    col_pad = (abs(shift[1]), abs(shift[1]))
    pad = (row_pad, col_pad, (0, 0))
    img = np.pad(img, pad, "constant")
    # Shift the image
    img = np.roll(img, shift, axis=(0, 1))
    # Crop the image back to original size
    img = img[
        row_pad[0] : img.shape[0] - row_pad[1],
        col_pad[0] : img.shape[1] - col_pad[1],
        :,
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
    # from pystackreg import StackReg

    unaligned_dream3d_path = (
        "/Users/jameslamb/Documents/research/data/CoNi90-thin/CoNi90-thin_basic.dream3d"
    )
    aligned_dream3d_path = "/Users/jameslamb/Documents/research/data/CoNi90-thin/CoNi90-thin_aligned.dream3d"
    calculate = True
    # slc = (slice(None), slice(None))
    slc = (slice(800, 1200), slice(800, 1200))
    slc = (slice(None, 600), slice(600, None))
    slc = (slice(400, 800), slice(400, 800))

    if calculate:
        # Get the images for alignment
        h5_in = h5py.File(unaligned_dream3d_path, "r")
        in_key = get_key_string(h5_in)
        imgs1 = h5_in[in_key + "IPFColors_001"][:]
        imgs2 = h5_in[in_key + "Confidence Index"][:]
        h5_in.close()

        imgs1 = imgs1.astype(np.float32) / 255.0
        imgs2 = np.clip(imgs2, np.percentile(imgs2, 1), np.percentile(imgs2, 99))
        imgs2 = (imgs2 - imgs2.min()) / (imgs2.max() - imgs2.min())
        imgs = imgs1 * imgs2
        imgs = (imgs - np.min(imgs, axis=(1, 2), keepdims=True)) / (
            np.max(imgs, axis=(1, 2), keepdims=True)
            - np.min(imgs, axis=(1, 2), keepdims=True)
        )

        center = np.array(imgs.shape[:-1]) // 2

        all_shifts = []
        idx0 = np.arange(imgs.shape[0])[:-1][::-1]
        idx1 = np.arange(imgs.shape[0])[1:][::-1]

        for i in tqdm(range(idx0.shape[0])):
            ref = imgs[idx1[i]]
            misaligned = imgs[idx0[i]]
            shifts, error, phasediff = registration.phase_cross_correlation(
                ref[slc], misaligned[slc], normalization=None
            )
            all_shifts.append((idx0[i], idx1[i], shifts[1], shifts[0]))

        all_shifts = np.array(all_shifts, dtype=int)
        cum_shifts = np.cumsum(all_shifts[:, 2:4], axis=0)
        all_shifts = np.hstack((all_shifts, cum_shifts))
        np.savetxt(
            "/Users/jameslamb/Documents/research/data/CoNi90-thin/manual_shifts.txt",
            all_shifts,
            delimiter="\t",
            fmt="%d",
            comments="#",
        )

    else:
        # Load the shifts
        all_shifts = np.loadtxt(
            "/Users/jameslamb/Documents/research/data/CoNi90-thin/manual_shifts.txt",
            delimiter="\t",
            dtype=int,
            comments="#",
        )

    # Load the unaligned data
    h5_in = h5py.File(unaligned_dream3d_path, "r")
    h5_out = h5py.File(aligned_dream3d_path, "r+")

    # Get keys
    in_key = get_key_string(h5_in)
    out_key = get_key_string(h5_out)

    keys = h5_in[in_key].keys()

    # fig, ax = plt.subplots(2, 2, figsize=(4, 6), sharex=True, sharey=True)
    # ax[0, 0].imshow(h5_in[in_key + "IPFColors_001"][:, :, 600])
    # ax[0, 1].imshow(h5_in[in_key + "IPFColors_001"][:, 600])

    for i in tqdm(range(len(all_shifts))):
        # idx0, idx1, xshift, yshift, _, _ = all_shifts[i]
        idx0, idx1, _, _, xshift, yshift = all_shifts[i]

        for key in keys:
            # Get the data slice
            data = h5_in[in_key + key][idx0, ...]
            # Shift the data
            data = shift_image(data, (yshift, xshift))
            # Store the data
            h5_out[out_key + key][idx0, ...] = data

        imgs[idx0] = shift_image(imgs[idx0], (yshift, xshift))

    Interactive3D_Alignment(
        imgs,
        # h5_out[in_key + "IPFColors_001"][:],
        82,
        "Aligned",
    )

    # ax[1, 0].imshow(h5_out[in_key + "IPFColors_001"][:, :, 600])
    # ax[1, 1].imshow(h5_out[in_key + "IPFColors_001"][:, 600])
    # ratio = 1.61
    # xleft, xright = ax[0, 0].get_xlim()
    # ybottom, ytop = ax[0, 0].get_ylim()
    # ax[0, 0].set_aspect(abs((xright - xleft) / (ybottom - ytop)) * ratio)
    # ax[0, 1].set_aspect(abs((xright - xleft) / (ybottom - ytop)) * ratio)
    # ax[1, 0].set_aspect(abs((xright - xleft) / (ybottom - ytop)) * ratio)
    # ax[1, 1].set_aspect(abs((xright - xleft) / (ybottom - ytop)) * ratio)
    # plt.tight_layout()
    # plt.show()

    # Close the files
    h5_in.close()
    h5_out.close()
    print("Done")
