import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform, exposure

import warping
import tps


# Get the reference images
dst_bse = (
    "/Users/jameslamb/Documents/research/data/CoNi-DIC-S1/full_images/stitched_CBS.tif"
)
bse_im = io.imread(dst_bse)
bse_shape = bse_im.shape
dst_dic = (
    "/Users/jameslamb/Documents/research/data/CoNi-DIC-S1/full_images/Step0_image.tif"
)
dic_im = io.imread(dst_dic)
dic_shape = dic_im.shape

# Get the transformation parameters
ebsd2bse_params = np.load(
    "/Users/jameslamb/Documents/research/data/CoNi-DIC-S1/transform/EBSD-BSE-transform.npy"
)
EBSD_2_BSE = tps.ThinPlateSplineTransform()
EBSD_2_BSE._estimated = True
EBSD_2_BSE.params = ebsd2bse_params
bse2dic_params = np.load(
    "/Users/jameslamb/Documents/research/data/CoNi-DIC-S1/transform/pre-post-transform.npy"
)
BSE_2_DIC = tps.ThinPlateSplineTransform()
BSE_2_DIC._estimated = True
BSE_2_DIC.params = bse2dic_params

# Set the cropping and rescaling
ebsd_res = 1.5
bse_res = 0.16276
dic_res = 0.06201
ebsd2bse_crop = (slice(None), slice(None))
bse2dic_crop = (slice(2100, 7800), slice(5100, 11800))

# Get the data
path = "/Users/jameslamb/Documents/research/data/CoNi-DIC-S1/stitched_EBSD.dream3d"
h5 = h5py.File(path, "r")
# gnd = h5["DataStructure/ImageDataContainer/CellData/GND"][0, ..., 0]
# fdam = h5["DataStructure/ImageDataContainer/CellData/FDAM"][0, ..., 0]
# ipf_ld = h5["DataStructure/ImageDataContainer/CellData/IPF_LD"][0, ...]
# ipf_bd = h5["DataStructure/ImageDataContainer/CellData/IPF_BD"][0, ...]
# quats = h5["DataStructure/ImageDataContainer/CellData/Quats"][0, ...]
ids = h5["DataStructure/ImageDataContainer/CellData/FeatureIds"][0, ..., 0]
h5.close()
# gnd = np.log10(np.clip(gnd, 1, None))
# quats = np.roll(quats, 1, axis=-1)

# datas = [gnd, fdam, ipf_ld, ipf_bd, quats, ids]
# keys = ["GND", "FDAM", "IPF_LD", "IPF_BD", "Quats", "GrainIDs"]
datas = [ids]
keys = ["GrainIDs"]
original = datas[0].copy()


# EBSD -> BSE -> DIC
for i in range(len(datas)):
    print(f"Processing {keys[i]} ({i + 1}/{len(datas)})")
    data = datas[i][ebsd2bse_crop]
    data = transform.warp(
        data, EBSD_2_BSE, output_shape=bse_shape, order=0, mode="constant", cval=0
    )
    data = data[bse2dic_crop]
    data = transform.warp(
        data, BSE_2_DIC, output_shape=dic_shape, order=0, mode="constant", cval=0
    )
    datas[i] = data

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(original, cmap="gray")
ax[0].set_title("Original")
ax[1].imshow(datas[0], cmap="gray")
ax[1].set_title("Warped")
plt.tight_layout()
plt.show()

# Save the data
h5_path = "/Users/jameslamb/Documents/research/data/CoNi-DIC-S1/HDIC_results.h5"
h5 = h5py.File(h5_path, "a")

# del h5["step0"]["EulerAngles"]
for key in keys:
    print(f"Saving {key} ({keys.index(key) + 1}/{len(keys)})")
    if key in h5["step0"].keys():
        del h5["step0"][key]
    h5["step0"].create_dataset(name=key, data=datas[keys.index(key)])
    h5["step0"][key].attrs.create(name="name", data=key)
    h5["step0"][key].attrs.create(name="dtype", data=str(datas[keys.index(key)].dtype))
    h5["step0"][key].attrs.create(name="shape", data=datas[keys.index(key)].shape)

h5.close()
