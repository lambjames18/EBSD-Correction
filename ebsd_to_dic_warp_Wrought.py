import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform, exposure

import warping
import tps


# Get the reference images
dst_dic = "/Users/jameslamb/Documents/research/data/Wrought-DIC/step0.tif"
dic_im = io.imread(dst_dic)
dic_shape = dic_im.shape

# Get the transformation parameters
ebsd2dic_params = np.load(
    "/Users/jameslamb/Documents/research/data/Wrought-DIC/EBSD2DIC_transform.npy"
)
EBSD_2_DIC = tps.ThinPlateSplineTransform()
EBSD_2_DIC._estimated = True
EBSD_2_DIC.params = ebsd2dic_params

# Get the data
path = "/Users/jameslamb/Documents/research/data/Wrought-DIC/EBSD.dream3d"
h5 = h5py.File(path, "r")
ipf_ld = h5["DataStructure/ImageGeometry/Cell Data/IPF_LD"][0, ...]
quats = h5["DataStructure/ImageGeometry/Cell Data/Quats"][0, ...]
ids = h5["DataStructure/ImageGeometry/Cell Data/FeatureIds"][0, ..., 0]
h5.close()
quats = np.roll(quats, 1, axis=-1)

datas = [ipf_ld, quats, ids]
keys = ["IPF_LD", "Quats", "GrainIDs"]

# EBSD -> DIC
for i in range(len(datas)):
    print(f"Processing {keys[i]} ({i + 1}/{len(datas)})")
    data = datas[i]
    data = transform.warp(
        data, EBSD_2_DIC, output_shape=dic_shape, order=0, mode="constant", cval=0
    )
    datas[i] = data

# Save the data
h5_path = "/Users/jameslamb/Documents/research/data/Wrought-DIC/HDIC_results.h5"
h5 = h5py.File(h5_path, "a")

for key in keys:
    if key in h5["step0"].keys():
        del h5["step0"][key]
    h5["step0"].create_dataset(name=key, data=datas[keys.index(key)])
    h5["step0"][key].attrs.create(name="name", data=key)
    h5["step0"][key].attrs.create(name="dtype", data=str(datas[keys.index(key)].dtype))
    h5["step0"][key].attrs.create(name="shape", data=datas[keys.index(key)].shape)

h5.close()
