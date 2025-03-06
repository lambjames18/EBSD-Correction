import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform

import warping
import tps


# Get the reference image
dst = "/Users/jameslamb/Documents/research/data/CoNi-DIC-S1/Step0_image.tif"
dshape = io.imread(dst).shape

# Get the transformation parameters
params = np.load("/Users/jameslamb/Documents/research/data/CoNi-DIC-S1/pre-post-transform.npy")
tform = tps.ThinPlateSplineTransform()
tform._estimated = True
tform.params = params

# Set the cropping and rescaling
s_res = 0.16276
d_res = 0.06201
ratio = s_res / d_res
crop = (slice(2100, 7800), slice(5100, 11800))

# Get the source data
src = "/Users/jameslamb/Documents/research/data/CoNi-DIC-S1/EBSD-SEM.h5"
old_h5 = h5py.File(src, "r")
# io.imsave("/Users/jameslamb/Documents/research/data/CoNi-DIC-S1/stitched_CBS_cropped_resized.tif", old_h5["CBS"][...][crop])
# exit()

# Create the new h5 file
src_new = "/Users/jameslamb/Documents/research/data/CoNi-DIC-S1/EBSD-SEM_aligned_HDIC.h5"
new_h5 = h5py.File(src_new, "w")
new_h5.attrs.create(name="resolution", data=old_h5.attrs["resolution"])
new_h5.attrs.create(name="header", data=old_h5.attrs["header"])

# keys = list(old_h5.keys())
keys = ["GrainIDs", "CBS", "ETD", "PRIAS Center Square"]

for key in keys:

    print(f"Processing {key} ({keys.index(key) + 1}/{len(keys)})")
    src = old_h5[key][...]
    src = src[crop]
    src = transform.warp(src, tform, output_shape=dshape, order=0, mode="constant", cval=0)

    if not np.allclose(src.shape, dshape):
        print(key, src.shape, dshape)
        break

    new_h5.create_dataset(name=key, data=src)
    new_h5[key].attrs.create(name="name", data=key)
    new_h5[key].attrs.create(name="dtype", data=str(src.dtype))
    new_h5[key].attrs.create(name="shape", data=src.shape)

keys = ["phi1", "PHI", "phi2"]
src = np.dstack([old_h5[key][()] for key in keys])
src = src[crop]
src = transform.warp(src, tform, output_shape=dshape, order=0, mode="constant", cval=0)
new_h5.create_dataset(name="EulerAngles", data=src)
new_h5["EulerAngles"].attrs.create(name="name", data="EulerAngles")
new_h5["EulerAngles"].attrs.create(name="dtype", data=str(src.dtype))
new_h5["EulerAngles"].attrs.create(name="shape", data=src.shape)

old_h5.close()
new_h5.close()
