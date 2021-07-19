# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 11:52:54 2021

@author: Arsenic
"""

import numpy as np
import h5py
from skimage import io

path = "D:\\Research\\NiAlMo_APS\\Data\\3D\\Basic2.dream3d"
hf = h5py.File(path, "r")
feat = np.array(hf.get("DataContainers/ImageDataContainer/CellData/Image Quality"))

slice_id = 22
print(feat.shape)

slice_ebsd = np.copy(feat[slice_id, :, :, 0])
# slice_ebsd = np.sum(np.square(slice_ebsd), axis=2)
# io.imsave("slice_%s_ebsd.png" % (slice_id), slice_ebsd)
# io.imsave("slice_%s_ebsd.tif" % (slice_id), slice_ebsd)
