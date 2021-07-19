# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 11:52:54 2021

@author: Arsenic
"""

import numpy as np
import h5py
from PIL import Image


hf = h5py.File('Ti7_1percent_resegmented.dream3d', 'r')
feat = np.array(hf.get('DataContainers/ImageDataContainer/CellData/FeatureIds'))

slice_id = 76

slice_ebsd = np.copy(feat[:,slice_id,:,0])
slice_ebsd_im = Image.fromarray(slice_ebsd)
slice_ebsd_im.save("slice_%s_3debsd.tiff" %(slice_id))