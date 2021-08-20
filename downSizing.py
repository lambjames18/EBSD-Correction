import os
import sys

import numpy as np
import skimage.transform as tf
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib

full_res_path = "D:/Research/NiAlMo_APS/Data/ALL_BSE/"
small_res_path = "D:/Research/NiAlMo_APS/Data/ALL_BSE_resized/"
# bse_name = "385_bse.tif"
# ebsd_name = "385_ebsd.tif"
imgs = os.listdir(full_res_path)
ebsd_shape = (781, 1101)
for img in imgs:
    bse = io.imread(os.path.join(full_res_path, img))
    print(img, bse.shape)
    # bse_resized = tf.resize(bse, ebsd_shape, anti_aliasing=True)
    # matplotlib.image.imsave(
    #     os.path.join(small_res_path, img + "f"), bse_resized, cmap="gray", dpi=1
    # )
    # print(f"Saved image {img}")
