import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io




cbs = "/Users/jameslamb/Documents/research/data/CoNi-DIC-S1/stitched_CBS.tif"
cbs_clahe = "/Users/jameslamb/Documents/research/data/CoNi-DIC-S1/stitched_CLAHE_CBS.tif"
etd = "/Users/jameslamb/Documents/research/data/CoNi-DIC-S1/stitched_ETD.tif"
etd_clahe = "/Users/jameslamb/Documents/research/data/CoNi-DIC-S1/stitched_CLAHE_ETD.tif"

cbs = io.imread(cbs)
etd = io.imread(etd)
cbs_clahe = io.imread(cbs_clahe)
etd_clahe = io.imread(etd_clahe)
print(cbs.shape, etd.shape, cbs_clahe.shape, etd_clahe.shape)
cbs = cbs[5:, :-3]
etd = etd[:-11, 5:]
cbs_clahe = cbs_clahe[5:, :-3]
etd_clahe = etd_clahe[:-11, 5:]
print(cbs.shape, etd.shape, cbs_clahe.shape, etd_clahe.shape)

# fig, ax = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
# ax[0, 0].imshow(cbs[:100, 100:200], cmap="gray")
# ax[0, 1].imshow(etd[:100, 100:200], cmap="gray")
# ax[1, 0].imshow(cbs_clahe[:100, 100:200], cmap="gray")
# ax[1, 1].imshow(etd_clahe[:100, 100:200], cmap="gray")
# for a in ax.flatten():
#     a.axis("off")
# plt.tight_layout()
# plt.show()

io.imsave("/Users/jameslamb/Documents/research/data/CoNi-DIC-S1/stitched_CBS.tif", cbs)
io.imsave("/Users/jameslamb/Documents/research/data/CoNi-DIC-S1/stitched_ETD.tif", etd)
io.imsave("/Users/jameslamb/Documents/research/data/CoNi-DIC-S1/stitched_CLAHE_CBS.tif", cbs_clahe)
io.imsave("/Users/jameslamb/Documents/research/data/CoNi-DIC-S1/stitched_CLAHE_ETD.tif", etd_clahe)
