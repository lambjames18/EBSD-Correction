import numpy as np
import h5py
import matplotlib.pyplot as plt
import imageio
from skimage import transform, io

# h5 = h5py.File("D:/Research/CoNi_16/Data/3D/CoNi16_test.dream3d", "r")
# s = "DataContainers/ImageDataContainer/CellData/SEM"
# d = h5[s][:, :, :, 0]
# plt.imshow(d[-79])
# plt.show()
path = "D:/Research/scripts/Alignment/Slice78_CoNi16/aligned_0078.tif"
d = imageio.imread(path)
bse_slice_x = slice(476, 2696)
bse_slice_y = slice(175, 1840)
d = transform.resize(d[bse_slice_y, bse_slice_x, 0], (483, 644)).astype(np.float32)
print(d.shape)
imageio.imsave("D:/Research/scripts/Alignment/Slice78_CoNi16/bse.tif", d)
