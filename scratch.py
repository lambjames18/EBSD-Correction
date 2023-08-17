import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from skimage import transform, io, registration
import SIFT
import InteractiveView
import Inputs

# HFW 1.6 mm
se_paths = sorted([f for f in os.listdir('/Users/jameslamb/Documents/Research/CoNi67/se_images_aligned/') if f.endswith(".tif")], key=lambda f: int(f.split('.')[0]))
# bse_paths = sorted([f for f in os.listdir('/Users/jameslamb/Documents/Research/CoNi67/image_bse_images/') if f.endswith(".tif")], key=lambda f: int(f.split('.')[0].replace("Slice", "")))
se_imgs = np.array([np.flipud(io.imread('/Users/jameslamb/Documents/Research/CoNi67/se_images_aligned/' + p)) for p in se_paths])
se_imgs_small = Inputs.rescale_control(se_imgs, 0.52, 1.5)
os.makedirs('/Users/jameslamb/Documents/Research/CoNi67/se_images_aligned_small/', exist_ok=True)
for i in range(se_imgs_small.shape[0]):
    io.imsave('/Users/jameslamb/Documents/Research/CoNi67/se_images_aligned_small/{}.tif'.format(i), se_imgs_small[i])
exit()

# d3d = h5py.File('/Users/jameslamb/Documents/Research/CoNi67/CoNi67_basic.dream3d', 'r')
# fig, ax = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
# ax[0, 0].imshow(d3d["DataStructure/DataContainer/CellData/Image Quality"][0, :, :, 0])
# ax[0, 1].imshow(d3d["DataStructure/DataContainer/CellData/Confidence Index"][0, :, :, 0])
# ax[0, 2].imshow(d3d["DataStructure/DataContainer/CellData/XC Metric"][0, :, :, 0])
# ax[1, 0].imshow(d3d["DataStructure/DataContainer/CellData/EulerAngles"][1, :, :, 0])
# ax[1, 1].imshow(d3d["DataStructure/DataContainer/CellData/Phases"][1, :, :, 0])
# ax[1, 2].imshow(d3d["DataStructure/DataContainer/CellData/X Position"][1, :, :, 0])
# plt.show()
# d3d.close()
# exit()

paths = sorted([f for f in os.listdir('/Users/jameslamb/Documents/Research/CoNi67/h5/') if f.endswith(".h5")], key=lambda f: int(f.split('.')[0]))

d3d = h5py.File('/Users/jameslamb/Documents/Research/CoNi67/CoNi67_basic.dream3d', 'r+')

for i, p in enumerate(paths):
    index = p.split('.')[0]
    h5 = h5py.File('/Users/jameslamb/Documents/Research/CoNi67/h5/' + p, 'r')
    xc = h5['Scan 1']['EBSD']['Data']['Metric'][:]
    iq = h5['Scan 1']['EBSD']['Data']['IQ'][:]
    phi1 = h5['Scan 1']['EBSD']['Data']['Phi1'][:]
    Phi = h5['Scan 1']['EBSD']['Data']['Phi'][:]
    phi2 = h5['Scan 1']['EBSD']['Data']['Phi2'][:]
    phase = h5['Scan 1']['EBSD']['Data']['Phase'][:]
    y, x = np.indices((534, 587))
    h5.close()
    print(index)
    d3d["DataStructure/DataContainer/CellData/Image Quality"][i, :, :, 0] = np.fliplr(iq.reshape(534, 587))
    d3d["DataStructure/DataContainer/CellData/Confidence Index"][i, :, :, 0] = np.fliplr(xc.reshape(534, 587))
    d3d["DataStructure/DataContainer/CellData/EulerAngles"][i, :, :, 0] = np.fliplr(phi1.reshape(534, 587))
    d3d["DataStructure/DataContainer/CellData/EulerAngles"][i, :, :, 1] = np.fliplr(Phi.reshape(534, 587))
    d3d["DataStructure/DataContainer/CellData/EulerAngles"][i, :, :, 2] = np.fliplr(phi2.reshape(534, 587))
    d3d["DataStructure/DataContainer/CellData/Phases"][i, :, :, 0] = np.fliplr(phase.reshape(534, 587))
    d3d["DataStructure/DataContainer/CellData/X Position"][i, :, :, 0] = x.reshape(534, 587)
    d3d["DataStructure/DataContainer/CellData/Y Position"][i, :, :, 0] = y.reshape(534, 587)

d3d.close()