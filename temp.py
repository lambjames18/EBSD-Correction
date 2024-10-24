import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, transform
import h5py
from tqdm.auto import tqdm

h5 = h5py.File("D:/Research/CoNi_16/Data/3D/CoNi16_BSE.dream3d", "r")
bse = h5["DataContainers/ImageDataContainer/CellData/BSE"][..., 0]
spacing = h5["DataContainers/ImageDataContainer/_SIMPL_GEOMETRY/SPACING"][...]
h5.close()

scale = spacing[0] / spacing[2]

for i in tqdm(range(bse.shape[0])):
    im = bse[i]
    im_blur = filters.gaussian(im, sigma=1)
    im_down = transform.rescale(im_blur, scale, anti_aliasing=False)
    im_out = np.around( 255 * (im_down - im_down.min()) / (im_down.max() - im_down.min()) ).astype(np.uint8)
    io.imsave(f"D:/Research/CoNi_16/Data/3D/BSE/{i}.png", im_out)