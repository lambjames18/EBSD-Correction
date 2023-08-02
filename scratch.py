import numpy as np
import h5py
import matplotlib.pyplot as plt
import imageio
from skimage import transform, io
import core

im = io.imread("D:/Research/Ta/Data/3D/AMSpall/BSE/small/0.tiff", as_gray=True)

im = im[54:-53, 99:-99]
print(im.shape)

plt.imshow(im, cmap='Blues')
plt.show()