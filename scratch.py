import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import imageio
from skimage import transform, io
import core


folder = "D:/Research/CMU-Registration/Datasets/Ta_AMS/Ta_AMS_small_BSE/"

paths = sorted([f for f in os.listdir(folder) if f.endswith(".tiff")], key=lambda x: int(x.split(".")[0]))
print(paths)

for i, p in enumerate(paths):
    img = io.imread(os.path.join(folder, p))
    img = np.fliplr(img)
    new_p = f"{i}.tif"
    print(p, new_p, img.shape)
    io.imsave(os.path.join(folder, new_p), img)