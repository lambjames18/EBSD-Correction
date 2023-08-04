import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import imageio
from skimage import transform, io
import core


path = "test_data/WCu/BSE/"
files = sorted([f for f in os.listdir(path) if f.endswith(".tif")])
print(files)

for i, f in enumerate(files):
    new_name = os.path.join(path, f"{i}.tif")
    old_name = os.path.join(path, f)
    print(old_name, ">", new_name)
    # os.rename(old_name, new_name)