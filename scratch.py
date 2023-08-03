import numpy as np
import h5py
import matplotlib.pyplot as plt
import imageio
from skimage import transform, io
import core


h = h5py.File("test_data/Test.dream3d", "r")
print(h["DataStructure/DataContainer"].attrs.get("_SPACING"))
h.close()