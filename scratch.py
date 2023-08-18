import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from skimage import transform, io, registration
import SIFT
import InteractiveView
import Inputs
import core
from scipy import interpolate

h5 = h5py.File("/Users/jameslamb/Documents/Research/CoNi67/Test.dream3d", "r")
print(h5["DataStructure/DataContainer/CellData"].keys())
data = h5["DataStructure/DataContainer/CellData/Image Quality"][:]
h5.close()

InteractiveView.Interactive3D(data, data)