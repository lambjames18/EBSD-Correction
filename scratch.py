import h5py
import InteractiveView
from skimage import morphology
import matplotlib.pyplot as plt
import numpy as np

h5 = h5py.File("D:/Research/CoNi_16/Data/3D/new/CoNi16_aligned_corrected.dream3d", "r")
print(h5["DataContainers/ImageDataContainer/CellData"].keys())
ipf = h5["DataContainers/ImageDataContainer/CellData/IPFColor_001"][...]
ci = h5["DataContainers/ImageDataContainer/CellData/Confidence Index"][..., 0]
iq = h5["DataContainers/ImageDataContainer/CellData/Image Quality"][..., 0]
h5.close()

mask = ci > 0.2
mask = morphology.remove_small_holes(mask, 100)
mask = morphology.binary_closing(mask, morphology.ball(3))
mask = morphology.remove_small_objects(mask, 100)
mask = morphology.binary_opening(mask, morphology.ball(3))

h5 = h5py.File("D:/Research/CoNi_16/Data/3D/new/CoNi16_aligned_corrected.dream3d", "r+")
h5["DataContainers/ImageDataContainer/CellData/Mask"][..., 0] = mask
h5.close()
