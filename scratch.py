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

def read_csv(path):
    data = np.genfromtxt(path, delimiter=",", skip_header=5, dtype=int)[:, 1:]
    return data

folder = "C:/Users/lambj/Downloads/CoNi16-EDS/EDS/"

al_paths = [f for f in os.listdir(folder) if "_al" in f]
o_paths = [f for f in os.listdir(folder) if "_o" in f]
fov_paths = [f for f in os.listdir(folder) if "_fov" in f]
cps_paths = [f for f in os.listdir(folder) if "_cps" in f]

al_paths.sort()
o_paths.sort()
fov_paths.sort()
cps_paths.sort()

al_imgs = [read_csv(folder + f) for f in al_paths]
o_imgs = [read_csv(folder + f) for f in o_paths]
fov_imgs = [read_csv(folder + f) for f in fov_paths]
cps_imgs = [read_csv(folder + f) for f in cps_paths]

for i in range(len(al_imgs)):
    eds_img = al_imgs[i] + o_imgs[i]
    eds_img = np.around(eds_img / np.max(eds_img) * 255, 0).astype(np.uint8).T
    io.imsave(folder + "Al2O3-EDS_" + str(i) + ".tif", eds_img)
    fov_img = fov_imgs[i] / fov_imgs[i].max() + cps_imgs[i] / cps_imgs[i].max()
    fov_img = np.around(fov_img / np.max(fov_img) * 255, 0).astype(np.uint8).T
    io.imsave(folder + "FoV-EDS_" + str(i) + ".tif", fov_img)
    img = np.dstack((al_imgs[i]/al_imgs[i].max(), o_imgs[i]/o_imgs[i].max(), cps_imgs[i]/cps_imgs[i].max()))
    img = np.around(img / np.max(img) * 255, 0).astype(np.uint8).T
    io.imsave(folder + "EDS_" + str(i) + ".tif", img)