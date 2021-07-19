# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:32:16 2019

@author: Arsenic
"""
import h5py
import numpy as np
from skimage import io

from rich.progress import track


hf = h5py.File("D:\\Research\\NiAlMo_APS\\Data\\3D\\Basic3.dream3d", "r+")
import skimage.transform as tf


nz = 391
ny = 781
nx = 1101


params = np.load("paramsDistortion.npy")
transform = tf._geometric.PolynomialTransform(params)


def applyDistortion(array):
    dtype = array.dtype
    for slice in track(range(nz), "Applying distortion to volume..."):
        if array.shape[-1] > 1:
            array_xy = np.copy(array[slice])
        else:
            array_xy = np.copy(array[slice, :, :, 0])
        buf = 50
        dims_buffered = np.array(array_xy.shape)
        dims_buffered[:2] += buf
        buffered = np.zeros(dims_buffered)
        buffered[int(buf / 2) : -int(buf / 2), int(buf / 2) : -int(buf / 2)] = array_xy
        if array.shape[-1] > 1:
            for i in range(buffered.shape[-1]):
                array_distort = tf.warp(
                    buffered[:, :, i],
                    transform,
                    cval=0,  # new pixels are black pixel
                    order=0,  # k-neighbour
                    preserve_range=True,
                )
                array_distort = array_distort[:-buf, buf:]
                array_distort = np.round(array_distort).astype(float)
                array[slice, :, :, i] = np.copy(array_distort)
        if array.shape[-1] == 1:
            array_distort = tf.warp(
                buffered,
                transform,
                cval=0,  # new pixels are black pixel
                order=0,  # k-neighbour
                preserve_range=True,
            )
            array_distort = array_distort[:-buf, buf:]
            array_distort = np.round(array_distort).astype(dtype)

        array[slice, :, :] = np.copy(array_distort).reshape(*array_distort.shape, 1)
    return array


for key in hf["DataContainers/ImageDataContainer/CellData"]:
    print(f"\nStarting {key}")
    data = np.array(hf.get(f"DataContainers/ImageDataContainer/CellData/{key}"))
    data_corrected = applyDistortion(data)
    data_original = hf[f"DataContainers/ImageDataContainer/CellData/{key}"]
    data_original[...] = data_corrected
    print(f"{key} correction completed")

"""
# 3D * 1 arrays int
bc = np.array(hf.get("DataContainers/ImageDataContainer/CellData/BC"))
bc = applyDistortion_int(bc)
bandContrast = hf["DataContainers/ImageDataContainer/CellData/BC"]
bandContrast[...] = bc
print("BC correction completed")

featids = np.array(hf.get("DataContainers/ImageDataContainer/CellData/FeatureIds"))
featids = applyDistortion_int(featids)
featidS = hf["DataContainers/ImageDataContainer/CellData/FeatureIds"]
featidS[...] = featids
print("feat ids correction completed")

kam = np.array(hf.get("DataContainers/ImageDataContainer/CellData/KernelAverageMisorientations"))
kam = applyDistortion_flt(kam)
KAM = hf["DataContainers/ImageDataContainer/CellData/KernelAverageMisorientations"]
KAM[...] = kam
print("KAM correction completed")

mask = np.array(hf.get("DataContainers/ImageDataContainer/CellData/Mask"))
mask = applyDistortion_int(mask)
Mask = hf["DataContainers/ImageDataContainer/CellData/Mask"]
Mask[...] = mask
print("Mask correction completed")

phases = np.array(hf.get("DataContainers/ImageDataContainer/CellData/Phases"))
phases = applyDistortion_int(phases)
Phases = hf["DataContainers/ImageDataContainer/CellData/Phases"]
Phases[...] = phases
print("Phases correction completed")


# 3D * 1 arrays float
ne = np.array(hf.get("DataContainers/ImageDataContainer/CellData/NumElements"))
ne = applyDistortion_int(ne)
NE = hf["DataContainers/ImageDataContainer/CellData/NumElements"]
NE[...] = ne
print("num elements correction completed")


x = np.array(hf.get("DataContainers/ImageDataContainer/CellData/X"))
x = applyDistortion_flt(x)
X = hf["DataContainers/ImageDataContainer/CellData/X"]
X[...] = x
print("X coordinate correction completed")

y = np.array(hf.get("DataContainers/ImageDataContainer/CellData/Y"))
y = applyDistortion_flt(y)
Y = hf["DataContainers/ImageDataContainer/CellData/Y"]
Y[...] = y
print("Y coordinate correction completed")


# 3D * 3 arrays: Euler Angles
euler = np.array(hf.get("DataContainers/ImageDataContainer/CellData/EulerAngles"))
euler1 = np.copy(euler[:, :, :, 0])
euler2 = np.copy(euler[:, :, :, 1])
euler3 = np.copy(euler[:, :, :, 2])

euler1 = applyDistortion_euler(euler1)
euler2 = applyDistortion_euler(euler2)
euler3 = applyDistortion_euler(euler3)

euler[:, :, :, 0] = np.copy(euler1)
euler[:, :, :, 1] = np.copy(euler2)
euler[:, :, :, 2] = np.copy(euler3)

Euler = hf["DataContainers/ImageDataContainer/CellData/EulerAngles"]
Euler[...] = euler
print("Euler angles correction completed")


# quaternions
quat = np.array(hf.get("DataContainers/ImageDataContainer/CellData/Quats"))
qu1 = np.copy(quat[:, :, :, 0])
qu2 = np.copy(quat[:, :, :, 1])
qu3 = np.copy(quat[:, :, :, 2])
qu4 = np.copy(quat[:, :, :, 3])

qu1 = applyDistortion_euler(qu1)
qu2 = applyDistortion_euler(qu2)
qu3 = applyDistortion_euler(qu3)
qu4 = applyDistortion_euler(qu4)


quat[:, :, :, 0] = np.copy(qu1)
quat[:, :, :, 1] = np.copy(qu2)
quat[:, :, :, 2] = np.copy(qu3)
quat[:, :, :, 3] = np.copy(qu4)

Quat = hf["DataContainers/ImageDataContainer/CellData/Quats"]
Quat[...] = quat
print("Quaternions correction completed")


# IPF Color
ipf = np.array(hf.get("DataContainers/ImageDataContainer/CellData/IPFColor"))
ipf1 = np.copy(ipf[:, :, :, 0])
ipf2 = np.copy(ipf[:, :, :, 1])
ipf3 = np.copy(ipf[:, :, :, 2])

ipf1 = applyDistortion_euler(ipf1)
ipf2 = applyDistortion_euler(ipf2)
ipf3 = applyDistortion_euler(ipf3)

ipf[:, :, :, 0] = np.copy(ipf1)
ipf[:, :, :, 1] = np.copy(ipf2)
ipf[:, :, :, 2] = np.copy(ipf3)

IPF = hf["DataContainers/ImageDataContainer/CellData/IPFColor"]
IPF[...] = ipf
print("IPF colors correction completed")
"""

hf.close()
# slice0 = np.copy(bc[0,:,:,0])
# slice1 = np.copy(bc[1,:,:,0])
# io.imsave("slice1.tif",slice1)
