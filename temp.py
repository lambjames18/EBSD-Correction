import numpy as np
import h5py
import matplotlib.pyplot as plt
import imageio
from skimage import transform, io

bb = ["0001", "0002", "0004", "0008", "0010", "0080", "0100", "0200", "0400"]
options = {4: "Ctrl",
           1: "Shift",
           8: "NumLock",
           10: "CapsLock",
           1024: "RightClick",
           256: "LeftClick",
           512: "MiddleClick",
           131072: "Alt"}

print(hex(1 + 8))

for option in options.items():
    print(option)
    print(hex(option[0]))
