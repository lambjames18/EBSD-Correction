import numpy as np
import h5py
import matplotlib.pyplot as plt
import imageio
from skimage import transform, io
import core

def write_coords(points, points_path):
    for mode in ["ebsd", "bse"]:
        path = points_path[mode]
        data = []
        pts = points[mode]
        for key in pts.keys():
            s = np.hstack((np.ones((len(pts[key]), 1)) * key, pts[key]))
            data.append(s)
        data = np.vstack(data)
        np.savetxt(path, data, fmt="%i", delimiter=" ")

points_path = {"ebsd": "test_data/points_ebsd.txt", "bse": "test_data/points_bse.txt"}
p = [[0, 0], [0, 1], [1, 0], [1, 1]]
points = {"ebsd": {0: p, 1: p}, "bse": {0: p, 1: p}}

write_coords(points, points_path)