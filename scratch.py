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

def linear(p0, p1, x):
    m = (p1[1] - p0[1]) / (p1[0] - p0[0])
    b = p0[1] - m * p0[0]
    return m * x + b


M = np.array([[1, 2, 0], [0, 1, 0], [0, 0, 1]])
points0 = np.loadtxt("/Users/jameslamb/Documents/Research/CoNi67/control_pts.txt")
c0 = points0[points0[:, 0] == 0][:, 1:]
c1 = points0[points0[:, 0] == 49][:, 1:]
c2 = points0[points0[:, 0] == 99][:, 1:]
points1 = np.loadtxt("/Users/jameslamb/Documents/Research/CoNi67/distorted_pts.txt")
d0 = points1[points1[:, 0] == 0][:, 1:]
d1 = points1[points1[:, 0] == 49][:, 1:]
d2 = points1[points1[:, 0] == 99][:, 1:]

cp = {0: c0, 49: c1, 99: c2}
dp = {0: d0, 49: d1, 99: d2}
solutions = {}
for i in list(cp.keys()):
    align = core.Alignment(cp[i], dp[i])
    align.get_solution((4, 4))
    solutions[i] = align.TPS_solution
solution_idx = np.array(list(solutions.keys()))
sol = solutions[0]
for i in range(100):
    if i in cp.keys():
        sol = solutions[i]
        print(i, sol[:, 0, 0])
    else:
        # Interpolate the solution (linearly)
        key_up = solution_idx[solution_idx > i][0]
        key_down = solution_idx[solution_idx < i][-1]
        sol_up = solutions[key_up]
        sol_down = solutions[key_down]
        sol = linear((key_down, sol_down), (key_up, sol_up), i)
        print(i, sol[:, 0, 0])
