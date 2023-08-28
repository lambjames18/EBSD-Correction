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


# Only one slice
# points = {"ebsd": {0: [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]}, "bse": {0: [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]}}

# All slices
# points = {"ebsd": {0: [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], 1: [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], 2: [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], 3: [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], 4: [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]}, "bse": {0: [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], 1: [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], 2: [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], 3: [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], 4: [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]}}

# Top and bottom slices
# points = {"ebsd": {0: [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], 4: [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]}, "bse": {0: [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], 4: [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]}}

# No top or bottom slices
points = {"ebsd": {1: [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], 3: [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]}, "bse": {1: [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], 3: [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]}}

dataset = np.random.randint(0, 100, (5, 10, 10))
bse = np.random.randint(0, 100, (5, 10, 10))

# Get slice numbers
slice_numbers = list(points["ebsd"].keys())
if len(slice_numbers) != len(list(points["bse"].keys())):
    raise RuntimeError("Number of slices with reference points in control and distorted do not match")

print("Number of slices with reference points: {}".format(len(slice_numbers)))
# Get the solution for each set of control points
solutions = {}
for key in slice_numbers:
    source = np.array(points["bse"][key])
    distorted = np.array(points["ebsd"][key])
    if source.shape != distorted.shape:
        raise RuntimeError("Number of control points in control and distorted do not match for slice {}".format(key))
    solutions[key] = np.random.randint(0, 100, (10, 10))
solution_keys = np.array(list(solutions.keys()))
print("Number of slices with solutions: {}".format(len(solution_keys)))
# Three cases: 1) all slices have control points, 2) a few slices have control points 3) only one slice has control points
# If 1, then just apply the solution to each slice
# If 3, then apply the solution to the entire dataset
# Versions of case 2: 1) Top and bottom slices are accounted for, 2) Top or bottom is accounted for, 3) Neither are accounted for
# If 2.1, then interpolate solutions between pairs of slices that have control points
# If 2.2, extend the lowest/highest solution to the top/bottom of the dataset, depending on which is missing
# If 2.3, same as 2.2 but extend the highest solution to the top and the lowest solution to the bottom
# Case 1
if dataset.shape[0] == len(solution_keys):
    # print("All slices have control points, applying solution to each slice.")
    aligned_dataset = np.zeros(bse.shape, dtype=dataset.dtype)
    for i in range(dataset.shape[0]):
        print("aligning slice {}, unique solution".format(i))
# Case 3
elif len(slice_numbers) == 1:
    # print("Only one slice has control points, applying solution to entire dataset.")
    key = slice_numbers[0]
    for i in range(dataset.shape[0]):
        print("aligning slice {}, all slices are using the same solution".format(i))
# Case 2
else:
    # Handle Case 2.2 and 2.3
    if 0 not in solution_keys:
        print("no slice 0 solution, extending lowest solution to top of dataset")
        solutions[0] = solutions[solution_keys[0]]
        solution_keys = np.insert(solution_keys, 0, 0)
    if dataset.shape[0] - 1 not in solution_keys:
        print("no slice {} solution, extending highest solution to bottom of dataset".format(dataset.shape[0] - 1))
        solutions[len(slice_numbers) - 1] = solutions[solution_keys[-1]]
        solution_keys = np.append(solution_keys, dataset.shape[0] - 1)
    
    print(solution_keys)
    # Treat like it is Case 2.1 now
    # print("Only a few slices have control points, interpolating solutions between slices.")
    aligned_dataset = np.zeros(bse.shape, dtype=dataset.dtype)
    for i in range(dataset.shape[0]):
        if i in solution_keys:
            print("aligning slice {}, slice has a unique solution".format(i))
        else:
            key_up = solution_keys[solution_keys > i][0]
            key_down = solution_keys[solution_keys < i][-1]
            sol_up = solutions[key_up]
            sol_down = solutions[key_down]
            print("aligning slice {}, interpolating between slices {} and {}".format(i, key_up, key_down))