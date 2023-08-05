# -*- coding: utf-8 -*-

"""Author: James Lamb"""

import os

import numpy as np
import matplotlib
import h5py
import imageio
from skimage import io
from skimage import transform as tf

import numpy.linalg as nl
from scipy.spatial.distance import cdist
from scipy import interpolate


class Alignment:
    def __init__(self, referencePoints, distortedPoints, algorithm="TPS"):
        self.source = referencePoints
        self.distorted = distortedPoints
        if algorithm.upper() == "TPS":
            self.get_solution = self.TPS
            self.apply = self.TPS_apply
            self.import_solution = self.TPS_import
        elif algorithm.upper() == "LR":
            self.get_solution = self.LR
            self.apply = self.LR_apply
            self.import_solution = self.LR_import
        else:
            raise IOError(
                "Arg :algorithm: must be either 'LR' for linear regression or 'TPS' for Thin Plate Spline."
            )

    def TPS(
        self,
        size,
        affineOnly=False,
        checkParams=False,
        saveParams=False,
        saveSolution=True,
        solutionFile="TPS_mapping.npy",
        verbose=True,
    ):
        TPS_Params = "TPS_params.csv"
        # solutionFile = "TPS_mapping.npy"
        # source = np.loadtxt(self.referencePoints, delimiter=" ")
        source = self.source
        xs = source[:, 0]
        ys = source[:, 1]
        # distorted = np.loadtxt(self.distortedPoints, delimiter=" ")
        distorted = self.distorted
        xt = distorted[:, 0]
        yt = distorted[:, 1]
        # check to make sure each control point is paired
        if len(xs) == len(ys) and len(xt) == len(yt) and len(xt) == len(yt):
            n = len(xs)
            if verbose:
                print("Given {} points...".format(n))
        else:
            raise ValueError(
                f"Control point arrays are not of equal length: xs {xs.shape}, ys {ys.shape}, xt {xt.shape}, yt {yt.shape}"
            )

        # convert input pixels in arrays. cps are control points
        xs = np.asarray(xs)
        ys = np.array(ys)
        cps = np.vstack([xs, ys]).T
        xt = np.asarray(xt)
        yt = np.array(yt)

        # construct L
        L = self._TPS_makeL(cps)

        # construct Y
        xtAug = np.concatenate([xt, np.zeros(3)])
        ytAug = np.concatenate([yt, np.zeros(3)])
        Y = np.vstack([xtAug, ytAug]).T

        # calculate unknown params in (W | a).T
        params = np.dot(nl.inv(L), Y)
        wi = params[:n, :]
        a1 = params[n, :]
        ax = params[n + 1, :]
        ay = params[n + 2, :]

        # print("TPS parameters found\n")
        header = "TPS Parameters given as x y pairs (x value on row 3 and y value on row 4)\n"
        for i in range(0, n):
            header = header + "w{}, ".format(i + 1)
        header = header + "a1, ax, ay "

        if saveParams:
            np.savetxt(TPS_Params, params.T, delimiter=",", header=header)
            if verbose:
                print("Parameters saved to {}\n".format(TPS_Params))

        # verifies that functional has square integrable second derivatives. Print outs should be zero or basically zero
        wShiftX = params[:n, 0]
        wShiftY = params[:n, 1]
        if checkParams:
            print("Checking if Thin Plate Spline parameters are valid:")
            print("\tSum   Wi  should be 0 and is: {:1.2e}".format(np.sum(wi)))
            print("\tSum Wi*xi should be 0 and is: {:1.2e}".format(np.dot(wShiftX, xs)))
            print("\tSum Wi*yi should be 0 and is: {:1.2e}\n".format(np.dot(wShiftY, ys)))

        # Thin plate spline calculation
        # at some point (x,y) in reference, the corresponding point in the distorted data is at
        # [X,Y] = a1 + ax*xRef + ay*yRef + sum(wi*Ui)
        # dimensions of reference image in pixels
        lx = size[1]
        ly = size[0]

        # for fineness of grid, if you want to fix all points, leave nx=lx, ny=ly
        nx = lx  # num points along reference x-direction, full correction will have nx = lx
        ny = ly  # num points along reference y-direction, full correction will have ny = ly

        # (x,y) coordinates from reference image
        x = np.linspace(1, lx, nx)
        y = np.linspace(1, ly, ny)
        xgd, ygd = np.meshgrid(x, y)
        pixels = np.vstack([xgd.flatten(), ygd.flatten()]).T

        # affine transformation portion
        axs = np.einsum("i,jk->ijk", ax, xgd)
        ays = np.einsum("i,jk->ijk", ay, ygd)
        affine = axs + ays
        affine[0, :, :] += a1[0]
        affine[1, :, :] += a1[1]

        # bending portion
        R = cdist(pixels, cps, "euclidean")  # are nx*ny pixels, cps = num reference pairs
        Rsq = R * R
        Rsq[R == 0] = 1  # avoid log(0) undefined, will correct itself as log(1) = 0, so U(0) = 0
        U = Rsq * np.log(Rsq)
        bend = np.einsum("ij,jk->ik", U, wi).T
        bend = np.reshape(bend, (2, ny, nx))

        # add together portions
        if affineOnly:
            sol = affine
        else:
            sol = affine + bend

        self.TPS_solution = sol
        self.TPS_grid_spacing = (ny, nx)

        if saveSolution:
            np.save(solutionFile, sol)
            if verbose:
                print("Point-wise solution save to {}\n".format(solutionFile))

    def TPS_apply(self, im_array, save_name="TPS_out.tif", out="image"):
        if len(im_array.shape) > 2:
            im_array = im_array[:, :, 0]
        # get locations in original image to place back into the created grid
        # sol[0] are the corresponding x-coordinates in the distorted image
        # sol[1] are the corresponding y-coorindates in the distorted image
        xgtId = np.around(self.TPS_solution[0])  # round to nearest pixel
        xgtId = xgtId.astype(int)
        xgtId = xgtId.flatten()
        ygtId = np.around(self.TPS_solution[1])  # round to nearest pixel
        ygtId = ygtId.astype(int)
        ygtId = ygtId.flatten()

        # determine which pixels actually lie within the distorted image
        validX = (xgtId < im_array.shape[1]) * (xgtId > 0)
        validY = (ygtId < im_array.shape[0]) * (ygtId > 0)
        valid = validX * validY

        # get data from distorted image at appropriate locations, make any non-valid points = 0
        c = im_array[validY * ygtId, validX * xgtId]
        c = c * valid

        imageArray = np.reshape(c, self.TPS_grid_spacing)
        if out == "image":
            imageio.imsave(save_name, imageArray)
            print("Corrected image save to {}\n".format(save_name))
        else:
            return imageArray

    def TPS_import(self, sol_path, referenceImage):
        a = imageio.imread(referenceImage)
        nx = a.shape[1]
        ny = a.shape[0]
        self.TPS_grid_spacing = (ny, nx)
        self.TPS_solution = np.load(sol_path)

    def TPS_apply_3D(self, points, dataset, bse):
        # Get slice numbers
        slice_numbers = list(points["ebsd"].keys())
        if len(slice_numbers) != len(list(points["bse"].keys())):
            raise RuntimeError("Number of slices with reference points in control and distorted do not match")
        # Get the solution for each set of control points
        params = {}
        if len(slice_numbers) == 1:
            print("Only one slice has control points, only calculating one solution.")
            key = slice_numbers[0]
            self.source = np.array(points["bse"][key])
            self.distorted = np.array(points["ebsd"][key])
            self.TPS(bse.shape[1:], saveSolution=False, verbose=False)
            # Create transform object for each slice and warp
            aligned_dataset = np.zeros(bse.shape, dtype=dataset.dtype)
            for i in range(dataset.shape[0]):
                aligned_dataset[i] = self.TPS_apply(dataset[i], out="array")
            return aligned_dataset
        else:
            for key in slice_numbers:
                self.source = np.array(points["bse"][key])
                self.distorted = np.array(points["ebsd"][key])
                self.TPS(bse.shape[1:], saveSolution=False, verbose=False)
                params[key] = self.TPS_solution
            # Interpolate the solutions
            interpolations = {}
            for i in range(len(slice_numbers) - 1):
                f = interpolate.interp1d([slice_numbers[i], slice_numbers[i + 1]], [params[slice_numbers[i]], params[slice_numbers[i + 1]]], axis=0)
                interpolations[f"{slice_numbers[i]} {slice_numbers[i + 1]}"] = {index: f(index) for index in range(slice_numbers[i] + 1, slice_numbers[i + 1])}
            solutions = np.zeros((bse.shape[0], * self.TPS_solution.shape))
            # Get the solution for each slice
            for i in range(solutions.shape[0]):
                found_match = False
                if i in slice_numbers:
                    solutions[i] = params[i]
                    found_match = True
                    continue
                elif i not in slice_numbers:
                    max_lower = 0
                    min_upper = solutions.shape[0]
                    for j in range(len(slice_numbers) - 1):
                        if slice_numbers[j] < i < slice_numbers[j + 1]:
                            max_lower = slice_numbers[j]
                            min_upper = slice_numbers[j + 1]
                            solutions[i] = interpolations[f"{max_lower} {min_upper}"][i]
                            found_match = True
                    if not found_match:
                        # Copy the closest slice
                        if i < slice_numbers[0]:
                            solutions[i] = params[0]
                        elif i > slice_numbers[-1]:
                            solutions[i] = params[max(list(params.keys()))]
                            
                else:
                    raise RuntimeError("Something went wrong while generating solutions")
            # Create transform object for each slice and warp
            aligned_dataset = np.zeros(bse.shape, dtype=dataset.dtype)
            for i in range(dataset.shape[0]):
                sol = solutions[i]
                self.TPS_solution = sol
                aligned_dataset[i] = self.TPS_apply(dataset[i], out="array")
            return aligned_dataset

    def _TPS_makeL(self, cp):
        # cp: [K x 2] control points
        # L: [(K+3) x (K+3)]
        K = cp.shape[0]
        L = np.zeros((K + 3, K + 3))
        # make P in L
        L[:K, K] = 1
        L[:K, K + 1: K + 3] = cp
        # make P.T in L
        L[K, :K] = 1
        L[K + 1:, :K] = cp.T
        R = cdist(cp, cp, "euclidean")
        Rsq = R * R
        Rsq[R == 0] = 1  # avoid log(0) undefined, will correct itself as log(1) = 0, so U(0) = 0
        U = Rsq * np.log(Rsq)
        np.fill_diagonal(U, 0)  # should be redundant
        L[:K, :K] = U
        return L

# Functions #
def resize_imgs(bse_path, size):
    if "." in bse_path:
        basename = bse_path[: bse_path.index(".")]
        ext = bse_path[bse_path.index("."):]
        im = io.imread(bse_path)
        resized = tf.resize(im, size, anti_aliasing=True)
        matplotlib.image.imsave(basename + "_resized" + ext, resized, cmap="gray", dpi=1)
        i = 0
    else:
        bse_path = os.listdir(bse_path)
        ext = bse_path[0][bse_path.index(".")]
        basename = [path[: path.index(".")] for path in bse_path]
        for i in range(len(bse_path)):
            im = io.imread(bse_path[i])
            resized = tf.resize(im, size, anti_aliasing=True)
            matplotlib.image.imsave(basename[i] + "_resized" + ext, resized, cmap="gray", dpi=1)
    print("Resized {} images (ext: {}, size: {})\n".format(i + 1, ext, size))


def h5_to_img(h5_path, slice_id, fname, view="Confidence Index", axis=0, format=None):
    h5 = h5py.File(h5_path, "r")
    data = h5[f"DataContainers/ImageDataContainer/CellData/{view}"][slice_id]
    if format is None:
        im = data[:, :, 0]
    if format == "mag":
        im = np.sqrt(np.sum(np.square(data), axis=2))
    if format == "sum":
        im = np.sum(data, axis=2)
    io.imsave(fname, im)
    print("Saved the {} image to {}".format(view, fname))

def handle_dtype(data, dtype):
    if dtype == np.uint8:
        data = np.around(255 * (data - np.min(data)) / (np.max(data) - np.min(data))).astype(dtype)
    elif dtype == np.uint16:
        data = np.around(65535 * (data - np.min(data)) / (np.max(data) - np.min(data))).astype(dtype)
    elif dtype == np.uint32:
        data = np.around(4294967295 * (data - np.min(data)) / (np.max(data) - np.min(data))).astype(dtype)
    elif dtype == np.uint64:
        data = np.around(18446744073709551615 * (data - np.min(data)) / (np.max(data) - np.min(data))).astype(dtype)
    elif dtype == np.int8:
        data = np.around(255 * (data - np.min(data)) / (np.max(data) - np.min(data)) - 128).astype(dtype)
    elif dtype == np.int16:
        data = np.around(65535 * (data - np.min(data)) / (np.max(data) - np.min(data)) - 32768).astype(dtype)
    elif dtype == np.int32:
        data = np.around(4294967295 * (data - np.min(data)) / (np.max(data) - np.min(data)) - 2147483648).astype(dtype)
    elif dtype == np.int64:
        data = np.around(18446744073709551615 * (data - np.min(data)) / (np.max(data) - np.min(data)) - 9223372036854775808).astype(dtype)
    elif dtype == np.float32 or dtype == np.float64 or dtype == np.float128 or dtype == np.float16:
        data = data.astype(np.float32)
    elif dtype == bool:
        data = (data != 0).astype(dtype)
    else:
        raise RuntimeError("Unknown dtype")
    return data
