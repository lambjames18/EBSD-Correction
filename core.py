# -*- coding: utf-8 -*-

"""Author: James Lamb"""

import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py
from rich.progress import track
from rich import print
import imageio
from skimage import io
from skimage import transform as tf

# TPS Stuff
import numpy.linalg as nl
from scipy.spatial.distance import cdist

# LR Stuff
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

np.set_printoptions(linewidth=200)

# Matplotlib window that records clicked locations and stores them in a text file
# Prints the point number and the (x,y) of the point on each click
# Designed to be ran twice on the bse and the ebsd
class SelectCoords:
    def __init__(self, name, save_folder="", ext="tif"):
        self.save_folder = save_folder
        self.name = name
        self.txt_path = f"{self.save_folder}ctr_pts_{self.name}.txt"
        self.im_path = f"{self.save_folder}{self.name}.{ext}"
        self.check_txt_file()
        self.im = io.imread(self.im_path)
        self.get_coords()

    def get_coords(self):
        self.fig1 = plt.figure(1, figsize=(12, 8))
        ax1 = self.fig1.add_subplot(111)
        ax1.imshow(self.im)
        self.cid1 = self.fig1.canvas.mpl_connect("button_press_event", self.onclick)
        self.qid1 = self.fig1.canvas.mpl_connect("close_event", self.close)
        plt.show()

    def onclick(self, event):
        ix, iy = event.xdata, event.ydata
        x = np.around(ix, 0).astype(np.uint32)
        y = np.around(iy, 0).astype(np.uint32)
        with open(self.txt_path, "a", encoding="utf8") as output:
            output.write(f"{x} {y}\n")
        points = np.loadtxt(self.txt_path, delimiter=" ")
        if len(points.shape) < 2:
            points = [points]
        print("Point #{} -> {}".format(len(points), tuple(points[-1].astype(int))))

    def close(self, event):
        self.fig1.canvas.mpl_disconnect(self.cid1)
        self.fig1.canvas.mpl_disconnect(self.qid1)
        plt.close(1)
        self.draw_points()

    def draw_points(self):
        pts = np.loadtxt(self.txt_path, delimiter=" ")
        fig = plt.figure(2)
        ax = fig.add_subplot(111)
        ax.imshow(self.im)
        for i in range(pts.shape[0]):
            ax.scatter(pts[i, 0], pts[i, 1], c="r", s=1)
            ax.text(pts[i, 0] + 2, pts[i, 1] + 2, i)
        fig.savefig(f"{self.save_folder}{str(self.name)}_points.png")
        plt.close()
        print(f"Points drawn on image saved to [blue]{self.save_folder}{self.name}_points.png")

    def check_txt_file(self):
        mode = input("Clear old points? (y/n) ")
        if mode == "y":
            try:
                os.remove(self.txt_path)
            except FileNotFoundError:
                pass
            with open(self.txt_path, "w", encoding="utf8") as o:
                pass


class Alignment:
    def __init__(self, referencePoints, distortedPoints, algorithm="TPS"):
        self.referencePoints = referencePoints
        self.distortedPoints = distortedPoints
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
        referenceImage,
        affineOnly=False,
        checkParams=True,
        saveParams=False,
        saveSolution=False,
        solutionFile="TPS_mapping.npy",
    ):
        TPS_Params = "TPS_params.csv"
        # solutionFile = "TPS_mapping.npy"
        source = np.loadtxt(self.referencePoints, delimiter=" ")
        xs = source[:, 0]
        ys = source[:, 1]
        distorted = np.loadtxt(self.distortedPoints, delimiter=" ")
        xt = distorted[:, 0]
        yt = distorted[:, 1]
        # check to make sure each control point is paired
        if len(xs) == len(ys) and len(xt) == len(yt):
            n = len(xs)
            print("Given {} points...".format(n))
        else:
            raise ValueError("Control point arrays are not of equal length")

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

        print("TPS parameters found\n")
        header = "TPS Parameters given as x y pairs (x value on row 3 and y value on row 4)\n"
        for i in range(0, n):
            header = header + "w{}, ".format(i + 1)
        header = header + "a1, ax, ay "

        if saveParams:
            np.savetxt(TPS_Params, params.T, delimiter=",", header=header)
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
        a = imageio.imread(referenceImage)

        # dimensions of reference image in pixels
        lx = a.shape[1]
        ly = a.shape[0]

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
        if saveSolution:
            np.save(solutionFile, sol)
            print("Point-wise solution save to {}\n".format(solutionFile))

        self.TPS_solution = sol
        self.TPS_grid_spacing = (ny, nx)

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

        # get data from distorted image at apporpiate locations, make any non-valid points = 0
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

    def _TPS_makeL(self, cp):
        # cp: [K x 2] control points
        # L: [(K+3) x (K+3)]
        K = cp.shape[0]
        L = np.zeros((K + 3, K + 3))
        # make P in L
        L[:K, K] = 1
        L[:K, K + 1 : K + 3] = cp
        # make P.T in L
        L[K, :K] = 1
        L[K + 1 :, :K] = cp.T
        R = cdist(cp, cp, "euclidean")
        Rsq = R * R
        Rsq[R == 0] = 1  # avoid log(0) undefined, will correct itself as log(1) = 0, so U(0) = 0
        U = Rsq * np.log(Rsq)
        np.fill_diagonal(U, 0)  # should be redundant
        L[:K, :K] = U
        return L

    def LR(self, degree=3, saveSolution=False, solutionFile="LR_mapping.npy"):
        # Read in the source/distorted points
        coord_ebsd = np.loadtxt(open(self.distortedPoints, "rb"), delimiter=" ").astype(int)
        coord_bse = np.loadtxt(open(self.referencePoints, "rb"), delimiter=" ").astype(int)

        # check to make sure each control point is paired
        if len(coord_ebsd) == len(coord_bse):
            print("Given {} points...".format(coord_bse.shape[0]))
        else:
            raise ValueError("Control point arrays are not of equal length")

        # check
        dist_s = self._LR_successiveDistances(coord_ebsd)
        dist_t = self._LR_successiveDistances(coord_bse)
        ratio = self._LR_findRatio(dist_t, dist_s)

        # Convert control point coords into same reference frame
        coord_bse_rescaled = coord_bse / ratio
        src = coord_bse_rescaled
        dst = coord_ebsd

        # Define polymomial regression
        model_i = Pipeline(
            [
                ("poly", PolynomialFeatures(degree=degree, include_bias=True)),
                ("linear", LinearRegression(fit_intercept=False, normalize=False)),
            ]
        )

        model_j = Pipeline(
            [
                ("poly", PolynomialFeatures(degree=degree, include_bias=True)),
                ("linear", LinearRegression(fit_intercept=False, normalize=False)),
            ]
        )

        # Solve regression
        model_i.fit(src, dst[:, 0])
        model_j.fit(src, dst[:, 1])

        # Define the image transformation
        self.params = np.stack(
            [model_i.named_steps["linear"].coef_, model_j.named_steps["linear"].coef_], axis=0
        )
        if saveSolution:
            np.save(solutionFile, self.params)
            print("Point-wise solution save to {}\n".format(solutionFile))
        # Get transform
        self.transform = tf._geometric.PolynomialTransform(self.params)

    def LR_apply(self, im_array, save_name="LR_out.tif", out="image"):
        if len(im_array.shape) > 2:
            im_array = im_array[:, :, 0]
        # Read in distorted image
        imageArray = tf.warp(
            im_array,
            self.transform,
            cval=0,  # new pixels are black pixel
            order=0,  # k-neighbour
            preserve_range=True,
        )
        # Save corrected image
        if out == "image":
            imageio.imsave(save_name, imageArray)
            print("Corrected image save to {}\n".format(save_name))
        else:
            return imageArray

    def LR_import(self, sol_path):
        self.params = np.load(sol_path)
        self.transform = tf._geometric.PolynomialTransform(self.params)

    def _LR_successiveDistances(self, array):
        """Calculates euclidian distance btw sets of points in an array"""
        nb_of_points = int(np.ma.size(array) / 2)
        diffs = np.zeros(nb_of_points - 1, dtype=float)
        for i in range(0, nb_of_points - 1):
            diffs[i] = (array[i][0] - array[i + 1][0]) ** 2 + (array[i][1] - array[i + 1][1]) ** 2
        dists = np.sqrt(diffs, dtype=float)
        return dists

    def _LR_findRatio(self, array1, array2):
        """Finds scaling ratio btw 2 arrays of reference points on their own grids"""
        ratios = np.divide(array1, array2)
        ratio = np.average(ratios)
        return ratio


### Functions ###


def resize_imgs(bse_path, size):
    if "." in bse_path:
        basename = bse_path[: bse_path.index(".")]
        ext = bse_path[bse_path.index(".") :]
        im = io.imread(bse_path)
        resized = tf.resize(im, size, anti_aliasing=True)
        matplotlib.image.imsave(basename + "_resized" + ext, resized, cmap="gray", dpi=1)
        i = 0
    else:
        bse_path = os.listdir(bse_path)
        ext = bse_path[0][bse_path.index(".") :]
        basename = [path[: path.index(".")] for path in bse_path]
        for i in range(len(bse_path)):
            im = io.imread(bse_path[i])
            resized = tf.resize(im, size, anti_aliasing=True)
            matplotlib.image.imsave(basename[i] + "_resized" + ext, resized, cmap="gray", dpi=1)
    print("Resized {} images (ext: [green]{}[/], size: [magenta]{}[/])\n".format(i + 1, ext, size))


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
    print("Saved the [red]{}[/] image to [blue]{}[/]".format(view, fname))