# Author: James Lamb

import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py
from rich.progress import track
from rich import print
import imageio
from skimage import io, transform
import numpy.linalg as nl
from scipy.spatial.distance import cdist

np.set_printoptions(linewidth=200)


class SelectCoords:
    def __init__(self, name, ctr_name=None, return_path=False) -> None:
        self.name = name[: name.index(".")]
        if ctr_name is None:
            self.txt_path = f"ctr_pts_{self.name}.txt"
        self.im_path = name
        self.clean_txt_file()
        self.im = io.imread(self.im_path)
        self.get_coords()
        if return_path:
            self.return_paths()

    def get_coords(self) -> None:
        self.fig1 = plt.figure(1, figsize=(12, 8))
        ax1 = self.fig1.add_subplot(111)
        ax1.imshow(self.im)
        self.cid1 = self.fig1.canvas.mpl_connect("button_press_event", self.onclick)
        self.qid1 = self.fig1.canvas.mpl_connect("close_event", self.close)
        plt.show()

    def onclick(self, event) -> None:
        ix, iy = event.xdata, event.ydata
        x = np.around(ix, 0).astype(np.uint32)
        y = np.around(iy, 0).astype(np.uint32)
        with open(self.txt_path, "a", encoding="utf8") as output:
            output.write(f"{x} {y}\n")
        print(np.loadtxt(self.txt_path, delimiter=" ")[-1])

    def close(self, event) -> None:
        self.fig1.canvas.mpl_disconnect(self.cid1)
        self.fig1.canvas.mpl_disconnect(self.qid1)
        plt.close(1)
        self.draw_points()

    def draw_points(self) -> None:
        pts = np.loadtxt(self.txt_path, delimiter=" ")
        fig = plt.figure(2)
        ax = fig.add_subplot(111)
        ax.imshow(self.im)
        for i in range(pts.shape[0]):
            ax.scatter(pts[i, 0], pts[i, 1], c="r", s=1)
            ax.text(pts[i, 0] + 2, pts[i, 1] + 2, i)
        fig.savefig(f"{str(self.name)}_points.png")
        plt.close()
        print(f"Points drawn on image saved to [blue]{self.name}_points.png")

    def clean_txt_file(self, path=None) -> None:
        if path is None:
            path = self.txt_path
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    def return_paths(self) -> str:
        return str(self.txt_path)


class Alignment:
    def __init__(self, referencePoints, distortedPoints) -> None:
        self.referencePoints = referencePoints
        self.distortedPoints = distortedPoints

    def TPS(
        self,
        referenceImage,
        affineOnly=False,
        checkParams=True,
        saveParams=False,
        saveSolution=False,
    ) -> None:
        TPS_Params = "TPS_params.csv"
        solutionFile = "TPS_mapping.npy"
        source = np.loadtxt(self.referencePoints, delimiter=" ")
        xs = source[:, 0]
        ys = source[:, 1]
        distorted = np.loadtxt(self.distortedPoints, delimiter=" ")
        xt = distorted[:, 0]
        yt = distorted[:, 1]
        # check to make sure each control point is paired
        if len(xs) == len(ys) and len(xt) == len(yt) and len(xs) == len(ys):
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
        L = self.makeL(cps)

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

    def TPS_apply(self, im_array, save_name="TPS_out.tif") -> None:
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
        imageio.imsave(save_name, imageArray)
        print(imageArray.shape)
        print("Corrected image save to {}\n".format(save_name))

    def makeL(self, cp):
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


### Functions ###


def resize_imgs(bse_path, size) -> None:
    if "." in bse_path:
        basename = bse_path[: bse_path.index(".")]
        ext = bse_path[bse_path.index(".") :]
        im = io.imread(bse_path)
        resized = transform.resize(im, size, anti_aliasing=True)
        matplotlib.image.imsave(basename + "_resized" + ext, resized, cmap="gray", dpi=1)
        i = 0
    else:
        bse_path = os.listdir(bse_path)
        ext = bse_path[0][bse_path.index(".") :]
        basename = [path[: path.index(".")] for path in bse_path]
        for i in range(len(bse_path)):
            im = io.imread(bse_path[i])
            resized = transform.resize(im, size, anti_aliasing=True)
            matplotlib.image.imsave(basename[i] + "_resized" + ext, resized, cmap="gray", dpi=1)
    print("Resized {} images (ext: [green]{}[/], size: [magenta]{}[/])\n".format(i + 1, ext, size))


def h5_to_img(h5_path, slice_id, fname, view="Confidence Index", axis=0, format=None) -> None:
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
