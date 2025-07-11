import numpy as np
from scipy.spatial.distance import cdist
from skimage import transform as tf


class ThinPlateSplineTransform:
    def __init__(self, affine_only=False):
        self._estimated = False
        self.params = None
        self.affine_only = affine_only

    def __call__(self, coords):
        """Transform coordinates from source to destination using thin plate spline."""
        if not self._estimated:
            raise ValueError("Transformation not estimated.")
        params = np.moveaxis(self.params, 0, -1)
        coords = np.asarray(coords).astype(int)
        out = params[(coords[:, 1], coords[:, 0])]
        return out

    def estimate(self, src, dst, size):
        """Estimate optimal spline mappings between source and destination points.

        Parameters
        ----------
        src : (N, 2) array_like
            Control points at source coordinates.
        dst : (N, 2) array_like
            Control points at destination coordinates.
        size : tuple
            Size of the reference image (height, width).

        Returns
        -------
        success: bool
            True indicates that the estimation was successful.

        Notes
        -----
        The number N of source and destination points must match.
        """
        # convert input pixels in arrays. cps are control points
        xs = np.asarray(dst[:, 0])
        ys = np.array(dst[:, 1])
        cps = np.vstack([xs, ys]).T
        xt = np.asarray(src[:, 0])
        yt = np.array(src[:, 1])
        n = len(xs)
        # print("Number of control points:", n)

        # construct L
        L = self._TPS_makeL(cps)

        # construct Y
        xtAug = np.concatenate([xt, np.zeros(3)])
        ytAug = np.concatenate([yt, np.zeros(3)])
        Y = np.vstack([xtAug, ytAug]).T

        # calculate unknown params in (W | a).T
        params = np.dot(np.linalg.inv(L), Y)
        wi = params[:n, :]
        a1 = params[n, :]
        ax = params[n + 1, :]
        ay = params[n + 2, :]

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
        del xgd, ygd, x, y

        # bending portion, but do it in parts for memory reasons
        bend = np.zeros((2, nx * ny))
        for i in range(n):
            # print("Calculating bending portion for control point {}...".format(i))
            R = np.linalg.norm(pixels - cps[i], axis=1).reshape(-1, 1)
            Rsq = R * R
            Rsq[R == 0] = 1
            U = Rsq * np.log(Rsq)
            bend += np.einsum("ij,jk->ik", U, wi[i].reshape(1, 2)).T
        bend = np.reshape(bend, (2, ny, nx))

        # print("Bending portion calculated...")
        if self.affine_only:
            self.params = affine
        else:
            self.params = affine + bend
        self.size = size
        self._estimated = True
        return True

    def _TPS_makeL(self, cp):
        """Function to make the L matrix for thin plate spline calculation."""
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
        Rsq[R == 0] = (
            1  # avoid log(0) undefined, will correct itself as log(1) = 0, so U(0) = 0
        )
        U = Rsq * np.log(Rsq)
        np.fill_diagonal(U, 0)  # should be redundant
        L[:K, :K] = U
        return L


if __name__ == "__main__":
    from core import Alignment
    from skimage import data
    import matplotlib.pyplot as plt

    image = data.checkerboard()
    src = np.loadtxt("src.txt", delimiter=",", dtype=int)[8:, 1:].astype(int)
    dst = np.loadtxt("dst.txt", delimiter=",", dtype=int)[8:, 1:].astype(int)

    tform = ThinPlateSplineTransform()
    tform.estimate(src, dst, image.shape)

    image_warped = tf.warp(image, tform)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image, cmap="gray")
    ax[0].plot(dst[:, 0], dst[:, 1], "ro")
    ax[1].imshow(image_warped, cmap="gray")
    ax[1].plot(src[:, 0], src[:, 1], "ro")
    ax[0].set_xlim(0, image.shape[1])
    ax[0].set_ylim(image.shape[0], 0)
    ax[1].set_xlim(0, image.shape[1])
    ax[1].set_ylim(image.shape[0], 0)
    plt.show()
