import numpy as np
import numpy.linalg as nl
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist, squareform
import scipy
from scipy import misc


def makeL(cp):
    np.set_printoptions(linewidth=200)
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
    # calculate U(r) and make K in L
    # R = squareform(pdist(cp, metric='euclidean'))
    R = cdist(cp, cp, "euclidean")
    Rsq = R * R
    Rsq[R == 0] = 1  # avoid log(0) undefined, will correct itself as log(1) = 0, so U(0) = 0
    U = Rsq * np.log(Rsq)
    np.fill_diagonal(U, 0)  # should be redundant
    L[:K, :K] = U
    return L


# put source control (reference) points in here, must be paired
# use same spatial resolution as EBSD data
bse = np.loadtxt("ctr_pts_bse_resized.txt")
xs = bse[:, 0]
ys = bse[:, 1]
# xs = [-1,0,1,-1,0,1,-1,0,1]
# ys = [-1,-1,-1,0,0,0,1,1,1]

xs = np.asarray(xs)
ys = np.array(ys)
cps = np.vstack([xs, ys]).T

# put EBSD points here corresponding to same reference points
ebsd = np.loadtxt("ctr_pts_ebsd.txt")
xt = ebsd[:, 0]
yt = ebsd[:, 1]
# xt = [-1.04679684, 0.26460071, 1.20070563, -1.2005926, -0.25850049, 1.02804064, -0.82763901, -0.29600121, 0.79250861]
# yt = [-0.7820908, -0.9433111, -0.91892397, -0.21130865, 0.27281083, -0.06648004, 0.74353688, 1.29009827, 1.19395588]
xt = np.asarray(xt)
yt = np.array(yt)

n = len(xt)
np.set_printoptions(linewidth=200)

# construct L
L = makeL(cps)

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

# # verifies that functional has square integrable second derivatives. Print outs should be zero or basically zero
# wShiftX = params[:n,0]
# wShiftY = params[:n,1]
# print("Sum Wi should be zero and is: {}".format(np.sum(wi)))
# print("Sum Wi*xi should be zero and is: {}".format(np.dot(wShiftX,xs)))
# print("Sum Wi*yi should be zdro and is: {}".format(np.dot(wShiftY,ys)))

# TPS function here
# at some point (x,y) in reference, the corresponding point in the EBSD data is at
# [X,Y] = a1 + ax*xRef + ay*yRef + sum(wi*Ui)

# (x,y) coordinates from reference image
N = 10  # numpoints
nx = 11
ny = 8
x = np.linspace(0, 1100, nx)
y = np.linspace(0, 800, ny)
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
# print R.shape #shape of R is pixels, reference pairs
# print R #prints R(px1-CP1) (px1-CP2) (px1-CP3)....
Rsq = R * R
Rsq[R == 0] = 1  # avoid log(0) undefined, will correct itself as log(1) = 0, so U(0) = 0
U = Rsq * np.log(Rsq)
bend = np.einsum("ij,jk->ik", U, wi).T
bend = np.reshape(bend, (2, ny, nx))

# add together portions
sol = affine + bend
print(sol.shape, xgd.shape, ygd.shape)
dx = sol[0] - xgd
dy = sol[1] - ygd
mag = np.sqrt(dx ** 2 + dy ** 2)
cmap = plt.get_cmap()

plt.figure(1)
plt.scatter(xgd, ygd, c="b", s=5)
plt.scatter(xs, ys, marker="+", c="r", s=100)
plt.title("BSE ctr pts and target grid")

plt.figure(2)
plt.scatter(sol[0], sol[1], c="r", s=5)
plt.scatter(xt, yt, marker="+", c="b", s=100)
plt.title("EBSD ctr pts and resulting grid")

plt.figure(3)
plt.scatter(xgd, ygd, c="k", s=5)
plt.scatter(xs, ys, marker="+", c="r", s=100, label="BSE ctr pts")
plt.scatter(xt, yt, marker="+", c="b", s=100, label="EBSD ctr pts")
plt.quiver(xgd, ygd, dx, dy, mag, cmap=plt.get_cmap("magma"))
plt.title("Deformation field")
plt.colorbar()
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1))
plt.tight_layout()
plt.show()
