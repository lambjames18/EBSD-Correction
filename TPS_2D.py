# ***********************************************************************************
# * Copyright 2019 Andrew T. Polonsky. All rights reserved.                         *
# *                                                                                 *
# * Redistribution and use in source and binary forms, with or without              *
# * modification, are permitted provided that the following conditions are met:     *
# *                                                                                 *
# *  1. Redistributions of source code must retain the above copyright notice,      *
# *  this list of conditions and the following disclaimer.                          *
# *                                                                                 *
# *  2. Redistributions in binary form must reproduce the above copyright notice,   *
# *  this list of conditions and the following disclaimer in the documentation      *
# *  and/or other materials provided with the distribution.                         *
# *                                                                                 *
# * 3. Neither the name of the copyright holder nor the names of its                *
# *    contributors may be used to endorse or promote products derived from         *
# *    this software without specific prior written permission.                     *
# *                                                                                 *
# * THIS SOFTWARE IS PROVIDED BY ANDREW T. POLONSKY ``AS IS'' AND ANY EXPRESS OR    *
# * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF    *
# * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO      *
# * EVENT SHALL <COPYRIGHT HOLDER> OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,        *
# * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,  *
# * BUT NOT LIMITED TO, PROCUREMEN OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,    *
# * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY           *
# * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING  *
# * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS              *
# * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                    *
# ***********************************************************************************

"""
This program takes two images as input and performs the thin plate spline algorithm
to map the distorted image into the reference frame of the reference image using
control points selected by the user.
Details of the calculation of the thin plate spline can be found in the paper,
"A method to correct coordinate distortion in EBSD maps" by Zhang et. al, in 
Materials Characterization, 2014: http://dx.doi.org/10.1016/j.matchar.2014.08.003
"""

import numpy as np
import numpy.linalg as nl
from scipy.spatial.distance import cdist
import imageio
import matplotlib.pyplot as plt

# # #
# #
# # #

### BEGIN USER INPUT AREA

affineOnly = False  # whether to include bending portion
checkParams = True  # ensure TPS algorithm finds valid solution
saveParams = False  # TPS parameters
saveMap = True  # pointwise mapping matrix, a larger file

# File path inputs
referenceImage = "8_bse_resized.png"
distortedImage = "8_ebsd.png"
#referenceImage = "coni16_459.tif"
#distortedImage = "coni16_459_eds.tif"

# File path outputs
TPS_Params = "params.csv"
correctedImage = "TPS_out.tif"
solutionFile = "pointWise_mapping.npy"  # will be a 3D array, with x,y mappings at each X,Y location

# put source (reference) control points in here, must be paired, in pixel coordinates
source = np.loadtxt("ctr_pts_8_bse_resized.txt", delimiter=" ")
source = np.loadtxt("ctr_pts_coni16_459.txt", delimiter=" ")
xs = source[:, 0]
ys = source[:, 1]
# xs = [45,  43, 488, 619, 451]
# ys = [66, 472, 474, 293,  73]

# corresponding control points in image to transform, in pixel coordinates
distorted = np.loadtxt("ctr_pts_8_ebsd.txt", delimiter=" ")
distorted = np.loadtxt("ctr_pts_coni16_459_eds.txt", delimiter=" ")
xt = distorted[:, 0]
yt = distorted[:, 1]
# xt = [39,  13, 466, 608, 448]
# yt = [39, 470, 431, 225,   6]

### END USER INPUT AREA

# # #
# #
# # #


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
    R = cdist(cp, cp, "euclidean")
    Rsq = R * R
    Rsq[R == 0] = 1  # avoid log(0) undefined, will correct itself as log(1) = 0, so U(0) = 0
    U = Rsq * np.log(Rsq)
    np.fill_diagonal(U, 0)  # should be redundant
    L[:K, :K] = U
    return L


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

# # #
# #
# # #

# Thin plate spline calculation
# at some point (x,y) in reference, the corresponding point in the distorted data is at
# [X,Y] = a1 + ax*xRef + ay*yRef + sum(wi*Ui)

print("Reading images...\n")
##import images
a = imageio.imread(referenceImage)
print(a.shape)
b = imageio.imread(distortedImage)
print(b.shape)

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

if saveMap:
    np.save(solutionFile, sol)
    print("Point-wise solution save to {}\n".format(solutionFile))

# # #
# #
# # #

# get locations in original image to place back into the created grid
# sol[0] are the corresponding x-coordinates in the distorted image
# sol[1] are the corresponding y-coorindates in the distorted image
xgtId = np.around(sol[0])  # round to nearest pixel
xgtId = xgtId.astype(int)
xgtId = xgtId.flatten()
ygtId = np.around(sol[1])  # round to nearest pixe
ygtId = ygtId.astype(int)
ygtId = ygtId.flatten()

# determine which pixels actually lie within the distorted image
validX = (xgtId < b.shape[1]) * (xgtId > 0)
validY = (ygtId < b.shape[0]) * (ygtId > 0)
valid = validX * validY

# get data from distorted image at apporpiate locations, make any non-valid points = 0
c = b[validY * ygtId, validX * xgtId]
print(c.shape)
print((ny, nx))
c = c * valid

imageArray = np.reshape(c, (ny, nx))

imageio.imsave(correctedImage, imageArray)
print(imageArray.shape)

print("Corrected image save to {}\n".format(correctedImage))

#plt.imshow(b)
#plt.show()
