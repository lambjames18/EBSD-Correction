import numpy as np
import numpy.linalg as nl
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist, squareform
import scipy
from scipy import misc

# # #
# #
# # #

# user input area

# Image to transform
    #FIB test image
a = scipy.misc.imread("ref_FIB.tif")
b = scipy.misc.imread("warped-1.png")

    #EBSD warped image
a = scipy.misc.imread("ref_IMAGE.png")
b = scipy.misc.imread("trial_BW.png")

# put source control (reference) points in here, must be paired
# use same spatial resolution as EBSD data
    #FIB test image
xs = [74,74,1095,1095,1095,800]
ys = [123,840,840,520,753,129]

    #SEM test reference
xs = [116, 433, 770, 765, 107]
ys = [136, 176, 141, 534, 523]

#put EBSD points here corresponding to same reference points
    #FIB test image
xt = [67,24,1059,1079,1067,796]
yt = [69,834,748,405,656,11]
    
    #EBSD test image
xt = [112, 445, 811 ,806, 96]
yt = [42, 105, 84, 311, 299]

# dimensions of reference image and fineness of grid (determines ultiamte resolution of transformed data)
    #FIB test image
lx = 1202 # x-length (in pixels) of reference image
ly = 960 # y-length (in pixels) of reference image

    #SEM image
lx = 833 # x-length (in pixels) of reference image
ly = 598 # y-length (in pixels) of reference image

nx = lx # num points along reference x-direction, full correction will have nx = lx
ny = ly # num points along reference y-direction, full correction will have ny = ly

# # #
# #
# # #

def makeL(cp):
    np.set_printoptions(linewidth=200)
    # cp: [K x 2] control points
    # L: [(K+3) x (K+3)]
    K = cp.shape[0]
    L = np.zeros((K+3, K+3))
    # make P in L
    L[:K, K] = 1
    L[:K, K+1:K+3] = cp
    # make P.T in L
    L[K, :K] = 1
    L[K+1:, :K] = cp.T
    # calculate U(r) and make K in L
    #R = squareform(pdist(cp, metric='euclidean'))
    R = cdist(cp, cp, 'euclidean')
    Rsq = R * R
    Rsq[R == 0] = 1 #avoid log(0) undefined, will correct itself as log(1) = 0, so U(0) = 0
    U = Rsq * np.log(Rsq)
    np.fill_diagonal(U, 0) #should be redundant
    L[:K, :K] = U
    return L

# check to make sure each control point is paired
if len(xs) == len(ys) and \
   len(xt) == len(yt) and \
   len(xs) == len(ys):
   n = len(xs)
else:
    raise ValueError('Control point arrays are not of equal length')

# convert input pixels in arrays. cps are control points
xs = np.asarray(xs)
ys = np.array(ys)
cps = np.vstack([xs,ys]).T

xt = np.asarray(xt)
yt = np.array(yt)


np.set_printoptions(linewidth=200)

# construct L
L = makeL(cps)

# construct Y
xtAug = np.concatenate([xt, np.zeros(3)])
ytAug = np.concatenate([yt, np.zeros(3)])
Y = np.vstack([xtAug,ytAug]).T

# calculate unknown params in (W | a).T
params = np.dot(nl.inv(L),Y)
wi = params[:n,:]
a1 = params[n,:]
ax = params[n+1,:]
ay = params[n+2,:]

# # verifies that functional has square integrable second derivatives. Print outs should be zero or basically zero
# wShiftX = params[:n,0]
# wShiftY = params[:n,1]
# print("Sum Wi should be zero and is: {}".format(np.sum(wi)))
# print("Sum Wi*xi should be zero and is: {}".format(np.dot(wShiftX,xs)))
# print("Sum Wi*yi should be zdro and is: {}".format(np.dot(wShiftY,ys)))

# # #
# #
# # #

#Thin plate spline calculation
# at some point (x,y) in reference, the corresponding point in the EBSD data is at
# [X,Y] = a1 + ax*xRef + ay*yRef + sum(wi*Ui)

# (x,y) coordinates from reference image
x = np.linspace(1, lx, nx)
y = np.linspace(1, ly, ny)
xgd,ygd = np.meshgrid(x,y)
pixels = np.vstack([xgd.flatten(),ygd.flatten()]).T

# affine transformation portion
axs = np.einsum('i,jk->ijk',ax,xgd)
ays = np.einsum('i,jk->ijk',ay,ygd)
affine = axs+ays
affine[0,:,:] += a1[0]
affine[1,:,:] += a1[1]

# bending portion
R = cdist(pixels, cps, 'euclidean') #are nx*ny pixels, cps = num reference pairs
# print R.shape #shape of R is pixels, reference pairs
# print R #prints R(px1-CP1) (px1-CP2) (px1-CP3)....
Rsq = R * R
Rsq[R == 0] = 1 #avoid log(0) undefined, will correct itself as log(1) = 0, so U(0) = 0
U = Rsq * np.log(Rsq)
bend = np.einsum('ij,jk->ik',U,wi).T
bend = np.reshape(bend,(2,ny,nx))

# add together portions
sol = affine + bend

# # #
# #
# # #


#get locations in original image to place back into the created grid
    #sol[0] are the corresponding x-coordinates in the distorted image
    #sol[1] are the corresponding y-coorindates in the distorted image
xgtId = np.around(sol[0]) #round to nearest pixel
xgtId = xgtId.astype(int)
xgtId = xgtId.flatten()
ygtId = np.around(sol[1]) #round to nearest pixe
ygtId = ygtId.astype(int)
ygtId = ygtId.flatten()

# determine which pixels actually lie within the distorted image
validX = (xgtId < b.shape[1])*(xgtId > 0)
validY = (ygtId < b.shape[0])*(ygtId > 0)
valid = validX*validY

# get data from distorted image at apporpiate locations, make any non-valid points = 0
c = b[validY * ygtId, validX * xgtId]
c = c * valid

#imaging

plt.gray()
plt.figure(1)
plt.title('Desired Arrangement')

plt.imshow(a)
#plt.scatter(xgd, ygd,c='b',s=1)
plt.scatter(xs,ys,marker='x', c='r', s=100)
plt.gca().set_xlim(0-.1*lx,lx+.1*lx)
plt.gca().set_ylim(0-.1*ly,ly+.1*ly)
plt.gca().invert_yaxis()

plt.figure(2)
plt.title('Source Arrangement')

plt.imshow(b)
#plt.scatter(sol[0],sol[1],c='r',s = 1)
plt.scatter(xt,yt,marker='x', c='b', s=100)
plt.gca().set_xlim(0-.1*lx,lx+.1*lx)
plt.gca().set_ylim(0-.1*ly,ly+.1*ly)
# plt.gca().set_xlim(0-.2*b.shape[1],b.shape[1]+.2*b.shape[1])
# plt.gca().set_ylim(0-.2*b.shape[0],b.shape[0]+.2*b.shape[0])
plt.gca().invert_yaxis()
plt.show()

# # plot grayscale image on nice grid
# plt.gray()
# plt.scatter(xgd.flatten(),ygd.flatten(), c=c, edgecolors = 'none',s=10)
# plt.scatter(xs,ys,marker='x', c='r', s=100)
# plt.gca().set_xlim(0-.1*lx,lx+.1*lx)
# plt.gca().set_ylim(0-.1*ly,ly+.1*ly)
# plt.gca().invert_yaxis()
# plt.show()
# #plt.savefig('foo.png',dpi=300)
