import numpy as np
import matplotlib.pyplot as plt
from skimage import io

bse_pts = np.loadtxt("ctr_pts_bse.txt", delimiter="\t")
bse_im = io.imread("Slice76.tif")
print(bse_pts.shape)

plt.imshow(bse_im)
for i in range(bse_pts.shape[0]):
    plt.scatter(bse_pts[i, 0], bse_pts[i, 1], c="r")
    plt.text(bse_pts[i, 0] + 2, bse_pts[i, 1] + 2, i)
plt.show()
