import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import os


# Simple mouse click function to store coordinates
def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata

    global img
    x = np.around(ix, 0).astype(np.uint32)
    y = np.around(iy, 0).astype(np.uint32)

    with open(txt_path, "a") as output:
        output.write(f"{x} {y}\n")
    print(np.loadtxt(txt_path, delimiter=" ")[-1])


def close(event):
    fig1.canvas.mpl_disconnect(cid)
    fig1.canvas.mpl_disconnect(qid)
    plt.close(1)


# Specify which image to look at
which = "ebsd"
slice_id = 385
txt_path = f"ctr_pts_{which}.txt"
im_path = f"{slice_id}_{which}.tif"
"""
# Old points
try:
    os.remove(txt_path)
except FileNotFoundError:
    pass
"""

# Open image and select points
img = io.imread(im_path)
fig1 = plt.figure(1, figsize=(12, 8))
ax1 = fig1.add_subplot(111)
ax1.imshow(img)
cid = fig1.canvas.mpl_connect("button_press_event", onclick)
qid = fig1.canvas.mpl_connect("close_event", close)
plt.show()

# Save image with coordinates drawn on top
bse_pts = np.loadtxt(txt_path, delimiter=" ")

fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
ax2.imshow(img)
for i in range(bse_pts.shape[0]):
    ax2.scatter(bse_pts[i, 0], bse_pts[i, 1], c="r", s=1)
    ax2.text(bse_pts[i, 0] + 2, bse_pts[i, 1] + 2, i)
fig2.savefig(f"{slice_id}_{which}_points.png")
