import core
from skimage import io
import matplotlib.pyplot as plt
import numpy as np

from masking import create_ped_mask

# Create the control points
bse = "coni16_459.tif"
ebsd = "coni16_459_eds.tif"
loop = True
while loop:
    pick = input("Select control points? (y/n) ")
    if pick == "y":
        bse_ctr = core.SelectCoords(bse)
        bse_ctr_path = str(bse_ctr.txt_path)
        ebsd_ctr = core.SelectCoords(ebsd)
        ebsd_ctr_path = str(ebsd_ctr.txt_path)
        loop = False
    elif pick == "n":
        bse_ctr_path = "ctr_pts_coni16_459.txt"
        ebsd_ctr_path = "ctr_pts_coni16_459_eds.txt"
        loop = False
    else:
        loop = True

# Find solution and apply
ebsd = io.imread(ebsd)
ebsd = io.imread("Al.tif")
pick = input("Find alignment solution? (y/n) ")
align = core.Alignment(bse_ctr_path, ebsd_ctr_path)
loop = True
while loop:
    if pick == 'y':
        align.TPS(bse, saveSolution=True)
        loop = False
    elif pick == 'n':
        align.TPS_import("TPS_mapping.npy", bse)
        loop = False
    else: loop = True

align.TPS_apply(ebsd)

# View in the slider
from matplotlib.widgets import Slider

# Read in images
im0 = io.imread("TPS_out.tif")
im1 = io.imread(bse)[:,:,0]

im0 = np.where(im0 > 15, 255, 0)

# Crop out background
mask, filled = create_ped_mask(im1)
im0[filled == False] = 0
im1[filled == False] = 0

max_r = im0.shape[0]
max_c = im0.shape[1]
alphas = np.ones(im0.shape)
fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(im1, cmap="viridis")
im = ax.imshow(im0, alpha=alphas, cmap="gray")

# plt.tight_layout()
plt.subplots_adjust(left=0.15, bottom=0.15)
left = ax.get_position().x0
bot = ax.get_position().y0
width = ax.get_position().width
height = ax.get_position().height
axrow = plt.axes([left - 0.15, bot, 0.05, height])
axcol = plt.axes([left, bot - 0.15, width, 0.05])
row_slider = Slider(
    ax=axrow, label="Y position", valmin=0, valmax=max_r, valinit=max_r, orientation="vertical"
)
col_slider = Slider(
    ax=axcol, label="X position", valmin=0, valmax=max_c, valinit=max_c, orientation="horizontal"
)


def update_row(val):
    val = int(np.around(val, 0))
    new_alphas = np.copy(alphas)
    new_alphas[:val, :] = 0
    im.set(alpha=new_alphas[::-1])
    fig.canvas.draw_idle()


def update_col(val):
    val = int(np.around(val, 0))
    new_alphas = np.copy(alphas)
    new_alphas[:, :val] = 0
    im.set(alpha=new_alphas)
    fig.canvas.draw_idle()


row_slider.on_changed(update_row)
col_slider.on_changed(update_col)
plt.show()
# plt.savefig("test.png")
