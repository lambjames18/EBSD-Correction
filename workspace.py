"""
Author: James Lamb

Script designed to align distorted ebsd to bse using either linear regression or thin-plate spline
"""

import core
from skimage import io
import matplotlib.pyplot as plt
import numpy as np

####
# Create the control points
folder = "Slice420_CoNi16/"  # Folder where to save everything, can be empty ""
bse = "bse"  # The name of the distorted image (should be tif) without the extension
ebsd = "ebsd"  # The name of the distorted image (should be tif) without the extension
algorithm = "TPS"  # Select the algorithm, either LR or TPS
view_overlay = True  # Overlays the corrected distortion over the control image with sliders
####

loop = True
while loop:
    pick = input("Select control points? (y/n) ")
    if pick == "y":
        bse_ctr = core.SelectCoords(bse, save_folder=folder)
        bse_ctr_path = str(bse_ctr.txt_path)
        ebsd_ctr = core.SelectCoords(ebsd, save_folder=folder)
        ebsd_ctr_path = str(ebsd_ctr.txt_path)
        loop = False
    elif pick == "n":
        bse_ctr_path = f"{folder}ctr_pts_{bse}.txt"
        ebsd_ctr_path = f"{folder}ctr_pts_{ebsd}.txt"
        loop = False
    else:
        loop = True

# Find solution and apply
ebsd_im = io.imread(folder + ebsd + ".tif")
align = core.Alignment(bse_ctr_path, ebsd_ctr_path, algorithm=algorithm)
loop = True
kwargs = {"referenceImage": folder + bse + ".tif"}
# kwargs = {}
while loop:
    pick = input("Find alignment solution? (y/n) ")
    if pick == "y":
        align.get_solution(
            saveSolution=True, solutionFile=f"{folder}{algorithm}_mapping.npy", **kwargs
        )
        # align.TPS(folder + bse + ".tif", saveSolution=True)
        loop = False
    elif pick == "n":
        print(f"{folder}TPS_mapping.npy")
        align.import_solution(f"{folder}{algorithm}_mapping.npy", f"{folder}{bse}.tif")
        # align.TPS_import(folder + "TPS_mapping.npy", folder + bse + ".tif")
        loop = False
    else:
        loop = True

# align.TPS_apply(ebsd_im, folder + "TPS_out.tif")
align.apply(ebsd_im, f"{folder}{algorithm}_out.tif")

# View in the slider
from matplotlib.widgets import Slider

# Read in images
im0 = io.imread(folder + f"{algorithm}_out.tif")
im1 = io.imread(folder + bse + ".tif")

# Setup figure
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111)
ax.set_title(f"{algorithm.upper()} Output")

if view_overlay:
    # Setup stuff
    max_r = im0.shape[0]
    max_c = im0.shape[1]
    alphas = np.ones(im0.shape)
    # Show images
    ax.imshow(im1, cmap="bone")
    im = ax.imshow(im0, alpha=alphas, cmap="gray")
    # Put slider on
    plt.subplots_adjust(left=0.15, bottom=0.15)
    left = ax.get_position().x0
    bot = ax.get_position().y0
    width = ax.get_position().width
    height = ax.get_position().height
    axrow = plt.axes([left - 0.15, bot, 0.05, height])
    axcol = plt.axes([left, bot - 0.15, width, 0.05])
    row_slider = Slider(
        ax=axrow, label="Y pos", valmin=0, valmax=max_r, valinit=max_r, orientation="vertical"
    )
    col_slider = Slider(
        ax=axcol, label="X pos", valmin=0, valmax=max_c, valinit=max_c, orientation="horizontal"
    )
    # Define update functions
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

    # Enable update functions
    row_slider.on_changed(update_row)
    col_slider.on_changed(update_col)
else:
    ax.imshow(im0, cmap="bone")
    plt.tight_layout()
plt.show()
# plt.savefig("test.png")
