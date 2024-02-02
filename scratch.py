import os
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import Inputs
import core
from copy import deepcopy


def _get_corrected_centroid(im, align, ref_data, points=None):
    if points is None:
        im = align.apply(im, out="array").astype(bool)
    else:
        im = align.TPS_apply_3D(points, im, ref_data)
        im = im.astype(bool).sum(axis=0).astype(bool)
    rows = np.any(im, axis=1)
    cols = np.any(im, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return (rmin + rmax) // 2, (cmin + cmax) // 2


def _get_cropping_slice(centroid, target_shape, current_shape):
    """Returns a slice object that can be used to crop an image"""
    rstart = centroid[0] - target_shape[0] // 2
    rend = rstart + target_shape[0]
    if rstart < 0:
        r_slice = slice(0, target_shape[0])
    elif rend > current_shape[0]:
        r_slice = slice(current_shape[0] - target_shape[0], current_shape[0])
    else:
        r_slice = slice(rstart, rend)

    cstart = centroid[1] - target_shape[1] // 2
    cend = cstart + target_shape[1]
    # print("Col raw:", cstart, cend)
    if cstart < 0:
        c_slice = slice(0, target_shape[1])
    elif cend > current_shape[1]:
        c_slice = slice(current_shape[1] - target_shape[1], current_shape[1])
    else:
        c_slice = slice(cstart, cend)
    # print("Col slice:", c_slice)
    return r_slice, c_slice


def _check_sizes(im1, im2, ndims=2):
    """im1: ebsd (distorted), im2: ebsd (corrected)"""
    if ndims == 2:
        if im1.shape[0] > im2.shape[0]:
            im2_temp = np.zeros((im1.shape[0], im2.shape[1]))
            im2_temp[:im2.shape[0], :] = im2
            im2 = im2_temp
        if im1.shape[1] > im2.shape[1]:
            im2_temp = np.zeros((im2.shape[0], im1.shape[1]))
            im2_temp[:, :im2.shape[1]] = im2
            im2 = im2_temp
        return im2
    else:
        if im1.shape[1] > im2.shape[1]:
            im2_temp = np.zeros((im2.shape[0], im1.shape[1], im2.shape[2]))
            im2_temp[:, :im2.shape[1], :] = im2
            im2 = im2_temp
        if im1.shape[2] > im2.shape[2]:
            im2_temp = np.zeros((im2.shape[0], im2.shape[1], im1.shape[2]))
            im2_temp[:, :, :im2.shape[2]] = im2
            im2 = im2_temp
        return im2

# START INPUT PROCESSING
ebsd_path = "/Users/jameslamb/Documents/Research/scripts/EBSD-Correction/test_data/EBSD.ang"
ebsd_points_path = "/Users/jameslamb/Documents/Research/scripts/EBSD-Correction/test_data/distorted_pts.txt"
bse_path = "/Users/jameslamb/Documents/Research/scripts/EBSD-Correction/test_data/BSE.tif"
bse_points_path = "/Users/jameslamb/Documents/Research/scripts/EBSD-Correction/test_data/control_pts.txt"
SAVE_PATH_EBSD = "/Users/jameslamb/Documents/Research/scripts/EBSD-Correction/test_data/test/EBSD-Output.ang"
SAVE_DIR_IMGS = "/Users/jameslamb/Documents/Research/scripts/EBSD-Correction/test_data/test/"
SAVE_PATH_BSE = "/Users/jameslamb/Documents/Research/scripts/EBSD-Correction/test_data/test/BSE-Output.tiff"
ebsd_res = 2.5
bse_res = 1.0
rescale = True
r180 = False
flip = False
crop = False
e_d, b_d, e_pts, b_pts = Inputs.read_data(ebsd_path, bse_path, ebsd_points_path, bse_points_path)
if e_pts is None:
    e_pts = {0: []}
    ebsd_points_path = os.path.dirname(ebsd_path) + "/distorted_pts.txt"
    with open(ebsd_points_path, "w", encoding="utf8") as output:
        output.write("")
if b_pts is None:
    b_pts = {0: []}
    bse_points_path = os.path.dirname(ebsd_path) + "/control_pts.txt"
    with open(bse_points_path, "w", encoding="utf8") as output:
        output.write("")
if rescale:
    b_d = Inputs.rescale_control(b_d, bse_res, ebsd_res)
if flip:
    b_d = np.flip(b_d, axis=1).copy(order='C')
elif r180:
    b_d = np.rot90(b_d, 2, axes=(1,2)).copy(order='C')
# END OF INPUT PROCESSING

print("End of input processing, starting correction")

# START OF CORRECTION
# Get the control points
index = 0
referencePoints = np.array(b_pts[index])
distortedPoints = np.array(e_pts[index])
align = core.Alignment(referencePoints, distortedPoints, algorithm="TPS")
# Get BSE
bse_im = b_d[int(index)]
# Create the output filename
extension = os.path.splitext(SAVE_PATH_EBSD)[1]
# Align the image
align.get_solution(size=bse_im.shape)

print("Starting saving")

# SAVE THE ANG FILE
data = deepcopy(e_d)
del data["EulerAngles"]
data["Phase index"] = np.ones_like(data["Phase index"])
columns = list(data.keys())
data_stack = np.zeros((len(columns), *data["x"][0].shape))
print("EBSD entries:", columns)
print("EBSD grid size:", data_stack.shape)
for i, key in enumerate(columns):
    # Get the images
    ebsd_im = np.squeeze(data[key][int(index)])
    print(key, ebsd_im.min(), ebsd_im.max(), ebsd_im.mean(), ebsd_im.std())
    if len(ebsd_im.shape) == 3:
        aligned = []
        for i in range(ebsd_im.shape[-1]):
            aligned.append(align.apply(ebsd_im[:, :, i], out="array"))
        aligned = np.moveaxis(np.array(aligned), 0, -1)
    else:
        aligned = align.apply(ebsd_im, out="array")
    if key.lower() in ["phi1", "phi", "phi2"]:
        aligned[aligned == 0] = 4*float(np.pi)
    elif key.lower() == "ci":
        aligned[aligned == 0] = -1
    elif key.lower() == "fit":
        aligned[aligned == 0] = 180
    # Correct dtype
    # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    # fig.suptitle(key)
    # ax[0].imshow(ebsd_im)
    # ax[1].imshow(aligned)
    aligned = np.around(core.handle_dtype(aligned, ebsd_im.dtype), 5)
    # ax[2].imshow(aligned)
    # plt.show()
    # Correct shape
    aligned = _check_sizes(ebsd_im, aligned)
    bse_im = _check_sizes(ebsd_im, bse_im)
    # Do this by correcting an empty image (all ones) and finding the centroid of the corrected image
    dummy = np.ones(ebsd_im.shape)
    rc, cc = _get_corrected_centroid(dummy, align, b_d)
    # Now crop the corrected image
    rslc, cslc = _get_cropping_slice((rc, cc), ebsd_im.shape, aligned.shape)
    aligned = aligned[rslc, cslc]
    print(key, aligned.min(), aligned.max(), aligned.mean(), aligned.std())
    data_stack[i] = aligned
d = data_stack.reshape(data_stack.shape[0], -1).T
x_index = columns.index("x")
y_index = columns.index("y")
y, x = np.indices(data["x"][0].shape)
x = x.ravel() * ebsd_res
y = y.ravel() * ebsd_res
d[:, x_index] = x
d[:, y_index] = y
# Get the header
with open(ebsd_path, "r") as f:
    header = []
    for line in f.readlines():
        if line.startswith("#"):
            header.append(line)
        else:
            break
header = "".join(header)
# Save the data
with open(SAVE_PATH_EBSD, "w") as f:
    f.write(header)
    for i in range(d.shape[0]):
        fmts = ["%.5f", "%.5f", "%.5f", "%.5f", "%.5f", "%.1f", "%.3f", "%.0f", "%.0f", "%.3f", "%.6f", "%.6f", "%.6f"]
        space = [3, 4, 4, 7, 7, 7, 3, 3, 7, 4, 7, 7, 7]
        line = [" "*(space[j]-len(str(int(d[i,j])))) + fmts[j] % (d[i,j]+0.0) for j in range(d.shape[1])]
        line = "".join(line)
        f.write(f" {line}\n")

print("Finished ANG saving, starting image saving")

temp_ebsd_im = np.squeeze(e_d[list(e_d.keys())[0]][int(index)])
dummy = np.ones(temp_ebsd_im.shape)
rc, cc = _get_corrected_centroid(dummy, align, b_d)
# Now crop the corrected image
bse_im = _check_sizes(temp_ebsd_im, bse_im)
rslc, cslc = _get_cropping_slice((rc, cc), ebsd_im.shape, bse_im.shape)
bse_im = bse_im[rslc, cslc]
bse_im = core.handle_dtype(bse_im, np.uint8)
io.imsave(SAVE_PATH_BSE, bse_im)

# SAVE THE EBSD DATA AS IMAGES
for key in e_d.keys():
    ebsd_im = np.squeeze(e_d[key][int(index)])
    if len(ebsd_im.shape) == 3:
        aligned = []
        for i in range(ebsd_im.shape[-1]):
            aligned.append(align.apply(ebsd_im[:, :, i], out="array"))
        aligned = np.moveaxis(np.array(aligned), 0, -1)
    else:
        aligned = align.apply(ebsd_im, out="array")
    if key.lower() in ["eulerangles", "phi1", "phi", "phi2"]:
        aligned[aligned == 0] = 4*float(np.pi)
    elif key.lower() == "ci":
        aligned[aligned == 0] = -1
    elif key.lower() == "fit":
        aligned[aligned == 0] = 180
    # Correct dtype
    aligned = core.handle_dtype(aligned, np.uint8)
    # Correct shape
    aligned = _check_sizes(ebsd_im, aligned)
    # Do this by correcting an empty image (all ones) and finding the centroid of the corrected image
    rslc, cslc = _get_cropping_slice((rc, cc), ebsd_im.shape, aligned.shape)
    aligned = aligned[rslc, cslc]
    # Save the image
    io.imsave(SAVE_DIR_IMGS + key + ".tiff", aligned)

print("Finished image saving, starting BSE saving")
