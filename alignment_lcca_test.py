import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import kornia
from tqdm.auto import tqdm

from lcca_cmaes_monofile import lcca_cmaes_homography, batch_cca

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)


path0 = "D:/Research/Ta/Data/3D/AMSpall/Ta_AM-Spalled_basic.dream3d"
h5 = h5py.File(path0, "r")
ipf0 = h5["DataContainers/ImageDataContainer/CellData/IPFColor_001"][...][:100]
q0 = h5["DataContainers/ImageDataContainer/CellData/Quats"][...][:100]
h5.close()

path1 = "D:/Research/Ta/Data/3D/AMSpall/TaAMSpall.dream3d"
h5 = h5py.File(path1, "r")
ipf1 = h5["DataContainers/ImageDataContainer/CellData/IPFColor_001"][...][:100]
h5.close()
print("Data read")


def correlate2d(image, template):
    """Perform cross correlation between an image and a template (another image).
    This is done on the GPU using PyTorch.

    Args:
        image (np.array): Image to correlate with. (H,W,C) or (H,W).
        template (np.array): Template to correlate. (H,W,C) or (H,W).

    Returns:
        np.array: Correlation result.
    """
    if template.ndim == 2:
        template = template[None, None]
    elif template.ndim == 3:
        template = np.moveaxis(template, -1, 0)[None]
    template = torch.tensor(template).float().to(device)
    if image.ndim == 2:
        image = image[None, None]
    elif image.ndim == 3:
        image = np.moveaxis(image, -1, 0)[None]
    image = torch.tensor(image).float().to(device)
    # result = kornia.geometry.filter2D(image, template)
    result = torch.nn.functional.conv2d(image, template, padding="same")
    print(result.shape)
    return result.squeeze().cpu().numpy()


im0 = ipf0[0]
im1 = ipf0[1]
cc = correlate2d(im0, im1)
max_idx = np.unravel_index(np.argmax(cc), cc.shape)
shift = np.array(max_idx) - np.array(im0.shape[:2]) // 2
m = kornia.geometry.transform.get_affine_matrix2d(
    translations=torch.tensor(shift).float().to(device)[None],
    center=torch.tensor((0, 0)).float().to(device)[None],
    scale=torch.tensor((1, 1)).float().to(device)[None],
    angle=torch.tensor((0,)).float().to(device),
)

print(m, im0.shape[:2])
im1_fixed = kornia.geometry.warp_perspective(
    torch.tensor(im1)[None].float().to(device), m, dsize=im0.shape[:2], mode="nearest"
)
im1_fixed = im1_fixed.squeeze().cpu().numpy()
print(im1_fixed.shape)

error0 = (im0 - im1) ** 2
error1 = (im0 - im1_fixed) ** 2

fig, ax = plt.subplots(2, 3)
ax[0, 0].imshow(im0)
ax[0, 0].set_title("Original")

ax[0, 1].imshow(im1)
ax[0, 1].set_title("Shifted")

ax[0, 2].imshow(im1_fixed)
ax[0, 2].set_title("Fixed")

ax[1, 0].imshow(cc)
ax[1, 0].plot(max_idx[1], max_idx[0], "ro")
ax[1, 0].set_title("Cross-correlation")

ax[1, 1].imshow(error0)
ax[1, 1].set_title("MSE (raw): {:.2f}".format(error0.mean()))

ax[1, 2].imshow(error1)
ax[1, 2].set_title("MSE (fixed): {:.2f}".format(error1.mean()))

plt.tight_layout()
plt.savefig("correlation.png")
plt.close()
