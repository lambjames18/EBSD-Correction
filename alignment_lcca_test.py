import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import kornia
from tqdm.auto import tqdm

from lcca_cmaes_monofile import lcca_cmaes_homography, batch_cca

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)


path0 = "D:/Research/Ta/Data/3D/AMSpall/Ta_AM-Spalled_basic.dream3d"
h5 = h5py.File(path0, 'r')
ipf0 = h5["DataContainers/ImageDataContainer/CellData/IPFColor_001"][...][:100]
q0 = h5["DataContainers/ImageDataContainer/CellData/Quats"][...][:100]
h5.close()

path1 = "D:/Research/Ta/Data/3D/AMSpall/TaAMSpall.dream3d"
h5 = h5py.File(path1, 'r')
ipf1 = h5["DataContainers/ImageDataContainer/CellData/IPFColor_001"][...][:100]
h5.close()
print("Data read")

# sigmas = [0.1, 1.0, 10.0]
# n_iterations = [10, 100, 1000]
# n_patches = [100, 1000, 10000]
# output = np.zeros((len(sigmas), len(n_iterations), len(n_patches)))
# pbar = tqdm(total=len(sigmas)*len(n_iterations)*len(n_patches))
# pbar.set_description("Processing")
# 
# for i, sigma in enumerate(sigmas):
#     for j, iters in enumerate(n_iterations):
#         for k, patches in enumerate(n_patches):
#             src = ipf0[0]
#             tar = ipf0[1]
#             src_norm = src / max([np.max(src), np.max(src)])
#             tar_norm = tar / max([np.max(tar), np.max(tar)])
#             src_tensor = torch.tensor(src_norm).float()[None].to(device)
#             tar_tensor = torch.tensor(tar_norm).float()[None].to(device)
#             src_tensor = torch.moveaxis(src_tensor, -1, 1)
#             tar_tensor = torch.moveaxis(tar_tensor, -1, 1)
#             h, lie = lcca_cmaes_homography(
#                 n_iterations=iters,
#                 cmaes_population=10,
#                 cmaes_sigma=sigma,
#                 n_patches=patches,
#                 img_source=src_tensor,
#                 img_target=tar_tensor,
#                 patch_size=(5, 5),
#                 guess_lie=None,
#                 x_translation_weight=1.0,
#                 y_translation_weight=1.0,
#                 rotation_weight=1.0,
#                 isotropic_scale_weight=1.0,
#                 anisotropic_stretch_weight=1.0,
#                 shear_weight=1.0,
#                 x_keystone_weight=0.0,  # set to zero for affine
#                 y_keystone_weight=0.0,  # set to zero for affine, determines if lines need to be parallel
#                 verbose=False,
#             )
#             tar_warped_tensor = kornia.geometry.warp_perspective(tar_tensor, h, dsize=tar_tensor.shape[-2:])
#             err = batch_cca(src_tensor, tar_warped_tensor).squeeze().cpu().numpy()[0]
#             output[i, j, k] = err
#             pbar.update(1)
# 
# fig, ax = plt.subplots(1, 3, figsize=(12, 4))
# ax[0].imshow(output[0])
# ax[1].imshow(output[1])
# ax[2].imshow(output[2])
# ax[0].set_title("Sigma = 0.1")
# ax[1].set_title("Sigma = 1.0")
# ax[2].set_title("Sigma = 10.0")
# for i in range(3):
#     ax[i].set_xticks(range(len(n_iterations)))
#     ax[i].set_xticklabels(n_iterations)
#     ax[i].set_yticks(range(len(n_patches)))
#     ax[i].set_yticklabels(n_patches)
#     ax[i].set_xlabel("Iterations")
#     ax[i].set_ylabel("Patches")
# plt.tight_layout()
# plt.show()
# exit()

ipf0_corrected = np.zeros_like(ipf0)
ipf0_corrected[0] = ipf0[0]
q0_corrected = np.zeros_like(q0)
q0_corrected[0] = q0[0]
for i in tqdm(range(1, ipf0.shape[0])):
    src = q0_corrected[i-1]
    tar = q0[i]
    tar_ipf = ipf0[i]

    src_norm = src / max([np.max(src), np.max(tar)])
    tar_norm = tar / max([np.max(src), np.max(tar)])
    tar_ipf_norm = tar_ipf / max([np.max(tar_ipf), np.max(tar_ipf)])
    src_tensor = torch.tensor(src_norm).float()[None].to(device)
    tar_tensor = torch.tensor(tar_norm).float()[None].to(device)
    tar_ipf_tensor = torch.tensor(tar_ipf_norm).float()[None].to(device)
    src_tensor = torch.moveaxis(src_tensor, -1, 1)
    tar_tensor = torch.moveaxis(tar_tensor, -1, 1)
    tar_ipf_tensor = torch.moveaxis(tar_ipf_tensor, -1, 1)
    
    homography, lie_solution = lcca_cmaes_homography(
        n_iterations=100,
        cmaes_population=10,
        cmaes_sigma=1.0,
        n_patches=1000,
        img_source=src_tensor,
        img_target=tar_tensor,
        patch_size=(5, 5),
        guess_lie=None,
        x_translation_weight=1.0,
        y_translation_weight=1.0,
        rotation_weight=1.0,
        isotropic_scale_weight=1.0,
        anisotropic_stretch_weight=1.0,
        shear_weight=1.0,
        x_keystone_weight=0.0,  # set to zero for affine
        y_keystone_weight=0.0,  # set to zero for affine, determines if lines need to be parallel
        loss="misorientation",
        verbose=False,
    )

    tar_warped_tensor = kornia.geometry.warp_perspective(tar_tensor, homography, dsize=tar_tensor.shape[-2:])
    tar_warped = (tar_warped_tensor[0].squeeze()* 255.0).byte().cpu().numpy()
    tar_warped = np.moveaxis(tar_warped, 0, -1)
    q0_corrected[i] = tar_warped

    tar_ipf_warped_tensor = kornia.geometry.warp_perspective(tar_ipf_tensor, homography, dsize=tar_ipf_tensor.shape[-2:])
    tar_ipf_warped = (tar_ipf_warped_tensor[0].squeeze()* 255.0).byte().cpu().numpy()
    tar_ipf_warped = np.moveaxis(tar_ipf_warped, 0, -1)
    ipf0_corrected[i] = tar_ipf_warped


original = ipf0[:, :, ipf0.shape[2]//2]
final = ipf1[:, :, ipf1.shape[2]//2]
new = ipf0_corrected[:, :, ipf0.shape[2]//2]

fig, ax = plt.subplots(1, 3)
ax[0].imshow(original)
ax[1].imshow(final)
ax[2].imshow(new)
ax[0].set_title("Original")
ax[1].set_title("Final")
ax[2].set_title("New")
plt.show()