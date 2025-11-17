import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import pickle

import Inputs


name = "CoNi67"
bse_paths = [
    "/Users/jameslamb/Documents/research/data/auto_reg_data/CoNi67/21_se.tif",
    "/Users/jameslamb/Documents/research/data/auto_reg_data/CoNi67/22_se.tif",
    "/Users/jameslamb/Documents/research/data/auto_reg_data/CoNi67/23_se.tif",
]
bse_pts_paths = [
    "/Users/jameslamb/Documents/research/data/auto_reg_data/CoNi67/21_bse_pts.txt",
    "/Users/jameslamb/Documents/research/data/auto_reg_data/CoNi67/22_bse_pts.txt",
    "/Users/jameslamb/Documents/research/data/auto_reg_data/CoNi67/23_bse_pts.txt",
]
ebsd_paths = [
    "/Users/jameslamb/Documents/research/data/auto_reg_data/CoNi67/21_ebsd.ang",
    "/Users/jameslamb/Documents/research/data/auto_reg_data/CoNi67/22_ebsd.ang",
    "/Users/jameslamb/Documents/research/data/auto_reg_data/CoNi67/23_ebsd.ang",
]
ebsd_pts_paths = [
    "/Users/jameslamb/Documents/research/data/auto_reg_data/CoNi67/21_ebsd_pts.txt",
    "/Users/jameslamb/Documents/research/data/auto_reg_data/CoNi67/22_ebsd_pts.txt",
    "/Users/jameslamb/Documents/research/data/auto_reg_data/CoNi67/23_ebsd_pts.txt",
]
save_folder = "/Users/jameslamb/Documents/research/data/auto_reg_data/test_data/"
bse_resolution = 0.52083333
ebsd_resolution = 1.5


# name = "AMSpalledTa"
# bse_paths = [
#     "/Users/jameslamb/Documents/research/data/auto_reg_data/AMSpalledTa/511_bse.tif",
#     "/Users/jameslamb/Documents/research/data/auto_reg_data/AMSpalledTa/512_bse.tif",
#     "/Users/jameslamb/Documents/research/data/auto_reg_data/AMSpalledTa/513_bse.tif",
# ]
# bse_pts_paths = [
#     "/Users/jameslamb/Documents/research/data/auto_reg_data/AMSpalledTa/511_bse_pts.txt",
#     "/Users/jameslamb/Documents/research/data/auto_reg_data/AMSpalledTa/512_bse_pts.txt",
#     "/Users/jameslamb/Documents/research/data/auto_reg_data/AMSpalledTa/513_bse_pts.txt",
# ]
# ebsd_paths = [
#     "/Users/jameslamb/Documents/research/data/auto_reg_data/AMSpalledTa/511_ebsd.ang",
#     "/Users/jameslamb/Documents/research/data/auto_reg_data/AMSpalledTa/512_ebsd.ang",
#     "/Users/jameslamb/Documents/research/data/auto_reg_data/AMSpalledTa/513_ebsd.ang",
# ]
# ebsd_pts_paths = [
#     "/Users/jameslamb/Documents/research/data/auto_reg_data/AMSpalledTa/511_ebsd_pts.txt",
#     "/Users/jameslamb/Documents/research/data/auto_reg_data/AMSpalledTa/512_ebsd_pts.txt",
#     "/Users/jameslamb/Documents/research/data/auto_reg_data/AMSpalledTa/513_ebsd_pts.txt",
# ]
# save_folder = "/Users/jameslamb/Documents/research/data/auto_reg_data/test_data/"
# bse_resolution = 0.27669271
# ebsd_resolution = 1.5


def normalize(img):
    img_min = np.min(img)
    img_max = np.max(img)
    return np.around(255 * ((img - img_min) / (img_max - img_min))).astype(np.uint8)


# Create the output folder if it doesn't exist
# save_folder
#     └── name
#          ├── eval_indexs
#          └── scenes
save_folder = os.path.join(save_folder, name)
scenes_path = os.path.join(save_folder, "scenes")
eval_indexs_path = os.path.join(save_folder, "eval_indexs")
os.makedirs(scenes_path, exist_ok=True)
os.makedirs(eval_indexs_path, exist_ok=True)

# Loop over the pairs
for i in range(len(bse_paths)):
    bse_path = bse_paths[i]
    bse_pts_path = bse_pts_paths[i]
    ebsd_path = ebsd_paths[i]
    ebsd_pts_path = ebsd_pts_paths[i]

    # Read in the data
    ebsd_data, bse_data, ebsd_points, bse_points = Inputs.read_data(
        ebsd_path, bse_path, ebsd_pts_path, bse_pts_path
    )

    # Process the EBSD data
    iq = normalize(ebsd_data["IQ"][0].astype(float))
    ci = normalize(ebsd_data["CI"][0].astype(float))
    prias = np.stack(
        (
            normalize(ebsd_data["PRIAS Bottom Strip"][0]),
            normalize(ebsd_data["PRIAS Center Square"][0]),
            normalize(ebsd_data["PRIAS Top Strip"][0]),
        ),
        axis=2,
    )

    # Process the BSE data
    bse_data = Inputs.rescale_control(bse_data, bse_resolution, ebsd_resolution)
    bse_data = bse_data["Intensity"][0, ..., 0]

    # Create save paths
    image_paths = [
        f"scenes/{name}_{i}/{os.path.basename(ebsd_path).replace('.ang', '')}_IQ.tif",
        f"scenes/{name}_{i}/{os.path.basename(ebsd_path).replace('.ang', '')}_CI.tif",
        f"scenes/{name}_{i}/{os.path.basename(ebsd_path).replace('.ang', '')}_PRIAS.tif",
        f"scenes/{name}_{i}/{os.path.basename(bse_path).replace('.tif', '')}.tif",
    ]

    # Save the images
    os.makedirs(os.path.join(save_folder, "scenes", f"{name}_{i}"), exist_ok=True)
    io.imsave(os.path.join(save_folder, image_paths[0]), iq)
    io.imsave(os.path.join(save_folder, image_paths[1]), ci)
    io.imsave(os.path.join(save_folder, image_paths[2]), prias)
    io.imsave(os.path.join(save_folder, image_paths[3]), bse_data)

    # Create the pair info
    pair_infos = [
        ([3, 0], 1),  # IQ to BSE
        ([3, 1], 1),  # CI to BSE
        ([3, 2], 1),  # PRIAS to BSE
    ]

    # Process the points
    ebsd_points = ebsd_points[0]
    bse_points = bse_points[0]
    points = np.hstack((bse_points, ebsd_points))
    # points = np.hstack((ebsd_points, bse_points))

    # Create the ground truth 2D matches
    gt_2D_matches = [points for _ in range(len(pair_infos))]
    # gt_2D_matches = [points.tolist() for _ in range(len(pair_infos))]

    # Save the .npz file
    data = dict(
        dataset_name=name,
        image_paths=image_paths,
        pair_infos=pair_infos,
        gt_2D_matches=gt_2D_matches,
    )
    data_path = os.path.join(save_folder, "eval_indexs", f"eval_{name}_{i}.npz")
    # np.savez(data_path, **data)
    pickle.dump(data, open(data_path, "wb"))

    # Verify the saved .npz file
    data = np.load(data_path, allow_pickle=True)
    assert data["dataset_name"] == name
    assert type(data["dataset_name"]) == str
    assert type(data["image_paths"]) == list
    assert type(data["image_paths"][0]) == str
    assert type(data["pair_infos"]) == list
    assert type(data["pair_infos"][0]) == tuple
    assert type(data["gt_2D_matches"]) == list
    assert type(data["gt_2D_matches"][0]) == np.ndarray
