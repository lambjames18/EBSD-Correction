import os
import json
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import pickle

import Inputs


def normalize(img):
    img_min = np.min(img)
    img_max = np.max(img)
    return np.around(255 * ((img - img_min) / (img_max - img_min))).astype(np.uint8)


save_folder_base = "/Users/jameslamb/Documents/research/data/match_anything_datasets/"

json_paths = [
    "/Users/jameslamb/Documents/research/data/CoNi-HREBSD-polish/67/CoNi67_high_OM-2-SE.json",
    "/Users/jameslamb/Documents/research/data/CoNi-HREBSD-polish/67/CoNi67_high_OM-2-BSE.json",
    "/Users/jameslamb/Documents/research/data/CoNi-HREBSD-polish/67/CoNi67_mid_OM-2-high_OM.json",
    "/Users/jameslamb/Documents/research/data/CoNi-HREBSD-polish/90/CoNi90_high_OM-2-BSE.json",
    "/Users/jameslamb/Documents/research/data/CoNi-HREBSD-polish/90/CoNi90_mid_OM-2-BSE.json",
]
rescale = False


for idx in range(len(json_paths)):
    json_path = json_paths[idx]
    print(f"Processing {json_path}...")

    # Handle paths
    # Create the output folder if it doesn't exist
    # save_folder
    #     └── name
    #          ├── eval_indexs
    #          └── scenes
    name = os.path.basename(json_path).replace(".json", "")
    save_folder = os.path.join(save_folder_base, name)
    scenes_path = os.path.join(save_folder, "scenes")
    eval_indexs_path = os.path.join(save_folder, "eval_indexs")
    os.makedirs(scenes_path, exist_ok=True)
    os.makedirs(eval_indexs_path, exist_ok=True)

    # Get data from project
    with open(json_path, "r") as f:
        project_data = json.load(f)
    source_image_path = project_data["source_image"]
    destination_image_path = project_data["destination_image"]
    source_resolution = project_data["settings"]["source_resolution"]
    destination_resolution = project_data["settings"]["destination_resolution"]
    source_points = np.array(project_data["source_points"]["0"])
    destination_points = np.array(project_data["destination_points"]["0"])
    points = np.hstack((destination_points, source_points))

    # Read in the images
    if (
        source_image_path.endswith(".tif")
        or source_image_path.endswith(".tiff")
        or source_image_path.endswith(".png")
    ):
        ext = os.path.splitext(source_image_path)[1].lower()
        source_image = normalize(io.imread(source_image_path))
        source_images = [source_image]
        names = [os.path.basename(source_image_path).replace(ext, "")]
    elif source_image_path.endswith(".ang"):
        source_image = Inputs.read_ang(source_image_path)[0]
        source_images = [
            normalize(source_image["IQ"][0]),
            normalize(source_image["CI"][0]),
            normalize(
                np.stack(
                    (
                        source_image["PRIAS Bottom Strip"][0],
                        source_image["PRIAS Center Square"][0],
                        source_image["PRIAS Top Strip"][0],
                    ),
                    axis=2,
                )
            ),
        ]
        names = [
            os.path.basename(source_image_path).replace(ext, "") + "_IQ",
            os.path.basename(source_image_path).replace(ext, "") + "_CI",
            os.path.basename(source_image_path).replace(ext, "") + "_PRIAS",
        ]
    else:
        raise ValueError("Unsupported source image format")
    if (
        destination_image_path.endswith(".tif")
        or destination_image_path.endswith(".tiff")
        or destination_image_path.endswith(".png")
    ):
        ext = os.path.splitext(destination_image_path)[1].lower()
        destination_image = normalize(io.imread(destination_image_path))
        if rescale:
            if len(destination_image.shape) == 2:
                destination_image = destination_image.reshape(
                    (1,) + destination_image.shape + (1,)
                )
            elif len(destination_image.shape) == 3:
                destination_image = destination_image.reshape(
                    (1,) + destination_image.shape
                )
            destination_image = np.squeeze(
                Inputs.rescale_control(
                    dict(intensity=destination_image),
                    destination_resolution,
                    source_resolution,
                    channel_axis=-1,
                )["intensity"]
            )
    else:
        raise ValueError("Unsupported destination image format")

    images = source_images + [destination_image]

    # Create save paths
    os.makedirs(os.path.join(scenes_path, "scene0"), exist_ok=True)
    image_paths = [f"scenes/scene0/{names[i]}.tif" for i in range(len(names))]
    image_paths.append(f"scenes/scene0/{os.path.basename(destination_image_path)}")

    # Save the images
    for i in range(len(images)):
        io.imsave(
            os.path.join(save_folder, image_paths[i]),
            images[i],
        )

    # Create the pair info
    num_source = len(source_images)
    pair_infos = [([num_source, i], 0) for i in range(num_source)]

    # Create the ground truth 2D matches
    gt_2D_matches = [points for _ in range(len(pair_infos))]

    # Save the .npz file
    data = dict(
        dataset_name=name,
        image_paths=image_paths,
        pair_infos=pair_infos,
        gt_2D_matches=gt_2D_matches,
    )
    print(data)
    data_path = os.path.join(eval_indexs_path, f"eval_{name}_{i}.npz")
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
