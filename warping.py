import numpy as np
from skimage import transform as tf

from tps import ThinPlateSplineTransform

def get_transform(src, dst, mode, *args, **kwargs):
    if mode.lower() == "tps":
        tform = ThinPlateSplineTransform()
        tform.estimate(src, dst, *args, **kwargs)
    else:
        tform = tf.estimate_transform(mode.lower(), src, dst, *args, **kwargs)
    return tform


def get_transform_params(tform):
    return tform.params


def set_transform_params(tform, params):
    tform.params = params


def transform_coords(src, dst, mode="tps", return_params=False, *args, **kwargs):
    """
    Transform coordinates from source to destination using thin plate spline or affine transformation.

    Parameters
    ----------
    src : (N, 2) ndarray
        Source coordinates.
    dst : (N, 2) ndarray
        Destination coordinates.
    mode : str, optional
        Transformation mode. One of euclidean, similarity, affine, piecewise-affine, projective, polynomial, tps.
        See skimage.transform.estimate_transform for more information.
    *args, **kwargs : optional, optional
        Additional arguments for skimage.transform.estimate_transform.

    Returns
    -------
    (N, 2) ndarray
        Transformed coordinates.
    """
    tform = get_transform(src, dst, mode, *args, **kwargs)
    warped = tform(src)
    if return_params:
        return warped, get_transform_params(tform)
    return warped


def transform_image(image, src, dst, output_shape=None, mode="tps", order=0, return_params=False, *args, **kwargs):
    """
    Transform an image from source to destination using thin plate spline or affine transformation.

    Parameters
    ----------
    image : (H, W) ndarray
        Input image.
    src : (N, 2) ndarray
        Source coordinates.
    dst : (N, 2) ndarray
        Destination coordinates.
    mode : str, optional
        Transformation mode. One of euclidean, similarity, affine, piecewise-affine, projective, polynomial, tps.
        See skimage.transform.estimate_transform for more information.
    *args, **kwargs : optional, optional
        Additional arguments for skimage.transform.estimate_transform.

    Returns
    -------
    (H, W) ndarray
        Transformed image.
    """
    if output_shape is None:
        output_shape = image.shape
    tform = get_transform(src, dst, mode, *args, **kwargs)
    warped = tf.warp(image, tform, mode="constant", cval=0, order=order, output_shape=output_shape)
    if return_params:
        return warped, get_transform_params(tform)
    return warped


def transform_image_stack(images, srcs, dsts, mode, order=0, *args, **kwargs):
    """
    Transform a stack of images from source to destination using thin plate spline or affine transformation.

    Parameters
    ----------
    images : (N, H, W, C) ndarray
        Input images.
    srcs : (N, M, 2) ndarray
        Source coordinates.
    dsts : (N, M, 2) ndarray
        Destination coordinates.
    mode : str
        Transformation mode. One of euclidean, similarity, affine, piecewise-affine, projective, polynomial, tps.
        See skimage.transform.estimate_transform for more information.
    order : int
        The order of the spline interpolation.
    *args, **kwargs : optional, optional
        Additional arguments for skimage.transform.estimate_transform.

    Returns
    -------
    (N, H, W) ndarray
        Transformed images.
    """
    # Parse the slices, cappping the first and last slices with the closest slice with points if necessary
    slice_numbers = np.arange(images.shape[0])
    slice_numbers_with_points = np.unique(srcs[:, 0])
    if slice_numbers[0] not in slice_numbers_with_points:
        src_temp = srcs[srcs[:, 0] == slice_numbers_with_points[0], 1:]
        dst_temp = dsts[dsts[:, 0] == slice_numbers_with_points[0], 1:]
        _0 = np.zeros((src_temp.shape[0], 1))
        src_temp = np.concatenate([_0, src_temp], axis=1)
        dst_temp = np.concatenate([_0, dst_temp], axis=1)
        srcs = np.concatenate([src_temp, srcs], axis=0)
        dsts = np.concatenate([dst_temp, dsts], axis=0)
        slice_numbers_with_points = np.concatenate([[0], slice_numbers_with_points])
    if slice_numbers[-1] not in slice_numbers_with_points:
        src_temp = srcs[srcs[:, 0] == slice_numbers_with_points[-1], 1:]
        dst_temp = dsts[dsts[:, 0] == slice_numbers_with_points[-1], 1:]
        _0 = np.zeros((src_temp.shape[0], 1))
        src_temp = np.concatenate([_0, src_temp], axis=1)
        dst_temp = np.concatenate([_0, dst_temp], axis=1)
        srcs = np.concatenate([srcs, src_temp], axis=0)
        dsts = np.concatenate([dsts, dst_temp], axis=0)
        slice_numbers_with_points = np.concatenate([slice_numbers_with_points, [slice_numbers[-1]]])

    # Determine the transformation parameters by running a single transformation
    src_temp = srcs[srcs[:, 0] == slice_numbers_with_points[0], 1:]
    dst_temp = dsts[dsts[:, 0] == slice_numbers_with_points[0], 1:]
    tform = get_transform(src_temp, dst_temp, mode, *args, **kwargs)
    param_shape = get_transform_params(tform).shape

    # Fill in the parameters where the points are
    params = np.zeros((images.shape[0], *param_shape))
    for slice_number in slice_numbers_with_points:
        src = srcs[srcs[:, 0] == slice_number, 1:]
        dst = dsts[dsts[:, 0] == slice_number, 1:]
        tform_temp = get_transform(src, dst, mode, *args, **kwargs)
        params[slice_number] = get_transform_params(tform_temp)

    # Interpolate the parameters where the points are not
    for i in range(1, len(slice_numbers_with_points)):
        slice_number_0 = slice_numbers_with_points[i - 1]
        slice_number_1 = slice_numbers_with_points[i]
        params[slice_number_0:slice_number_1] = np.linspace(params[slice_number_0], params[slice_number_1], slice_number_1 - slice_number_0)

    # Transform the images
    output = []
    for i in range(images.shape[0]):
        set_transform_params(tform, params[i])
        output.append(tf.warp(images[i], tform, output_shape=images[i].shape, mode="constant", cval=0, order=order))
    return np.array(output)


if __name__ == "__main__":
    import Inputs
    import matplotlib.pyplot as plt

    ebsd_path = "./test_data/EBSD.ang"
    ebsd_points_path = "./test_data/distorted_pts.txt"
    bse_path = "./test_data/BSE.tif"
    bse_points_path = "./test_data/control_pts.txt"
    ebsd_res = 2.5
    bse_res = 1.0
    rescale = True
    r180 = False
    flip = False
    crop = False
    src_data, dst_data, src_points, dst_points = Inputs.read_data(ebsd_path, bse_path, ebsd_points_path, bse_points_path)
    src_points, dst_points = src_points[0], dst_points[0]
    dst_data = Inputs.rescale_control(dst_data, bse_res, ebsd_res)
    dst_img = np.squeeze(dst_data[list(dst_data.keys())[0]])
    src_img = np.squeeze(src_data["CI"])

    if dst_img.shape[0] < src_img.shape[0]:
        dst_img = np.pad(dst_img, ((0, src_img.shape[0] - dst_img.shape[0]), (0, 0)), mode="constant", constant_values=0)
    if dst_img.shape[1] < src_img.shape[1]:
        dst_img = np.pad(dst_img, ((0, 0), (0, src_img.shape[1] - dst_img.shape[1])), mode="constant", constant_values=0)

    print("SRC", src_img.shape)
    print("DST", dst_img.shape)
    t_image = transform_image(src_img, dst_points, src_points, output_shape=dst_img.shape, mode="tps", order=0, size=dst_img.shape)
    print(t_image.shape)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    ax[0].imshow(src_img, cmap="gray")
    ax[0].scatter(src_points[:, 0], src_points[:, 1], c="r", s=10)
    ax[0].set_title("Source")
    ax[1].imshow(dst_img, cmap="gray")
    ax[1].scatter(dst_points[:, 0], dst_points[:, 1], c="b", s=10)
    ax[1].set_title("Destination")
    ax[2].imshow(t_image, cmap="gray")
    ax[2].set_title("Transformed")
    plt.tight_layout()
    plt.show()
