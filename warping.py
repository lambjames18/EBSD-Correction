import numpy as np
from skimage import transform as tf

from tps import ThinPlateSplineTransform


def get_transform(src, dst, mode, *args, **kwargs):
    if mode.lower() == "tps":
        tform = ThinPlateSplineTransform()
        tform.estimate(src, dst, *args, **kwargs)
    elif mode.lower() == "tps affine":
        tform = ThinPlateSplineTransform(affine_only=True)
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


def transform_image(
    image,
    src,
    dst,
    output_shape=None,
    mode="tps",
    order=0,
    return_params=False,
    *args,
    **kwargs
):
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
    warped = tf.warp(
        image, tform, mode="constant", cval=0, order=order, output_shape=output_shape
    )
    if return_params:
        return warped, get_transform_params(tform)
    return warped


def transform_image_stack(
    images, srcs, dsts, output_shape=None, mode="tps", order=0, *args, **kwargs
):
    """
    Transform a stack of images from source to destination using thin plate spline or affine transformation.

    Parameters
    ----------
    images : (N, H, W, C) ndarray
        Input images.
    srcs : (M, 3) ndarray
        Source coordinates.
    dsts : (M, 3) ndarray
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
    (N, H, W, C) ndarray
        Transformed images.
    """
    if output_shape is None:
        output_shape = images.shape[1:3]
    # Parse the slices, cappping the first and last slices with the closest slice with points if necessary
    # The first and last slice need to have points in order to the interpolation below to work
    # The user should select points in the first and last slice, but this is a workaround if they don't
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
        slice_numbers_with_points = np.concatenate(
            [slice_numbers_with_points, [slice_numbers[-1]]]
        )

    # Determine the transformation parameters by running a single transformation
    src_temp = srcs[srcs[:, 0] == slice_numbers_with_points[0], 1:]
    dst_temp = dsts[dsts[:, 0] == slice_numbers_with_points[0], 1:]
    tform = get_transform(src_temp, dst_temp, mode, *args, **kwargs)
    param_shape = get_transform_params(tform).shape

    # Fill in the parameters where the points are
    # Here we create the "knots" along the z-axis
    # Linear interpolation is done between the knots
    # Could change this to a more advanced interpolation
    params = np.zeros((images.shape[0], *param_shape))
    print(
        slice_numbers_with_points.shape,
        slice_numbers_with_points.dtype,
        slice_numbers_with_points,
    )
    for slice_number in slice_numbers_with_points:
        src = srcs[srcs[:, 0] == slice_number, 1:]
        dst = dsts[dsts[:, 0] == slice_number, 1:]
        tform_temp = get_transform(src, dst, mode, *args, **kwargs)
        params[slice_number] = get_transform_params(tform_temp)

    # Interpolate the parameters where the points are not
    for i in range(1, len(slice_numbers_with_points)):
        slice_number_0 = slice_numbers_with_points[i - 1]
        slice_number_1 = slice_numbers_with_points[i]
        params[slice_number_0:slice_number_1] = np.linspace(
            params[slice_number_0],
            params[slice_number_1],
            slice_number_1 - slice_number_0,
        )

    # Transform the images
    output = []
    for i in range(images.shape[0]):
        set_transform_params(tform, params[i])
        output.append(
            tf.warp(
                images[i],
                tform,
                output_shape=output_shape,
                mode="constant",
                cval=0,
                order=order,
            )
        )
    return np.array(output)


if __name__ == "__main__":
    import Inputs
    import h5py
    import core
    import matplotlib.pyplot as plt
    from skimage import io

    ebsd_path = "/Users/jameslamb/Documents/research/data/CoNi-DIC-S1/stitched_EBSD.ang"
    bse_path = "/Users/jameslamb/Documents/research/data/CoNi-DIC-S1/stitched_CBS.tif"
    src_img = Inputs.read_ang(ebsd_path)[0]["GrainIDs"][0]
    dst_img = Inputs.read_image(bse_path)[0]

    ebsd_points_path = (
        "/Users/jameslamb/Documents/research/data/CoNi-DIC-S1/preDIC-src_pts.txt"
    )
    bse_points_path = (
        "/Users/jameslamb/Documents/research/data/CoNi-DIC-S1/preDIC-dst_pts.txt"
    )
    src_points = np.loadtxt(ebsd_points_path, dtype=int)[:, 1:]
    dst_points = np.loadtxt(bse_points_path, dtype=int)[:, 1:]

    print("SRC", src_img.shape, src_img.dtype)
    print("DST", dst_img.shape, dst_img.dtype)
    t_image = transform_image(
        src_img,
        dst_points,
        src_points,
        output_shape=dst_img.shape,
        mode="tps",
        order=0,
        size=dst_img.shape,
    )
    t_image = t_image[: dst_img.shape[0], : dst_img.shape[1]]
    print("SRC", src_img.shape, src_img.dtype)
    t_image = core.handle_dtype(t_image, t_image.dtype)

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
