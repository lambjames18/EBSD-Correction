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


def transform_coords(src, dst, mode="tps", *args, **kwargs):
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
    return tform(src)


def transform_image(image, src, dst, mode="tps", order=0, *args, **kwargs):
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
    tform = get_transform(src, dst, mode, *args, **kwargs)
    return tf.warp(image, tform, output_shape=image.shape, mode="constant", cval=0, order=order)


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
    import matplotlib.pyplot as plt
    from skimage import data
    image = data.checkerboard()
    image_stack = np.array([image, image, image, image, image])
    src = np.loadtxt("src.txt", delimiter=",", dtype=int)
    dst = np.loadtxt("dst.txt", delimiter=",", dtype=int)
    print(image_stack.shape)
    print(src.shape)
    print(dst.shape)

    kw = dict(size=image.shape)
    t_images = transform_image_stack(image_stack, src, dst, "tps", 0, **kw)
    print(t_images.shape)

    fig, ax = plt.subplots(1, 5, figsize=(20, 4), sharex=True, sharey=True)
    for i in range(5):
        ax[i].imshow(t_images[i, ..., 1:], cmap="gray")
        ax[i].axis("off")
    plt.tight_layout()
    plt.show()
