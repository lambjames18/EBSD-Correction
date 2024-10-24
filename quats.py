import torch


# Quaternion stuff
@torch.jit.script
def laue_elements(laue_id: int) -> torch.Tensor:
    """
    Generators for Laue group specified by the laue_id parameter. The first
    element is always the identity.

    1) Laue C1       Triclinic: 1-, 1
    2) Laue C2      Monoclinic: 2/m, m, 2
    3) Laue D2    Orthorhombic: mmm, mm2, 222
    4) Laue C4  Tetragonal low: 4/m, 4-, 4
    5) Laue D4 Tetragonal high: 4/mmm, 4-2m, 4mm, 422
    6) Laue C3    Trigonal low: 3-, 3
    7) Laue D3   Trigonal high: 3-m, 3m, 32
    8) Laue C6   Hexagonal low: 6/m, 6-, 6
    9) Laue D6  Hexagonal high: 6/mmm, 6-m2, 6mm, 622
    10) Laue T       Cubic low: m3-, 23
    11) Laue O      Cubic high: m3-m, 4-3m, 432

    Args:
        laue_id: integer between inclusive [1, 11]

    Returns:
        torch tensor of shape (cardinality, 4) containing the elements of the

    Notes:

    https://en.wikipedia.org/wiki/Space_group

    """

    # sqrt(2) / 2 and sqrt(3) / 2
    R2 = 1.0 / (2.0**0.5)
    R3 = (3.0**0.5) / 2.0

    LAUE_O = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [R2, 0.0, 0.0, R2],
            [R2, 0.0, 0.0, -R2],
            [0.0, R2, R2, 0.0],
            [0.0, -R2, R2, 0.0],
            [0.5, 0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5, -0.5],
            [0.5, -0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5],
            [R2, R2, 0.0, 0.0],
            [R2, -R2, 0.0, 0.0],
            [R2, 0.0, R2, 0.0],
            [R2, 0.0, -R2, 0.0],
            [0.0, R2, 0.0, R2],
            [0.0, -R2, 0.0, R2],
            [0.0, 0.0, R2, R2],
            [0.0, 0.0, -R2, R2],
        ],
        dtype=torch.float64,
    )
    LAUE_T = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.5, 0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5, -0.5],
            [0.5, -0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5],
        ],
        dtype=torch.float64,
    )

    LAUE_D6 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, R3],
            [0.5, 0.0, 0.0, -R3],
            [0.0, 0.0, 0.0, 1.0],
            [R3, 0.0, 0.0, 0.5],
            [R3, 0.0, 0.0, -0.5],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -0.5, R3, 0.0],
            [0.0, 0.5, R3, 0.0],
            [0.0, R3, 0.5, 0.0],
            [0.0, -R3, 0.5, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )

    LAUE_C6 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, R3],
            [0.5, 0.0, 0.0, -R3],
            [0.0, 0.0, 0.0, 1.0],
            [R3, 0.0, 0.0, 0.5],
            [R3, 0.0, 0.0, -0.5],
        ],
        dtype=torch.float64,
    )

    LAUE_D3 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, R3],
            [0.5, 0.0, 0.0, -R3],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -0.5, R3, 0.0],
            [0.0, 0.5, R3, 0.0],
        ],
        dtype=torch.float64,
    )

    LAUE_C3 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, R3],
            [0.5, 0.0, 0.0, -R3],
        ],
        dtype=torch.float64,
    )

    LAUE_D4 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [R2, 0.0, 0.0, R2],
            [R2, 0.0, 0.0, -R2],
            [0.0, R2, R2, 0.0],
            [0.0, -R2, R2, 0.0],
        ],
        dtype=torch.float64,
    )

    LAUE_C4 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [R2, 0.0, 0.0, R2],
            [R2, 0.0, 0.0, -R2],
        ],
        dtype=torch.float64,
    )

    LAUE_D2 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )

    LAUE_C2 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )

    LAUE_C1 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )

    LAUE_GROUPS = [
        LAUE_C1,  #  1 - Triclinic
        LAUE_C2,  #  2 - Monoclinic
        LAUE_D2,  #  3 - Orthorhombic
        LAUE_C4,  #  4 - Tetragonal low
        LAUE_D4,  #  5 - Tetragonal high
        LAUE_C3,  #  6 - Trigonal low
        LAUE_D3,  #  7 - Trigonal high
        LAUE_C6,  #  8 - Hexagonal low
        LAUE_D6,  #  9 - Hexagonal high
        LAUE_T,  #  10 - Cubic low
        LAUE_O,  #  11 - Cubic high
    ]

    return LAUE_GROUPS[laue_id - 1]


@torch.jit.script
def qu_conj(qu: torch.Tensor) -> torch.Tensor:
    """
    Get the unit quaternions for the inverse action.

    Args:
        qu: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """
    scaling = torch.tensor([1, -1, -1, -1], device=qu.device, dtype=qu.dtype)
    return qu * scaling


@torch.jit.script
def qu_std(qu: torch.Tensor) -> torch.Tensor:
    """
    Standardize unit quaternion to have non-negative real part.

    Args:
        qu: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(qu[..., 0:1] >= 0, qu, -qu).to(qu.device)


@torch.jit.script
def qu_prod_raw(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: shape (..., 4) quaternions in form (w, x, y, z)
        b: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1).to(a.device)


@torch.jit.script
def qu_prod(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Quaternion multiplication, then make real part non-negative.

    Args:
        a: shape (..., 4) quaternions in form (w, x, y, z)
        b: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        a*b Tensor shape (..., 4) of the quaternion product.

    """
    ab = qu_prod_raw(a, b)
    return qu_std(ab)


@torch.jit.script
def qu_prod_pos_real(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Return only the magnitude of the real part of the quaternion product.

    Args:
        a: shape (..., 4) quaternions in form (w, x, y, z)
        b: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        a*b Tensor shape (..., ) of quaternion product real part magnitudes.
    """
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ow = aw * bw - ax * bx - ay * by - az * bz
    return ow.abs()


@torch.jit.script
def qu_triple_prod_pos_real(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Return only the magnitude of the real part of the quaternion triple product.

    Args:
        a: shape (..., 4) quaternions in form (w, x, y, z)
        b: shape (..., 4) quaternions in form (w, x, y, z)
        c: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        a*b*c Tensor shape (..., ) of quaternion triple product real part magnitudes.
    """
    return qu_prod_pos_real(a, qu_prod(b, c))


@torch.jit.script
def disori_angle_laue(quats1: torch.Tensor, quats2: torch.Tensor, laue_id_1: int, laue_id_2: int):
    """

    Return the disorientation angle in radians between the given quaternions.

    Args:
        quats1: quaternions of shape (..., 4)
        quats2: quaternions of shape (..., 4)
        laue_id_1: laue group ID of quats1
        laue_id_2: laue group ID of quats2

    Returns:
        disorientation quaternion of shape (..., 4)

    """

    # get the important shapes
    data_shape = quats1.shape

    # check that the shapes are the same
    if data_shape != quats2.shape:
        raise ValueError(
            f"quats1 and quats2 must have the same data shape, but got {data_shape} and {quats2.shape}"
        )

    # multiply by inverse of second (without symmetry)
    misori_quats = qu_prod(quats1, qu_conj(quats2))

    # find the number of quaternions (generic input shapes are supported)
    N = torch.prod(torch.tensor(data_shape[:-1]))

    # retrieve the laue group elements for the first quaternions
    laue_group_1 = laue_elements(laue_id_1).to(quats1.dtype).to(quats1.device)

    # if the laue groups are the same, then the second laue group is the same as the first
    if laue_id_1 == laue_id_2:
        laue_group_2 = laue_group_1
    else:
        laue_group_2 = laue_elements(laue_id_2).to(quats2.dtype).to(quats2.device)

    # pre / post mult by Laue operators of the second and first symmetry groups respectively
    # broadcasting is done so that the output is of shape (N, |laue_group_2|, |laue_group_1|, 4)
    equivalent_quat_pos_real = qu_triple_prod_pos_real(
        laue_group_2.reshape(1, -1, 1, 4),
        misori_quats.view(N, 1, 1, 4),
        laue_group_1.reshape(1, 1, -1, 4),
    )

    # flatten along the laue group dimensions
    equivalent_quat_pos_real = equivalent_quat_pos_real.reshape(N, -1)

    # find the largest real part magnitude and return the angle
    cosine_half_angle = torch.max(equivalent_quat_pos_real, dim=-1).values

    return 2.0 * torch.acos(cosine_half_angle)


@torch.jit.script
def misorientation_gpu(
    q1_image: torch.Tensor,
    q2_image: torch.Tensor, 
    degrees: bool = False,
    ) -> torch.Tensor:
    """Compute the average misorientation between two images containing quaternions.
    
    Args:
        q1_image (torch.Tensor): An image containing quaternions. Shape (B, 4, H, W)
        q2_image (torch.Tensor): An image containing quaternions. Shape (B, 4, H, W)
        degrees (bool, optional): Return the misorientation in degrees. Defaults to True.

    Returns:
        torch.Tensor: The average misorientation between the two images. Shape (B, H, W)
    """
    R2 = 0.7071067811865475244008443621048490392848359376884740365883398689
    R3 = 0.8660254037844386467637231707529361834714026269051903140279034897
    LAUE_O_GPU = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [R2, 0.0, 0.0, R2],
            [R2, 0.0, 0.0, -R2],
            [0.0, R2, R2, 0.0],
            [0.0, -R2, R2, 0.0],
            [0.5, 0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5, -0.5],
            [0.5, -0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5],
            [R2, R2, 0.0, 0.0],
            [R2, -R2, 0.0, 0.0],
            [R2, 0.0, R2, 0.0],
            [R2, 0.0, -R2, 0.0],
            [0.0, R2, 0.0, R2],
            [0.0, -R2, 0.0, R2],
            [0.0, 0.0, R2, R2],
            [0.0, 0.0, -R2, R2],
        ],
        dtype=torch.float64,
    ).to(q1_image.device)[None, None, None]

    # Move the quaternions to the last dimension
    q1_image = qu_std(q1_image.permute(0, 2, 3, 1).unsqueeze(3))
    q2_image = qu_std(q2_image.permute(0, 2, 3, 1).unsqueeze(3))

    # Create all symmetrically equivalent quaternions for both images
    q1_sym = qu_prod(q1_image, LAUE_O_GPU)
    q2_sym = qu_prod(q2_image, LAUE_O_GPU)

    # Compute the misorientation between the two images
    q_mis = qu_prod(q1_sym.unsqueeze(4), qu_conj(q2_sym).unsqueeze(3))

    # Get the angle of the misorientations
    norm = torch.sqrt((q_mis[..., 1:] ** 2).sum(dim=-1, keepdim=True))
    angles = 2 * torch.atan2(norm, q_mis[..., 0:1])
    # angles = 2 * torch.acos(q_mis[..., 0].abs())
    # axes = q_mis[..., 1:] / norm

    # Reshape
    angles = angles.reshape(q_mis.shape[:3] + (-1,))
    # axes = axes.reshape(q_mis.shape[:3] + (-1, 3))

    # Find the minimum angle for each point
    argmins = torch.argmin(angles.abs(), dim=-1, keepdim=True)
    min_angles = torch.gather(angles, -1, argmins).squeeze(-1)
    # min_axes = torch.gather(axes, -2, argmins.unsqueeze(-1).expand(-1, -1, -1, -1, 3)).squeeze(-2)

    # Preserve rotation direction
    # min_angles[min_axes[..., 2] < 0] *= -1
    # min_axes[min_axes[..., 2] < 0] *= -1

    # Convert to degrees if necessary
    if degrees:
        min_angles = torch.rad2deg(min_angles)

    return min_angles


if __name__ == "__main__":
    import numpy as np

    q1 = torch.zeros((1, 4, 2, 3)).double()
    q1[0] = torch.tensor([1.0, 0.0, 0.0, 0.0]).double().unsqueeze(1).unsqueeze(2)


    q2 = torch.zeros((1, 4, 2, 3)).double()
    u, theta = np.array([0, 0, 1]), -np.deg2rad(10)
    w, x, y, z = np.cos(theta / 2), *np.sin(theta / 2) * u
    q2[0] = torch.tensor([w, x, y, z]).double().unsqueeze(1).unsqueeze(2)


    o = misorientation_gpu(q1, q2, degrees=True)
    print(o, o.shape)