import time
from typing import Optional, Tuple
import torch
from torch import Tensor
from torch.nn import Module
import numpy as np
import math
import random
from cmaes import CMA

import quats


@torch.jit.script
def batch_cca(
    x: Tensor,
    y: Tensor,
    cca_type: str = "cov",
    std_threshold: float = 0.0001,
    eps: float = 1e-4,
) -> Tensor:
    """
    Computes the batched CCA between two matrices of
    shape (..., n, m1) and (..., n, m2). n is the number of
    data points and m is the dimension of the data points.
    The CCA is computed batch-wise on the last two dimensions.

    Args:
        x (Tensor): The first input matrix of shape (..., n, m1).
        y (Tensor): The second input matrix of shape (..., n, m2).
        cca_type (str):The type of CCA to compute. Options are 'cov' and 'corr'. Defaults to 'cov'.
            The 'cov' option computes the CCA using the covariance matrices. The 'corr' option
            computes the CCA using the correlation matrices.
        std_threshold (float, optional): A threshold on the standard deviation to prevent division
            by zero. Defaults to 0.0001. If the standard deviation is less than this value, the
            standard deviation is set to 1.
        eps (float, optional): A small value to add to the covariance matrices to
            prevent singular matrices. Defaults to 1e-4.

    Returns:
        Tensor: The mean-across channel CCA correlations of shape (...,)


    The canonical correlation coefficients are the square roots of the eigenvalues of the
    correlation matrix between the two sets of variables. The function returns the mean of
    the first min(m_dim(x), m_dim(y)) absolute values of the correlations.

    """

    # Standardize the input matrices
    x = x - x.mean(dim=-2, keepdim=True)
    y = y - y.mean(dim=-2, keepdim=True)

    # if type is correlation, normalize by the standard deviation
    if cca_type == "corr":
        x_std = x.std(dim=-2, keepdim=True)
        y_std = y.std(dim=-2, keepdim=True)
        x = x / torch.where(x_std < std_threshold, torch.ones_like(x_std), x_std)
        y = y / torch.where(y_std < std_threshold, torch.ones_like(y_std), y_std)

    # Compute covariance matrices
    cov_xx = torch.matmul(x.transpose(-2, -1), x) / (x.shape[-2] - 1)
    cov_yy = torch.matmul(y.transpose(-2, -1), y) / (y.shape[-2] - 1)
    cov_xy = torch.matmul(x.transpose(-2, -1), y) / (x.shape[-2] - 1)

    # Compute the inverse square root of cov_xx and cov_yy
    inv_sqrt_xx = torch.linalg.inv(
        torch.linalg.cholesky(
            cov_xx + eps * torch.eye(cov_xx.shape[-1], device=x.device)
        )
    )
    inv_sqrt_yy = torch.linalg.inv(
        torch.linalg.cholesky(
            cov_yy + eps * torch.eye(cov_yy.shape[-1], device=y.device), upper=True
        )
    )

    # Compute the canonical correlation matrix
    cov_matrices = torch.matmul(torch.matmul(inv_sqrt_xx, cov_xy), inv_sqrt_yy)

    # return the trace over the max dimension (min rank)
    return cov_matrices.diagonal(dim1=-2, dim2=-1).abs().mean(dim=-1)


class CMAESWrapper:
    def __init__(
        self,
        device: torch.device,
        dimension: int,
        population_size: int,
        mean: Tensor,
        sigma: float,
        minimize: bool = True,
        bounds: Optional[Tensor] = None,
        dtype_out: torch.dtype = torch.float32,
    ):
        if not isinstance(dimension, int):
            raise TypeError("dimension argument must be an int for CMAWrapper")
        if not isinstance(population_size, int):
            raise TypeError("population_size argument must be an int for CMAWrapper")
        if not isinstance(sigma, float):
            raise TypeError("sigma argument must be a float for CMAWrapper")
        if not isinstance(device, torch.device):
            raise TypeError("device argument must be a torch.device for CMAWrapper")

        self.dimension = dimension
        self.population_size = population_size
        self.mean = mean
        self.sigma = sigma
        self.minimize = minimize
        self.bounds = bounds
        self.device = device
        self.dtype_out = dtype_out

        self.cma = CMA(
            mean=mean.cpu().numpy(),
            sigma=sigma,
            bounds=bounds.cpu().numpy() if bounds is not None else None,
            population_size=population_size,
        )

        if self.minimize:
            self.best_fitness = torch.inf
        else:
            self.best_fitness = -torch.inf

        self.first_told = False

    def ask(self):
        candidates_np = np.array([self.cma.ask() for _ in range(self.population_size)])
        return torch.from_numpy(candidates_np).to(self.device).to(self.dtype_out)

    def first_tell(self, candidate, fitness):
        if not candidate.shape == (1, self.dimension):
            raise ValueError(
                f"Start solution of shape {candidate.shape} is not shape ({1}, {self.dimension}) for CMAWrapper"
            )

        if self.first_told:
            raise RuntimeError("first_tell has already been called for this CMAWrapper")

        self.first_told = True

        self.best_fitness = fitness.item()
        self.best_solution = candidate.cpu().numpy()[0]

    def tell(self, candidates, fitnesses):
        if (
            not candidates.shape == (self.population_size, self.dimension)
            and candidates.shape[0] != 1
        ):
            raise ValueError(
                f"solutions argument of shape {candidates.shape} is not shape ({self.population_size}, {self.dimension}) for CMAWrapper"
            )
        if not fitnesses.shape == (self.population_size,):
            raise ValueError(
                f"fitnesses argument of shape {fitnesses.shape} is not shape ({self.population_size},) for CMAWrapper"
            )

        # cma tell expects a list of tuples of (np.array, float)
        candidates = candidates.cpu().numpy()
        fitnesses = fitnesses.cpu().numpy()

        # update best solution CMA minimizes, so we need to negate the fitnesses if we are maximizing
        best_batch_index = (
            np.argmin(fitnesses) if self.minimize else np.argmax(fitnesses)
        )
        best_batch_fitness = fitnesses[best_batch_index]

        if self.minimize:
            tell_list = list(zip(candidates, fitnesses))
            if best_batch_fitness < self.best_fitness:
                self.best_fitness = best_batch_fitness
                self.best_solution = candidates[best_batch_index]
        else:
            tell_list = list(zip(candidates, -1.0 * fitnesses))
            if best_batch_fitness > self.best_fitness:
                self.best_fitness = best_batch_fitness
                self.best_solution = candidates[best_batch_index]

        self.cma.tell(tell_list)

    def should_stop(self):
        return self.cma.should_stop()

    def get_best_solution(self):
        return torch.from_numpy(self.best_solution).to(self.device)[None, :]

    def get_best_fitness(self):
        return self.best_fitness


"""

Lenstra elliptic-curve factorization method 

Originally from:

https://stackoverflow.com/questions/4643647/fast-prime-factorization-module

and

https://en.wikipedia.org/wiki/Lenstra_elliptic-curve_factorization


Useful for reshaping arrays into approximately square shapes for GPU processing.
Also useful for finding special bandwidth numbers for FFTs on SO(3).

"""


def FactorECM(N0):
    def factor_trial_division(x):
        factors = []
        while (x & 1) == 0:
            factors.append(2)
            x >>= 1
        for d in range(3, int(math.sqrt(x)) + 1, 2):
            while x % d == 0:
                factors.append(d)
                x //= d
        if x > 1:
            factors.append(x)
        return sorted(factors)

    def is_probably_prime_fermat(n, trials=32):
        for _ in range(trials):
            if pow(random.randint(2, n - 2), n - 1, n) != 1:
                return False
        return True

    def gen_primes_sieve_of_eratosthenes(end):
        composite = [False] * end
        for p in range(2, int(math.sqrt(end)) + 1):
            if composite[p]:
                continue
            for i in range(p * p, end, p):
                composite[i] = True
        return [p for p in range(2, end) if not composite[p]]

    def gcd(a, b):
        while b != 0:
            a, b = b, a % b
        return a

    def egcd(a, b):
        ro, r, so, s = a, b, 1, 0
        while r != 0:
            ro, (q, r) = r, divmod(ro, r)
            so, s = s, so - q * s
        return ro, so, (ro - so * a) // b

    def modular_inverse(a, n):
        g, s, _ = egcd(a, n)
        if g != 1:
            raise ValueError(a)
        return s % n

    def elliptic_curve_add(N, A, B, X0, Y0, X1, Y1):
        if X0 == X1 and Y0 == Y1:
            l = ((3 * X0**2 + A) * modular_inverse(2 * Y0, N)) % N
        else:
            l = ((Y1 - Y0) * modular_inverse(X1 - X0, N)) % N
        x = (l**2 - X0 - X1) % N
        y = (l * (X0 - x) - Y0) % N
        return x, y

    def elliptic_curve_mul(N, A, B, X, Y, k):
        k -= 1
        BX, BY = X, Y
        while k != 0:
            if k & 1:
                X, Y = elliptic_curve_add(N, A, B, X, Y, BX, BY)
            BX, BY = elliptic_curve_add(N, A, B, BX, BY, BX, BY)
            k >>= 1
        return X, Y

    def factor_ecm(N, bound=512, icurve=0):
        def next_factor_ecm(x):
            return factor_ecm(x, bound=bound + 512, icurve=icurve + 1)

        def prime_power(p, bound2=int(math.sqrt(bound) + 1)):
            mp = p
            while mp * p < bound2:
                mp *= p
            return mp

        if N < (1 << 16):
            return factor_trial_division(N)

        if is_probably_prime_fermat(N):
            return [N]

        while True:
            X, Y, A = [random.randrange(N) for _ in range(3)]
            B = (Y**2 - X**3 - A * X) % N
            if 4 * A**3 - 27 * B**2 != 0:
                break

        for p in gen_primes_sieve_of_eratosthenes(bound):
            k = prime_power(p)
            try:
                X, Y = elliptic_curve_mul(N, A, B, X, Y, k)
            except ValueError as ex:
                g = gcd(ex.args[0], N)
                if g != N:
                    return sorted(next_factor_ecm(g) + next_factor_ecm(N // g))
                else:
                    return next_factor_ecm(N)
        return next_factor_ecm(N)

    return factor_ecm(N0)


def nearly_square_factors(n):
    """
    For a given highly composite number n, find two factors that are roughly
    close to each other. This is useful for reshaping some arrays into square
    shapes for GPU processing.
    """
    factor_a = 1
    factor_b = 1
    factors = FactorECM(n)
    for factor in factors[::-1]:
        if factor_a > factor_b:
            factor_b *= factor
        else:
            factor_a *= factor
    return factor_a, factor_b


@torch.jit.script
def transform_points(trans_01: Tensor, points_1: Tensor) -> Tensor:
    r"""Function that applies transformations to a set of points.

    Args:
        trans_01: tensor for transformations of shape
          :math:`(B, D+1, D+1)`.
        points_1: tensor of points of shape :math:`(B, N, D)`.
    Returns:
        a tensor of N-dimensional points.

    Shape:
        - Output: :math:`(B, N, D)`

    """
    if not trans_01.shape[0] == points_1.shape[0] and trans_01.shape[0] != 1:
        raise ValueError(
            "Input batch size must be the same for both tensors or 1."
            f"Got {trans_01.shape} and {points_1.shape}"
        )
    if not trans_01.shape[-1] == (points_1.shape[-1] + 1):
        raise ValueError(
            "Last input dimensions must differ by one unit"
            f"Got{trans_01} and {points_1}"
        )

    # We reshape to BxNxD in case we get more dimensions, e.g., MxBxNxD
    shape_inp = list(points_1.shape)
    points_1 = points_1.reshape(-1, points_1.shape[-2], points_1.shape[-1])
    trans_01 = trans_01.reshape(-1, trans_01.shape[-2], trans_01.shape[-1])
    # We expand trans_01 to match the dimensions needed for bmm
    trans_01 = torch.repeat_interleave(
        trans_01, repeats=points_1.shape[0] // trans_01.shape[0], dim=0
    )
    # to homogeneous
    points_1_h = torch.nn.functional.pad(points_1, [0, 1], "constant", 1.0)  # BxNxD+1
    # transform coordinates
    points_0_h = torch.bmm(points_1_h, trans_01.permute(0, 2, 1))
    points_0_h = torch.squeeze(points_0_h, dim=-1)

    # we check for points at max_val
    z_vec: Tensor = points_0_h[..., -1:]
    mask: Tensor = torch.abs(z_vec) > 1e-8
    scale = torch.where(mask, 1.0 / (z_vec + 1e-8), torch.ones_like(z_vec))
    points_0 = scale * points_0_h[..., :-1]  # BxNxD
    # reshape to the input shape
    shape_inp[-2] = points_0.shape[-2]
    shape_inp[-1] = points_0.shape[-1]
    points_0 = points_0.reshape(shape_inp)
    return points_0


@torch.jit.script
def normed_homographies_on_points(
    points: Tensor,
    homographies: Tensor,
) -> Tensor:
    """
    Apply to a grid of points, a batch of homographies defined between [-1, 1]
    squares of the source and destination images.


    Args:
        points: Points to apply the homographies to shape (B, N, 2) or (B, H, W, 2).
        homographies: Homographies to apply to points shape (B, 3, 3).

    Returns:
        The transformed points shape (B, N, 2) or (B, H, W, 2).

    """
    B, N, K2, _ = points.shape
    B2, _, _ = homographies.shape

    if B != B2 and B2 != 1 and B != 1:
        raise ValueError(
            f"Batch size of points {B} and homographies {B2} do not match and broadcast is not possible."
        )

    if B == 1:
        points_repeated = points.repeat(B2, 1, 1, 1)
    else:
        points_repeated = points

    points_transformed = transform_points(homographies, points_repeated)

    return points_transformed


@torch.jit.script
def denormalize_homography(
    homographies: Tensor, start_HW: tuple[int, int], final_HW: tuple[int, int]
) -> Tensor:
    r"""
    Denormalize a given homography defined over [-1, 1] to be in terms of image sizes.

    Args:
        homographies: homography matrices to denormalize of shape :math:`(B, 3, 3)`.
        start_HW: source image size tuple :math:`(H, W)`.
        final_HW: destination image size tuple :math:`(H, W)`.

    """
    if not isinstance(homographies, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(homographies)}")

    if not (len(homographies.shape) == 3 or homographies.shape[-2:] == (3, 3)):
        raise ValueError(
            f"Input homographies must be a Bx3x3 tensor. Got {homographies.shape}"
        )

    # source and destination sizes
    src_h, src_w = start_HW
    dst_h, dst_w = final_HW

    # (B, 3, 3) source unto [0, 1]
    src_to_square = torch.tensor(
        [
            [2.0 / (src_w - 1), 0.0, -1.0],
            [0.0, 2.0 / (src_h - 1), -1.0],
            [0.0, 0.0, 1.0],
        ],
        device=homographies.device,
        dtype=homographies.dtype,
    )[None]

    # (B, 3, 3) destination unto [0, 1]
    dst_to_square = torch.tensor(
        [
            [2.0 / (dst_w - 1), 0.0, -1.0],
            [0.0, 2.0 / (dst_h - 1), -1.0],
            [0.0, 0.0, 1.0],
        ],
        device=homographies.device,
        dtype=homographies.dtype,
    )[None]

    square_to_dst = torch.inverse(dst_to_square)

    # compute the denormed homography from source to destination
    homographies_denormed = square_to_dst @ (homographies @ src_to_square)
    # make sure the bottom right element is 1.0
    homographies_denormed = homographies_denormed / homographies_denormed[:, 2:3, 2:3]
    return homographies_denormed


class LieAlgebraHomographies(Module):
    """

    Module for differentiable homography estimation using Lie algebra vectors. The additive Lie
    algebra basis vectors are linearly combined according to the the internal parameter weighted
    by the weights parameter. The resulting matrix exponential is the homography. This module is
    meant to be used with a gradient-free optimizer.

    """

    def __init__(
        self,
        dtype_cast_to: torch.dtype = torch.float64,
        dtype_out: torch.dtype = torch.float32,
        x_translation_weight: float = 1.0,
        y_translation_weight: float = 1.0,
        rotation_weight: float = 1.0,
        isotropic_scale_weight: float = 1.0,
        anisotropic_stretch_weight: float = 1.0,
        shear_weight: float = 1.0,
        x_keystone_weight: float = 1.0,
        y_keystone_weight: float = 1.0,
    ):
        super(LieAlgebraHomographies, self).__init__()

        self.dtype_cast_to = dtype_cast_to
        self.dtype_out = dtype_out

        weights = torch.zeros((8,), dtype=dtype_cast_to)
        weights[0] = x_translation_weight
        weights[1] = y_translation_weight
        weights[2] = rotation_weight
        weights[3] = isotropic_scale_weight
        weights[4] = anisotropic_stretch_weight
        weights[5] = shear_weight
        weights[6] = x_keystone_weight
        weights[7] = y_keystone_weight
        self.register_buffer("weights", weights[None, :, None, None])

        elements = torch.zeros((8, 3, 3), dtype=dtype_cast_to)
        elements[0, 0, 2] = 1  # translation in x
        elements[1, 1, 2] = 1  # translation in y
        elements[2, 0, 1] = -1  # rotation
        elements[2, 1, 0] = 1  # rotation
        elements[3, 0, 0] = 1  # isotropic scaling
        elements[3, 1, 1] = 1  # isotropic scaling
        elements[3, 2, 2] = -2  # isotropic scaling
        elements[4, 0, 0] = 1  # stretching
        elements[4, 1, 1] = -1  # stretching
        elements[5, 0, 1] = 1  # shear
        elements[5, 1, 0] = 1  # shear
        elements[6, 2, 0] = (
            1  # projective keystone in x (I might have these swapped for x/y)
        )
        elements[7, 2, 1] = (
            1  # projective keystone in y (I might have these swapped for x/y)
        )
        self.register_buffer("elements", elements)

    def forward(self, lie_vectors) -> Tensor:
        """
        Convert a batch of Lie algebra vectors to Lie group elements (homographies).

        Returns:
            The homographies shape (B, 3, 3).

        """
        # lie_vectors is shape (B, 8)
        # elements is shape (8, 3, 3)
        # weights is shape (1, 8, 1, 1)
        homographies = torch.linalg.matrix_exp(
            (
                lie_vectors[:, :, None, None].to(self.dtype_cast_to)
                * self.elements
                * self.weights
            ).sum(dim=1)
        )
        # make sure the homographies are normalized (bottom right element is 1.0)
        homographies = homographies / homographies[:, 2:3, 2:3]
        return homographies.to(self.dtype_out)

    def split_homography(self, lie_vectors: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Split a homography into its intrinsic, rotation and translation components.

        Args:
            homography: Homographys shaped (B, 3, 3) defined as source to destination.

        Returns:
            Sqrt_H, InvSqrt_H: The half forward and half inverse homographies.

        """
        log_h = (
            lie_vectors[:, :, None, None].to(self.dtype_cast_to)
            * self.elements
            * self.weights
        ).sum(dim=1)
        half_forward = torch.linalg.matrix_exp(0.5 * log_h)
        half_forward = half_forward / half_forward[:, 2:3, 2:3]
        half_inverse = torch.linalg.matrix_exp(-0.5 * log_h)
        half_inverse = half_inverse / half_inverse[:, 2:3, 2:3]
        return half_forward, half_inverse


@torch.jit.script
def generate_random_patch_coordinates(
    patch_size: Tuple[int, int],
    image_size: Tuple[int, int],
    n_patches: int,
    device: torch.device,
) -> Tensor:
    """
    This function generates random normalized patch coordinates within [-1, 1].

    Args:
        patch_size (Tuple[int, int]): Size of the patches to use in the grid
        image_size (Tuple[int, int]): Size of the image
        n_patches (int): Number of patches to generate

    Returns:
        Tensor: Tensor of shape (n_patches, 2) with patch coordinates
    """

    patch_h_normed = float(patch_size[0]) / float(image_size[0])
    patch_w_normed = float(patch_size[1]) / float(image_size[1])

    # generate random patch start coordinates in range [-1, 1-patch_x_normed]
    patch_coords = (
        torch.rand(n_patches, 2, device=device)
        * (2.0 - torch.tensor([patch_h_normed, patch_w_normed], device=device))
        - 1.0
    )

    # generate local patch coordinates via meshgrid
    local_grid = (
        torch.stack(
            torch.meshgrid(
                torch.arange(patch_size[0], device=device) / float(image_size[0]),
                torch.arange(patch_size[1], device=device) / float(image_size[1]),
                indexing="ij",
            ),
            dim=-1,
        )
        .float()
        .view(-1, 2)
    ) * 2.0

    # add local patch coordinates to patch start coordinates
    return patch_coords[:, None, :] + local_grid[None, :, :]


@torch.jit.script
def misorientation_error(img_source: Tensor, img_target: Tensor) -> Tensor:
    """
    Compute the misorientation error between two images.

    Args:
        img_source: Source image of shape (B, C, H, W)
        img_target: Target image of shape (B, C, H, W)

    Returns:
        The misorientation error between the two images.

    """
    # Convert image shape from cmaes to quats
    # (POP, NPATCH, P_H * P_W, C) -> (POP, C, NPATCH, P_H * P_W)
    misorientation = quats.disori_angle_laue(img_source, img_target, 11, 11)
    # img_source = img_source.permute(0, 3, 1, 2)
    # img_target = img_target.permute(0, 3, 1, 2)
    # misorientation = quats.misorientation_gpu(img_source, img_target)
    return misorientation.view((img_source.shape[0], -1)).mean(dim=-1)
    


def lcca_cmaes_homography(
    n_iterations: int,
    cmaes_population: int,
    cmaes_sigma: float,
    n_patches: int,  # few thousand
    img_source: Tensor,
    img_target: Tensor,
    patch_size: Tuple[int, int] = (5, 5),
    guess_lie: Optional[Tensor] = None,
    x_translation_weight: float = 1.0,
    y_translation_weight: float = 1.0,
    rotation_weight: float = 1.0,
    isotropic_scale_weight: float = 1.0,
    anisotropic_stretch_weight: float = 1.0,
    shear_weight: float = 1.0,
    x_keystone_weight: float = 1.0,  # set to zero for affine
    y_keystone_weight: float = 1.0,  # set to zero for affine, determines if lines need to be parallel
    loss: str = "misorientation",
    verbose: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    This function fits a homography between two images using CMA-ES and a grid of patches
    to use for CCA calculations.

    Args:
        patch_size (int): Size of the patches to use in the grid
        n_iterations (int): Number of iterations of CMA-ES to run
        cmaes_population (int): Population size for CMA-ES
        cmaes_sigma (float): Sigma for CMA-ES
        sample_fraction (float): Fraction of the grid to sample
        img_source (Tensor): Source image of shape (B, C, H, W)
        img_target (Tensor): Target image of shape (B, C, H, W)

    """
    _, C, H, W = img_source.shape

    if guess_lie is None:
        guess_lie = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            device=img_source.device,
        )
    else:
        # if its a 1x8 vector, then make it a 8 vector
        if len(guess_lie.shape) > 1:
            guess_lie = guess_lie.squeeze(0)

    # wrapper for CMA-ES optimizer
    cmaes_wrap = CMAESWrapper(
        device=img_source.device,
        dimension=8,
        population_size=cmaes_population,
        mean=guess_lie,
        sigma=cmaes_sigma,
        minimize=False,
    )

    # class for homography
    lie_homog = LieAlgebraHomographies(
        x_translation_weight=x_translation_weight,
        y_translation_weight=y_translation_weight,
        rotation_weight=rotation_weight,
        isotropic_scale_weight=isotropic_scale_weight,
        anisotropic_stretch_weight=anisotropic_stretch_weight,
        shear_weight=shear_weight,
        x_keystone_weight=x_keystone_weight,
        y_keystone_weight=y_keystone_weight,
    ).to(img_source.device)

    # need dimensionless shape of the kernel in the normed [-1, 1]^2 canvas
    # that is halfway between the source and target shapes
    medial_shape = torch.tensor(
        [
            int(0.5 * (img_source.shape[-2] + img_target.shape[-2])),
            int(0.5 * (img_source.shape[-1] + img_target.shape[-1])),
        ],
        device=img_source.device,
    )

    # for the grid sampling we will have (POP, NPATCH, P_H * P_W, 2) grids
    # and we want to interpolate the patches from the source and target images
    # which are each shaped (1, C, H, W) ordinarily you would repeat each image
    # to (POP, C, H, W) but that's not a good idea for large images or pops.
    # Instead we factorize the grid: (POP, NPATCH, P_H * P_W, 2) -> (1, faux_H, faux_W, 2)
    # where faux_H * faux_W = POP * NPATCH * P_H * P_W
    # then we do the interpolation and reshape/permute:
    # (1, C, faux_H, faux_W) -> (C, POP, NPATCH, P_H * P_W) -> (POP, NPATCH, P_H * P_W, C)
    # here we factorize NPATCH * P_H * P_W into as close of a square as possible
    faux_H, faux_W = nearly_square_factors(
        cmaes_population * n_patches * patch_size[0] * patch_size[1]
    )

    for iter in range(n_iterations):
        # grab random normalized image patch coordinates
        patch_coords = generate_random_patch_coordinates(
            patch_size=patch_size,
            image_size=medial_shape,
            # image_size=img_target.shape[-2:],
            n_patches=n_patches,
            device=img_source.device,
        )

        # ask for candidate solutions
        candidates = cmaes_wrap.ask()
        # half_forward: source to the medial canvas and medial canvas to target
        # half_inverse: target to the medial canvas and medial canvas to source
        half_forward, half_inverse = lie_homog.split_homography(candidates)

        # the homographies will remain normalized the entire time
        points_in_src_canvas = normed_homographies_on_points(
            patch_coords[None],
            half_inverse.float(),
        )
        points_in_tgt_canvas = normed_homographies_on_points(
            patch_coords[None],
            half_forward.float(),
        )

        # reshape the points to (1, faux_H, faux_W, 2)
        points_in_src_canvas = points_in_src_canvas.view(1, faux_H, faux_W, 2)
        points_in_tgt_canvas = points_in_tgt_canvas.view(1, faux_H, faux_W, 2)

        # sample the patches from the images
        patches_source = torch.nn.functional.grid_sample(
            img_source, points_in_src_canvas, align_corners=True
        )

        patches_target = torch.nn.functional.grid_sample(
            img_target, points_in_tgt_canvas, align_corners=True
        )

        # reshape and permute the patches to (POP, NPATCH, P_H * P_W, C)
        patches_source = patches_source.view(
            C, cmaes_population, n_patches, patch_size[0] * patch_size[1]
        ).permute(1, 2, 3, 0)
        patches_target = patches_target.view(
            C, cmaes_population, n_patches, patch_size[0] * patch_size[1]
        ).permute(1, 2, 3, 0)

        # make the patches contiguous
        patches_source = patches_source.contiguous()
        patches_target = patches_target.contiguous()

        if loss == "misorientation":
            # compute the misorientation error means over patches
            fitnesses = misorientation_error(patches_source, patches_target)
        elif loss == "cca":
            # compute the CCA score means over patches
            fitnesses = batch_cca(patches_source, patches_target).mean(dim=-1)

        # tell the CMA-ES optimizer the fitnesses
        cmaes_wrap.tell(candidates, fitnesses)

        # print out details
        if verbose:
            print(f"Iteration: {iter} Fitness: {cmaes_wrap.get_best_fitness()}")

    best_lie_solution = cmaes_wrap.get_best_solution()

    best_homography = denormalize_homography(
        lie_homog(best_lie_solution), img_source.shape[-2:], img_target.shape[-2:]
    )

    return best_homography, best_lie_solution
