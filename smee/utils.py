"""General utility functions"""
import typing

import networkx
import openff.toolkit
import torch

if typing.TYPE_CHECKING:
    import smee


_size = int | torch.Size | list[int] | tuple[int, ...]

ExclusionType = typing.Literal["scale_12", "scale_13", "scale_14", "scale_15"]


EPSILON = 1.0e-6
"""A small epsilon value used to prevent divide by zero errors."""


def find_exclusions(
    topology: openff.toolkit.Topology,
    v_sites: typing.Optional["smee.VSiteMap"] = None,
) -> dict[tuple[int, int], ExclusionType]:
    """Find all excluded interaction pairs and their associated scaling factor.

    Args:
        topology: The topology to find the interaction pairs of.
        v_sites: Virtual sites that will be added to the topology.

    Returns:
        A dictionary of the form ``{(atom_idx_1, atom_idx_2): scale}``.
    """

    graph = networkx.from_edgelist(
        tuple(
            sorted((topology.atom_index(bond.atom1), topology.atom_index(bond.atom2)))
        )
        for bond in topology.bonds
    )

    distances = dict(networkx.all_pairs_shortest_path_length(graph, cutoff=5))
    distance_to_scale = {1: "scale_12", 2: "scale_13", 3: "scale_14", 4: "scale_15"}

    exclusions = {}

    for idx_a in distances:
        for idx_b, distance in distances[idx_a].items():
            pair = tuple(sorted((idx_a, idx_b)))
            scale = distance_to_scale.get(distance)

            if scale is None:
                continue

            assert pair not in exclusions or exclusions[pair] == scale
            exclusions[pair] = scale

    if v_sites is None:
        return exclusions

    v_site_exclusions = {}

    for v_site_key in v_sites.keys:
        v_site_idx = v_sites.key_to_idx[v_site_key]
        parent_idx = v_site_key.orientation_atom_indices[0]

        v_site_exclusions[(v_site_idx, parent_idx)] = "scale_12"

        for pair, scale in exclusions.items():
            if parent_idx not in pair:
                continue

            if pair[0] == parent_idx:
                v_site_exclusions[(v_site_idx, pair[1])] = scale
            else:
                v_site_exclusions[(pair[0], v_site_idx)] = scale

    return {**exclusions, **v_site_exclusions}


def ones_like(size: _size, other: torch.Tensor) -> torch.Tensor:
    """Create a tensor of ones with the same device and type as another tensor."""
    return torch.ones(size, dtype=other.dtype, device=other.device)


def zeros_like(size: _size, other: torch.Tensor) -> torch.Tensor:
    """Create a tensor of zeros with the same device and type as another tensor."""
    return torch.zeros(size, dtype=other.dtype, device=other.device)


def tensor_like(data: typing.Any, other: torch.Tensor) -> torch.Tensor:
    """Create a tensor with the same device and type as another tensor."""
    return torch.tensor(data, dtype=other.dtype, device=other.device)


def arange_like(end: int, other: torch.Tensor) -> torch.Tensor:
    """Arange a tensor with the same device and type as another tensor."""
    return torch.arange(end, dtype=other.dtype, device=other.device)


def to_upper_tri_idx(i: torch.Tensor, j: torch.Tensor, n: int) -> torch.Tensor:
    """Converts pairs of 2D indices to 1D indices in an upper triangular matrix that
    excludes the diagonal.

    Args:
        i: A tensor of the indices along the first axis with ``shape=(n_pairs,)``.
        j: A tensor of the indices along the second axis with ``shape=(n_pairs,)``.
        n: The size of the matrix.

    Returns:
        A tensor of the indices in the upper triangular matrix with
        ``shape=(n_pairs * (n_pairs - 1) // 2,)``.
    """
    assert (i < j).all(), "i must be less than j"
    return (i * (2 * n - i - 1)) // 2 + j - i - 1


class _SafeGeometricMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, eps_a, eps_b):
        eps = torch.sqrt(eps_a * eps_b)

        ctx.save_for_backward(eps_a, eps_b, eps)
        return torch.sqrt(eps_a * eps_b)

    @staticmethod
    def backward(ctx, grad_output):
        eps_a, eps_b, eps = ctx.saved_tensors

        eps = torch.where(eps == 0.0, EPSILON, eps)

        grad_eps_a = grad_output * eps_b / (2 * eps)
        grad_eps_b = grad_output * eps_a / (2 * eps)

        return grad_eps_a, grad_eps_b


def geometric_mean(eps_a: torch.Tensor, eps_b: torch.Tensor) -> torch.Tensor:
    """Computes the geometric mean of two values 'safely'.

    A small epsilon (``smee.utils.EPSILON``) is added when computing the gradient in
    cases where the mean is zero to prevent divide by zero errors.

    Args:
        eps_a: The first value.
        eps_b: The second value.

    Returns:
        The geometric mean of the two values.
    """

    return _SafeGeometricMean.apply(eps_a, eps_b)
