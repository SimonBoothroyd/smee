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

    if v_sites is not None:
        for v_site_key in v_sites.keys:
            v_site_idx = v_sites.key_to_idx[v_site_key]
            parent_idx = v_site_key.orientation_atom_indices[0]

            for neighbour_idx in graph.neighbors(parent_idx):
                graph.add_edge(v_site_idx, neighbour_idx)

            graph.add_edge(v_site_idx, parent_idx)

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

    return exclusions


def ones_like(size: _size, other: torch.Tensor) -> torch.Tensor:
    """Create a tensor of ones with the same device and type as another tensor."""
    return torch.ones(size, dtype=other.dtype, device=other.device)


def zeros_like(size: _size, other: torch.Tensor) -> torch.Tensor:
    """Create a tensor of zeros with the same device and type as another tensor."""
    return torch.zeros(size, dtype=other.dtype, device=other.device)


def tensor_like(data: typing.Any, other: torch.Tensor) -> torch.Tensor:
    """Create a tensor with the same device and type as another tensor."""

    if isinstance(data, torch.Tensor):
        return data.clone().detach().to(other.device, other.dtype)

    return torch.tensor(data, dtype=other.dtype, device=other.device)


def arange_like(end: int, other: torch.Tensor) -> torch.Tensor:
    """Arange a tensor with the same device and type as another tensor."""
    return torch.arange(end, dtype=other.dtype, device=other.device)


def logsumexp(
    a: torch.Tensor,
    dim: int,
    keepdim: bool = False,
    b: torch.Tensor | None = None,
    return_sign: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Compute the log of the sum of the exponential of the input elements, optionally
    with each element multiplied by a scaling factor.

    Notes:
        This should be removed if torch.logsumexp is updated to support scaling factors.

    Args:
        a: The elements that should be summed over.
        dim: The dimension to sum over.
        keepdim: Whether to keep the summed dimension.
        b: The scaling factor to multiply each element by.

    Returns:
        The log of the sum of exponential of the a elements.
    """
    a_type = a.dtype

    if b is None:
        assert return_sign is False
        return torch.logsumexp(a, dim, keepdim)

    a = a.double()
    b = b if b is not None else b.double()

    a, b = torch.broadcast_tensors(a, b)

    if torch.any(b == 0):
        a[b == 0] = -torch.inf

    a_max = torch.amax(a, dim=dim, keepdim=True)

    if a_max.ndim > 0:
        a_max[~torch.isfinite(a_max)] = 0
    elif not torch.isfinite(a_max):
        a_max = 0

    exp_sum = torch.sum(b * torch.exp(a - a_max), dim=dim, keepdim=keepdim)
    sign = None

    if return_sign:
        sign = torch.sign(exp_sum)
        exp_sum = exp_sum * sign

    ln_exp_sum = torch.log(exp_sum)

    if not keepdim:
        a_max = torch.squeeze(a_max, dim=dim)

    ln_exp_sum += a_max
    ln_exp_sum = ln_exp_sum.to(a_type)

    if return_sign:
        return ln_exp_sum, sign.to(a_type)
    else:
        return ln_exp_sum


def to_upper_tri_idx(
    i: torch.Tensor, j: torch.Tensor, n: int, include_diag: bool = False
) -> torch.Tensor:
    """Converts pairs of 2D indices to 1D indices in an upper triangular matrix that
    excludes the diagonal.

    Args:
        i: A tensor of the indices along the first axis with ``shape=(n_pairs,)``.
        j: A tensor of the indices along the second axis with ``shape=(n_pairs,)``.
        n: The size of the matrix.
        include_diag: Whether the diagonal is included in the upper triangular matrix.

    Returns:
        A tensor of the indices in the upper triangular matrix with
        ``shape=(n_pairs * (n_pairs - 1) // 2,)``.
    """

    if not include_diag:
        assert (i < j).all(), "i must be less than j"
        return (i * (2 * n - i - 1)) // 2 + j - i - 1

    assert (i <= j).all(), "i must be less than or equal to j"
    return (i * (2 * n - i + 1)) // 2 + j - i


class _SafeGeometricMean(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(eps_a, eps_b):
        return torch.sqrt(eps_a * eps_b)

    @staticmethod
    def setup_context(ctx, inputs, output):
        eps_a, eps_b = inputs
        eps = output
        ctx.save_for_backward(eps_a, eps_b, eps)
        ctx.save_for_forward(eps_a, eps_b, eps)

    @staticmethod
    def backward(ctx, grad_output):
        eps_a, eps_b, eps = ctx.saved_tensors

        eps = torch.where(eps == 0.0, EPSILON, eps)

        grad_eps_a = grad_output * eps_b / (2 * eps)
        grad_eps_b = grad_output * eps_a / (2 * eps)

        return grad_eps_a, grad_eps_b

    @staticmethod
    def jvp(ctx, *grad_inputs):
        eps_a, eps_b, eps = ctx.saved_tensors

        eps = torch.where(eps == 0.0, EPSILON, eps)

        grad_eps_a = grad_inputs[0] * eps_b / (2 * eps)
        grad_eps_b = grad_inputs[1] * eps_a / (2 * eps)

        return grad_eps_a + grad_eps_b


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
