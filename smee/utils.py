"""General utility functions"""
import typing

import networkx
import openff.toolkit
import torch

if typing.TYPE_CHECKING:
    import smee.ff


_size = torch.Size | list[int] | tuple[int, ...]


def find_exclusions(
    topology: openff.toolkit.Topology,
    v_sites: typing.Optional["smee.ff.VSiteMap"] = None,
) -> dict[
    tuple[int, int], typing.Literal["scale_12", "scale_13", "scale_14", "scale_15"]
]:
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
