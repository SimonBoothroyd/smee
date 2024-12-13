"""Compute internal coordinates (e.g. bond lengths)."""

import typing

import torch

import smee.utils

if typing.TYPE_CHECKING:
    import smee


V_SITE_TYPE_TO_FRAME = {
    "BondCharge": torch.tensor(
        [[1.0, 0.0], [-1.0, 1.0], [-1.0, 1.0]], dtype=torch.float64
    ),
    "MonovalentLonePair": torch.tensor(
        [[1.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]], dtype=torch.float64
    ),
    "DivalentLonePair": torch.tensor(
        [[1.0, 0.0, 0.0], [-1.0, 0.5, 0.5], [-1.0, 1.0, 0.0]], dtype=torch.float64
    ),
    "TrivalentLonePair": torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [-1.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    ),
}


def compute_bond_vectors(
    conformer: torch.Tensor, atom_indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the vectors between each atom pair specified by the ``atom_indices`` as
    well as their norms.

    Args:
        conformer: The conformer [Å] to compute the bond vectors for with
            ``shape=(n_atoms, 3)`` or ``shape=(n_confs, n_atoms, 3)``.
        atom_indices: The indices of the atoms involved in each bond with
            ``shape=(n_bonds, 2)``

    Returns:
        The bond vectors and their norms [Å].
    """

    if len(atom_indices) == 0:
        return (
            smee.utils.tensor_like([], other=conformer),
            smee.utils.tensor_like([], other=conformer),
        )

    is_batched = conformer.ndim == 3

    if not is_batched:
        conformer = torch.unsqueeze(conformer, 0)

    directions = conformer[:, atom_indices[:, 1]] - conformer[:, atom_indices[:, 0]]
    distances = torch.norm(directions, dim=-1)

    if not is_batched:
        directions = torch.squeeze(directions, dim=0)
        distances = torch.squeeze(distances, dim=0)

    return directions, distances


def compute_angles(conformer: torch.Tensor, atom_indices: torch.Tensor) -> torch.Tensor:
    """Computes the angles [rad] between each atom triplet specified by the
    ``atom_indices``.

    Args:
        conformer: The conformer [Å] to compute the angles for with
            ``shape=(n_atoms, 3)`` or ``shape=(n_confs, n_atoms, 3)``.
        atom_indices: The indices of the atoms involved in each angle with
            ``shape=(n_angles, 3)``.

    Returns:
        The valence angles [rad].
    """

    if len(atom_indices) == 0:
        return smee.utils.tensor_like([], other=conformer)

    is_batched = conformer.ndim == 3

    if not is_batched:
        conformer = torch.unsqueeze(conformer, 0)

    vector_ab = conformer[:, atom_indices[:, 1]] - conformer[:, atom_indices[:, 0]]
    vector_ac = conformer[:, atom_indices[:, 1]] - conformer[:, atom_indices[:, 2]]

    # tan theta = sin theta / cos theta
    #
    # ||a x b|| = ||a|| ||b|| sin theta
    #   a . b   = ||a|| ||b|| cos theta
    #
    # => tan theta = (a x b) / (a . b)
    angles = torch.atan2(
        torch.norm(torch.cross(vector_ab, vector_ac, dim=-1), dim=-1),
        (vector_ab * vector_ac).sum(dim=-1),
    )

    if not is_batched:
        angles = torch.squeeze(angles, dim=0)

    return angles


def compute_dihedrals(
    conformer: torch.Tensor, atom_indices: torch.Tensor
) -> torch.Tensor:
    """Computes the dihedral angles [rad] between each atom quartet specified by the
    ``atom_indices``.

    Args:
        conformer: The conformer [Å] to compute the dihedral angles for with
            ``shape=(n_atoms, 3)`` or ``shape=(n_confs, n_atoms, 3)``.
        atom_indices: The indices of the atoms involved in each dihedral angle with
            ``shape=(n_dihedrals, 4)``.

    Returns:
        The dihedral angles [rad].
    """

    if len(atom_indices) == 0:
        return smee.utils.tensor_like([], other=conformer)

    is_batched = conformer.ndim == 3

    if not is_batched:
        conformer = torch.unsqueeze(conformer, 0)

    # Based on the OpenMM formalism.
    vector_ab = conformer[:, atom_indices[:, 0]] - conformer[:, atom_indices[:, 1]]
    vector_cb = conformer[:, atom_indices[:, 2]] - conformer[:, atom_indices[:, 1]]
    vector_cd = conformer[:, atom_indices[:, 2]] - conformer[:, atom_indices[:, 3]]

    vector_ab_cross_cb = torch.cross(vector_ab, vector_cb, dim=-1)
    vector_cb_cross_cd = torch.cross(vector_cb, vector_cd, dim=-1)

    vector_cb_norm = torch.norm(vector_cb, dim=-1).unsqueeze(-1)

    y = (
        torch.cross(vector_ab_cross_cb, vector_cb_cross_cd, dim=-1)
        * vector_cb
        / vector_cb_norm
    ).sum(axis=-1)

    x = (vector_ab_cross_cb * vector_cb_cross_cd).sum(axis=-1)

    phi = torch.atan2(y, x)

    if not is_batched:
        phi = torch.squeeze(phi, dim=0)

    return phi


def _build_v_site_coord_frames(
    v_sites: "smee.VSiteMap",
    conformer: torch.Tensor,
    force_field: "smee.TensorForceField",
) -> torch.Tensor:
    """Builds an orthonormal coordinate frame for each virtual particle
    based on the type of virtual site and the coordinates of the parent atoms.

    Notes:
        * See `the OpenMM documentation for further information
          <https://docs.openmm.org/7.0.0/userguide/theory.html#virtual-sites>`_.

    Args:
        v_sites: A mapping between the virtual sites to add and their corresponding
            force field parameters.
        conformer: The conformer to add the virtual sites to with
            ``shape=(n_batches, n_atoms, 3)`` and units of [Å].
        force_field: The force field containing the virtual site parameters.

    Returns:
        A tensor storing the local coordinate frames of all virtual sites with
        ``shape=(n_batches, 4, n_v_sites, 3)`` where ``local_frames[0]`` is the origin
        of each frame, ``local_frames[1]`` the x-direction, ``local_frames[2]`` the
        y-directions, and ``local_frames[2]`` the z-direction.
    """

    weights = [force_field.v_sites.weights[idx] for idx in v_sites.parameter_idxs]

    stacked_frames = [[], [], [], []]

    for key, weight in zip(v_sites.keys, weights, strict=True):
        parent_coords = conformer[:, key.orientation_atom_indices, :]
        weighted_coords = torch.transpose(
            (torch.transpose(parent_coords, 1, 2) @ weight.T), 1, 2
        )

        origin = weighted_coords[:, 0, :]

        xy_plane = weighted_coords[:, 1:, :]
        xy_plane_hat = xy_plane / torch.norm(xy_plane, dim=-1).unsqueeze(-1)

        x_hat = xy_plane_hat[:, 0, :]

        z = torch.cross(x_hat, xy_plane_hat[:, 1, :])
        z_norm = torch.norm(z, dim=-1).unsqueeze(-1)
        z_norm_clamped = torch.where(
            torch.isclose(z_norm, smee.utils.tensor_like(0.0, other=z_norm)),
            smee.utils.tensor_like(1.0, other=z_norm),
            z_norm,
        )
        z_hat = z / z_norm_clamped

        y = torch.cross(z_hat, x_hat)
        y_norm = torch.norm(y, dim=-1).unsqueeze(-1)
        y_norm_clamped = torch.where(
            torch.isclose(y_norm, smee.utils.tensor_like(0.0, other=y_norm)),
            smee.utils.tensor_like(1.0, other=y_norm),
            y_norm,
        )
        y_hat = y / y_norm_clamped

        stacked_frames[0].append(origin)
        stacked_frames[1].append(x_hat)
        stacked_frames[2].append(y_hat)
        stacked_frames[3].append(z_hat)

    local_frames = torch.stack(
        [torch.stack(weights, dim=1) for weights in stacked_frames], dim=1
    )

    return local_frames


def polar_to_cartesian_coords(polar_coords: torch.Tensor) -> torch.Tensor:
    """Converts a set of polar coordinates into cartesian coordinates.

    Args:
        polar_coords: The polar coordinates with ``shape=(n_coords, 3)`` and with
            columns of distance [Å], 'in plane angle' [rad] and 'out of plane'
            angle [rad].

    Returns:
        An array of the cartesian coordinates with ``shape=(n_coords, 3)`` and units
        of [Å].
    """

    d, theta, phi = polar_coords[:, 0], polar_coords[:, 1], polar_coords[:, 2]

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)

    # Here we use cos(phi) in place of sin(phi) and sin(phi) in place of cos(phi)
    # this is because we want phi=0 to represent a 0 degree angle from the x-y plane
    # rather than 0 degrees from the z-axis.
    coords = torch.stack(
        [d * cos_theta * cos_phi, d * sin_theta * cos_phi, d * sin_phi], dim=-1
    )
    return coords


def _convert_v_site_coords(
    local_frame_coords: torch.Tensor, local_coord_frames: torch.Tensor
) -> torch.Tensor:
    """Converts a set of local virtual site coordinates defined in a spherical
    coordinate system into a full set of cartesian coordinates.

    Args:
        local_frame_coords: The local coordinates with ``shape=(n_v_sites, 3)`` and
            with columns of distance [Å], 'in-plane angle' [rad] and 'out-of-plane'
            angle [rad].
        local_coord_frames: The orthonormal basis associated with each of the
            virtual sites with ``shape=(n_batches, 4, n_v_sites, 3)``.

    Returns:
        An array of the cartesian coordinates of the virtual sites with
        ``shape=(n_batches, n_v_sites, 3)`` and units of [Å].
    """

    local_frame_coords = polar_to_cartesian_coords(local_frame_coords)

    return (
        local_coord_frames[:, 0]
        + local_frame_coords[:, 0, None] * local_coord_frames[:, 1]
        + local_frame_coords[:, 1, None] * local_coord_frames[:, 2]
        + local_frame_coords[:, 2, None] * local_coord_frames[:, 3]
    )


def compute_v_site_coords(
    v_sites: "smee.VSiteMap",
    conformer: torch.Tensor,
    force_field: "smee.TensorForceField",
) -> torch.Tensor:
    """Computes the positions of a set of virtual sites relative to a specified
    conformer or batch of conformers.

    Args:
        v_sites: A mapping between the virtual sites to add and their corresponding
            force field parameters.
        conformer: The conformer(s) to add the virtual sites to with
            ``shape=(n_atoms, 3)`` or ``shape=(n_batches, n_atoms, 3)`` and units of
            [Å].
        force_field: The force field containing the virtual site parameters.

    Returns:
        A tensor of virtual site positions [Å] with ``shape=(n_v_sites, 3)`` or
        ``shape=(n_batches, n_v_sites, 3)``.
    """

    is_batched = conformer.ndim == 3

    if not is_batched:
        conformer = torch.unsqueeze(conformer, 0)

    if len(v_sites.parameter_idxs) > 0:
        local_frame_coords = force_field.v_sites.parameters[v_sites.parameter_idxs]
        local_coord_frames = _build_v_site_coord_frames(v_sites, conformer, force_field)

        v_site_coords = _convert_v_site_coords(local_frame_coords, local_coord_frames)
    else:
        v_site_coords = smee.utils.zeros_like((len(conformer), 0, 3), other=conformer)

    if not is_batched:
        v_site_coords = torch.squeeze(v_site_coords, 0)

    return v_site_coords


def add_v_site_coords(
    v_sites: "smee.VSiteMap",
    conformer: torch.Tensor,
    force_field: "smee.TensorForceField",
) -> torch.Tensor:
    """Appends the coordinates of any virtual sites to a conformer (or batch of
    conformers) containing only atomic coordinates.

    Notes:
        * This function only supports appending v-sites to the end of the list of
          coordinates, and not interleaving them between existing atomic coordinates.

    Args:
        v_sites: A mapping between the virtual sites to add and their corresponding
            force field parameters.
        conformer: The conformer(s) to add the virtual sites to with
            ``shape=(n_atoms, 3)`` or ``shape=(n_batches, n_atoms, 3)`` and units of
            [Å].
        force_field: The force field containing the virtual site parameters.

    Returns:
        The full conformer(s) with both atomic and virtual site coordinates [Å] with
        ``shape=(n_atoms+n_v_sites, 3)`` or ``shape=(n_batches, n_atoms+n_v_sites, 3)``.
    """

    v_site_coords = compute_v_site_coords(v_sites, conformer, force_field)

    return torch.cat([conformer, v_site_coords], dim=(1 if conformer.ndim == 3 else 0))
