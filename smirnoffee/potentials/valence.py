import torch

from smirnoffee.potentials import potential_energy_function

_EPSILON = 1.0e-8


@potential_energy_function("Bonds", "k/2*(r-length)**2")
def evaluate_harmonic_bond_energy(
    conformer: torch.Tensor,
    atom_indices: torch.Tensor,
    parameters: torch.Tensor,
) -> torch.Tensor:
    """Evaluates the potential energy [kJ / mol] of a set of bonds for a given conformer
    using a harmonic potential of the form:

    `1/2 * k * (r - length) ** 2`

    Args:
        conformer: The conformer to evaluate the potential at.
        atom_indices: The indices of the atoms involved in each bond with
            shape=(n_bonds, 4).
        parameters: A tensor with shape=(n_bonds, 2) where there first column
            contains the force constants ``k``, and the second the equilibrium
            bond ``length``.

    Returns:
        The evaluated potential energy [kJ / mol].
    """

    if len(atom_indices) == 0:
        return torch.zeros(1)

    distances = torch.norm(
        conformer[atom_indices[:, 1]] - conformer[atom_indices[:, 0]], dim=1
    )

    return (0.5 * parameters[:, 0] * (distances - parameters[:, 1]) ** 2).sum()


@potential_energy_function("Angles", "k/2*(theta-angle)**2")
def evaluate_harmonic_angle_energy(
    conformer: torch.Tensor,
    atom_indices: torch.Tensor,
    parameters: torch.Tensor,
) -> torch.Tensor:
    """Evaluates the potential energy [kJ / mol] of a set of valence angles
    for a given conformer using a harmonic potential of the form:

    `1/2 * k * (theta - angle) ** 2`

    Args:
        conformer: The conformer to evaluate the potential at.
        atom_indices: The indices of the atoms involved in each valence angle with
            shape=(n_angles, 4).
        parameters: A tensor with shape=(n_angles, 2) where there first column
            contains the force constants ``k``, and the second the equilibrium
            ``angle``.

    Returns:
        The evaluated potential energy [kJ / mol].
    """

    if len(atom_indices) == 0:
        return torch.zeros(1)

    vector_ab = conformer[atom_indices[:, 1]] - conformer[atom_indices[:, 0]]
    vector_ab = vector_ab / torch.norm(vector_ab, dim=1).unsqueeze(1)

    vector_ac = conformer[atom_indices[:, 1]] - conformer[atom_indices[:, 2]]
    vector_ac = vector_ac / torch.norm(vector_ac, dim=1).unsqueeze(1)

    cos_angle = (vector_ab * vector_ac).sum(dim=1)
    # TODO: properly handle the acos singularity.
    cos_angle = torch.clamp(cos_angle, -1.0 + _EPSILON, 1.0 - _EPSILON)

    angles = torch.rad2deg(torch.acos(cos_angle))

    return (0.5 * parameters[:, 0] * (angles - parameters[:, 1]) ** 2).sum()


def _evaluate_cosine_torsion_energy(
    conformer: torch.Tensor,
    atom_indices: torch.Tensor,
    parameters: torch.Tensor,
) -> torch.Tensor:
    """Evaluates the potential energy [kJ / mol] of a set of torsions
    for a given conformer using a cosine potential of the form:

    `k*(1+cos(periodicity*theta-phase))`

    Args:
        conformer: The conformer to evaluate the potential at.
        atom_indices: The indices of the atoms involved in each proper torsion with
            shape=(n_torsions, 4).
        parameters: A tensor with shape=(n_torsions, 4) where there first column
            contains the force constants ``k``, the second the ``periodicities``,
            the third the ``phase`` and the fourth an ``idivf`` factor to divide the
            force constant by.

    Returns:
        The evaluated potential energy [kJ / mol].
    """

    if len(atom_indices) == 0:
        return torch.zeros(1)

    # Based on the OpenMM formalism.
    vector_ab = conformer[atom_indices[:, 0]] - conformer[atom_indices[:, 1]]
    vector_cb = conformer[atom_indices[:, 2]] - conformer[atom_indices[:, 1]]
    vector_cd = conformer[atom_indices[:, 2]] - conformer[atom_indices[:, 3]]

    vector_ab_cross_cb = torch.cross(vector_ab, vector_cb, dim=1)
    vector_cb_cross_cd = torch.cross(vector_cb, vector_cd, dim=1)

    vector_ab_cross_cb = vector_ab_cross_cb / torch.norm(
        vector_ab_cross_cb, dim=1
    ).unsqueeze(1)
    vector_cb_cross_cd = vector_cb_cross_cd / torch.norm(
        vector_cb_cross_cd, dim=1
    ).unsqueeze(1)

    cos_phi = (vector_ab_cross_cb * vector_cb_cross_cd).sum(dim=1)
    # TODO: properly handle the acos singularity.
    cos_phi = torch.clamp(cos_phi, -1.0 + _EPSILON, 1.0 - _EPSILON)

    phi = torch.acos(cos_phi)
    phi = phi * torch.where((vector_ab * vector_cb_cross_cd).sum(dim=1) < 0, -1.0, 1.0)

    return (
        parameters[:, 0]
        / parameters[:, 3]
        * (1.0 + torch.cos(parameters[:, 1] * phi - torch.deg2rad(parameters[:, 2])))
    ).sum()


@potential_energy_function("ProperTorsions", "k*(1+cos(periodicity*theta-phase))")
def evaluate_cosine_proper_torsion_energy(
    conformer: torch.Tensor,
    atom_indices: torch.Tensor,
    parameters: torch.Tensor,
) -> torch.Tensor:
    """Evaluates the potential energy [kJ / mol] of a set of proper torsions
    for a given conformer using a cosine potential of the form:

    `k*(1+cos(periodicity*theta-phase))`

    Args:
        conformer: The conformer to evaluate the potential at.
        atom_indices: The indices of the atoms involved in each proper torsion with
            shape=(n_propers, 4).
        parameters: A tensor with shape=(n_propers, 4) where there first column
            contains the force constants ``k``, the second the ``periodicities``,
            the third the ``phase`` and the fourth an ``idivf`` factor to divide the
            force constant by.

    Returns:
        The evaluated potential energy [kJ / mol].
    """
    return _evaluate_cosine_torsion_energy(conformer, atom_indices, parameters)


@potential_energy_function("ImproperTorsions", "k*(1+cos(periodicity*theta-phase))")
def evaluate_cosine_improper_torsion_energy(
    conformer: torch.Tensor,
    atom_indices: torch.Tensor,
    parameters: torch.Tensor,
) -> torch.Tensor:
    """Evaluates the potential energy [kJ / mol] of a set of improper torsions
    for a given conformer using a cosine potential of the form:

    `k*(1+cos(periodicity*theta-phase))`

    Args:
        conformer: The conformer to evaluate the potential at.
        atom_indices: The indices of the atoms involved in each improper torsion with
            shape=(n_impropers, 4).
        parameters: A tensor with shape=(n_impropers, 4) where there first column
            contains the force constants ``k``, the second the ``periodicities``,
            the third the ``phase`` and the fourth an ``idivf`` factor to divide the
            force constant by.

    Returns:
        The evaluated potential energy [kJ / mol].
    """
    return _evaluate_cosine_torsion_energy(conformer, atom_indices, parameters)
