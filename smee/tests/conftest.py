import openff.interchange
import openff.toolkit
import openff.units
import pytest
import torch


@pytest.fixture(scope="module")
def default_force_field() -> openff.toolkit.ForceField:
    """Returns the OpenFF 1.3.0 force field with constraints removed."""

    force_field = openff.toolkit.ForceField("openff-1.3.0.offxml")
    force_field.deregister_parameter_handler("Constraints")

    return force_field


@pytest.fixture(scope="module")
def ethanol() -> openff.toolkit.Molecule:
    """Returns an OpenFF ethanol molecule with a fixed atom order."""

    return openff.toolkit.Molecule.from_mapped_smiles(
        "[H:5][C:2]([H:6])([H:7])[C:3]([H:8])([H:9])[O:1][H:4]"
    )


@pytest.fixture(scope="module")
def ethanol_conformer(ethanol) -> torch.Tensor:
    """Returns a conformer [Å] of ethanol with an ordering which matches the
    ``ethanol`` fixture."""

    ethanol.generate_conformers(n_conformers=1)
    conformer = ethanol.conformers[0].m_as(openff.units.unit.angstrom)

    return torch.from_numpy(conformer)


@pytest.fixture(scope="module")
def ethanol_interchange(ethanol, default_force_field) -> openff.interchange.Interchange:
    """Returns a parameterized system of ethanol."""

    return openff.interchange.Interchange.from_smirnoff(
        default_force_field, ethanol.to_topology()
    )


@pytest.fixture(scope="module")
def formaldehyde() -> openff.toolkit.Molecule:
    """Returns an OpenFF formaldehyde molecule with a fixed atom order."""

    return openff.toolkit.Molecule.from_mapped_smiles("[H:3][C:1](=[O:2])[H:4]")


@pytest.fixture(scope="module")
def formaldehyde_conformer(formaldehyde) -> torch.Tensor:
    """Returns a conformer [Å] of formaldehyde with an ordering which matches the
    ``formaldehyde`` fixture."""

    formaldehyde.generate_conformers(n_conformers=1)
    conformer = formaldehyde.conformers[0].m_as(openff.units.unit.angstrom)

    return torch.from_numpy(conformer)


@pytest.fixture(scope="module")
def formaldehyde_interchange(
    formaldehyde, default_force_field
) -> openff.interchange.Interchange:
    """Returns a parameterized system of formaldehyde."""

    return openff.interchange.Interchange.from_smirnoff(
        default_force_field, formaldehyde.to_topology()
    )
