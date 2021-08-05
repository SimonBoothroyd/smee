import pytest
import torch
from openff.interchange.components.interchange import Interchange
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField


@pytest.fixture(scope="module")
def default_force_field() -> ForceField:
    """Returns the OpenFF 1.3.0 force field with constraints removed."""

    force_field = ForceField("openff-1.0.0.offxml")
    # force_field.deregister_parameter_handler("ToolkitAM1BCC")
    force_field.deregister_parameter_handler("Constraints")

    return force_field


@pytest.fixture(scope="module")
def ethanol() -> Molecule:
    """Returns an OpenFF ethanol molecule with a fixed atom order."""

    return Molecule.from_mapped_smiles(
        "[H:5][C:2]([H:6])([H:7])[C:3]([H:8])([H:9])[O:1][H:4]"
    )


@pytest.fixture(scope="module")
def ethanol_conformer(ethanol) -> torch.Tensor:
    """Returns a conformer [A] of ethanol with an ordering which matches the
    ``ethanol`` fixture."""

    from simtk import unit as simtk_unit

    ethanol.generate_conformers(n_conformers=1)
    conformer = ethanol.conformers[0].value_in_unit(simtk_unit.angstrom)

    return torch.from_numpy(conformer)


@pytest.fixture(scope="module")
def ethanol_system(ethanol, default_force_field) -> Interchange:
    """Returns a parametermized system of ethanol."""

    return Interchange.from_smirnoff(default_force_field, ethanol.to_topology())


@pytest.fixture(scope="module")
def formaldehyde() -> Molecule:
    """Returns an OpenFF formaldehyde molecule with a fixed atom order.."""

    molecule: Molecule = Molecule.from_smiles("C=O")
    return molecule.canonical_order_atoms()


@pytest.fixture(scope="module")
def formaldehyde_conformer(formaldehyde) -> torch.Tensor:
    """Returns a conformer [A] of formaldehyde with an ordering which matches the
    ``formaldehyde`` fixture."""

    from simtk import unit as simtk_unit

    formaldehyde.generate_conformers(n_conformers=1)
    conformer = formaldehyde.conformers[0].value_in_unit(simtk_unit.angstrom)

    return torch.from_numpy(conformer)


@pytest.fixture(scope="module")
def formaldehyde_system(formaldehyde, default_force_field) -> Interchange:
    """Returns a parametermized system of formaldehyde."""

    return Interchange.from_smirnoff(default_force_field, formaldehyde.to_topology())
