import pytest
import torch
from openff.system.components.system import System
from openff.system.stubs import ForceField
from openff.toolkit.topology import Molecule


@pytest.fixture(scope="module")
def default_force_field() -> ForceField:
    """Returns the OpenFF 1.3.0 force field with constraints removed."""

    force_field = ForceField("openff-1.0.0.offxml")
    # force_field.deregister_parameter_handler("ToolkitAM1BCC")
    force_field.deregister_parameter_handler("Constraints")

    return force_field


@pytest.fixture(scope="module")
def ethanol() -> Molecule:
    """Returns an OpenFF ethanol molecule."""

    molecule: Molecule = Molecule.from_smiles("CCO")
    return molecule.canonical_order_atoms()


@pytest.fixture(scope="module")
def ethanol_conformer(ethanol) -> torch.Tensor:
    """Returns a conformer [A] of ethanol with an ordering which matches the
    ``ethanol`` fixture."""

    from simtk import unit as simtk_unit

    ethanol.generate_conformers(n_conformers=1)
    conformer = ethanol.conformers[0].value_in_unit(simtk_unit.angstrom)

    return torch.from_numpy(conformer)


@pytest.fixture(scope="module")
def ethanol_system(ethanol, default_force_field) -> System:
    """Returns a parametermized system of ethanol."""

    return default_force_field.create_openff_system(ethanol.to_topology())


@pytest.fixture(scope="module")
def formaldehyde() -> Molecule:
    """Returns an OpenFF formaldehyde molecule."""

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
def formaldehyde_system(formaldehyde, default_force_field) -> System:
    """Returns a parametermized system of formaldehyde."""

    return default_force_field.create_openff_system(formaldehyde.to_topology())
