SMIRNOFF Energy Evaluations
===========================
[![Test Status](https://github.com/simonboothroyd/smirnoffee/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/simonboothroyd/smirnoffee/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/simonboothroyd/smirnoffee/branch/main/graph/badge.svg)](https://codecov.io/gh/simonboothroyd/smirnoffee/branch/main)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The `smirnoffee` framework aims to offer a simple API for differentiably evaluating the energy of [SMIRNOFF](https://openforcefield.github.io/standards/standards/smirnoff/) 
force fields applied to molecules using `pytorch`.

The package currently supports evaluating the energy of force fields that contain: 

* `Bonds`, `Angles`, `ProperTorsions` and `ImproperTorsions` 
* `vdW`, `Electrostatics`, `ToolkitAM1BCC`

parameter handlers.

Force fields that apply virtual sites will likely be supported soon.

***Warning**: This code is currently experimental and under active development. If you are using this it, please be 
aware that it is not guaranteed to provide correct results, the documentation and testing maybe be incomplete, and the
API can change without notice.*

## Installation

A development conda environment can be created and activated by running:

```shell
make env
conda activate smirnoffee
```

The environment will include all development dependencies, including linters and testing apparatus.

## Getting Started

To get started, we will show how the energy of paracetamol in a particular conformer can be evaluated using `smirnoffee` 
framework. 

We start by loading in the molecule of interest, in this case paracetamol as defined by its SMILES representation,
using the [`openff-toolkit`](https://github.com/openforcefield/openff-toolkit) and generating a single conformer for it:

```python
import openff.toolkit
import openff.units
import torch

molecule = openff.toolkit.Molecule.from_smiles("CC(=O)NC1=CC=C(C=C1)O")
molecule.generate_conformers(n_conformers=1)

conformer = torch.tensor(molecule.conformers[0].m_as(openff.units.unit.angstrom))
```

Next we will load in the force field that encodes the potential energy function we wish to evaluate
and apply it to our molecule.

```python
import openff.interchange

force_field = openff.toolkit.ForceField(
    "openff_unconstrained-2.0.0.offxml"
)
interchange = openff.interchange.Interchange.from_smirnoff(
    force_field, molecule.to_topology()
)
```

In order to use an interchange object with this framework, we need to map it into a collection of tensors:

```python
from smirnoffee.ff.smirnoff import convert_interchange
force_field, [applied_parameters] = convert_interchange(interchange)
```

**Note:** The `convert_interchange` function can take either a single interchange object, or a list of multiple.

These tensors are returned as::

* a `smirnoffee.ff.TensorForceField`: this object stores the original values of the force field parameters.
* a list of ``smirnoffee.ff.AppliedParameters``: each object will store a map for every handler, that specifies which 
  parameters were assigned to which element (e.g. bond, angle, etc).

Storing the parameter values separately from how they should be applied allows us to easily modify the values of the
parameters and re-evaluate the energy using those parameters.

The energy of the molecule can then be directly be evaluated:

```python
from smirnoffee.potentials import evaluate_energy
energy = evaluate_energy(applied_parameters, conformer, force_field)

print(f"Energy = {energy.item():.3f} kJ / mol")
```

> Energy = 137.622 kJ / mol

Here we have provided a single conformer, but multiple can be batched together by stacking them along the first axis for
faster evaluation.

## License

The main package is release under the [MIT license](LICENSE). 

## Copyright

Copyright (c) 2023, Simon Boothroyd
