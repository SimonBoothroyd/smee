SMIRNOFF Energy Evaluations
===========================
[![Test Status](https://github.com/simonboothroyd/smirnoffee/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/simonboothroyd/smirnoffee/actions/workflows/ci.yaml)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/SimonBoothroyd/smirnoffee.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/SimonBoothroyd/smirnoffee/context:python)
[![codecov](https://codecov.io/gh/simonboothroyd/smirnoffee/branch/main/graph/badge.svg)](https://codecov.io/gh/simonboothroyd/smirnoffee/branch/main)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The `smirnoffee` framework aims to offer a simple API for differentiably evaluating the energy of [SMIRNOFF](
https://openforcefield.github.io/standards/standards/smirnoff/) force fields applied to **single** molecules using
``pytorch``.

The package currently supports evaluating the energy of force fields that contain: 

* `Bonds`, `Angles`, `ProperTorsions` and `ImproperTorsions` 
* `vdW`, `Electrostatics`, `LibraryCharges`, `ChargeIncrementModel`, and `ToolkitAM1BCC`

parameter handlers.

Force fields that apply virtual sites will likely be supported once the [`openff-interchange`](
https://github.com/openforcefield/openff-interchange) framework, which `smirnoffee` is currently built on top of, 
supports such particles.

***Warning**: This code is currently experimental and under active development. If you are using this it, please be 
aware that it is not guaranteed to provide correct results, the documentation and testing is incomplete, and the
API can change without notice.*

## Installation

The core dependencies can be installed using the [`conda`](https://docs.conda.io/en/latest/miniconda.html) 
package manager:

```shell
conda env create --name smirnoffee --file devtools/conda-envs/test-env.yaml
python setup.py develop
```

## Getting Started

To get started, we will show how the energy of paracetamol in a particular conformer can be evaluated using 
`smirnoffee` framework. We will then use ``pytorch`` to perform an energy minimization. 

We start by loading in the molecule of interest, in this case paracetamol as defined by its SMILES representation,
using the [`openff-toolkit`](https://github.com/openforcefield/openff-toolkit) and generating a single conformer for it:

```python
from openff.toolkit.topology import Molecule

molecule: Molecule = Molecule.from_smiles("CC(=O)NC1=CC=C(C=C1)O")
molecule.generate_conformers(n_conformers=1)

from simtk import unit
import torch

conformer = torch.tensor(molecule.conformers[0].value_in_unit(unit.angstrom))
```

Next we will load in the force field that encodes the potential energy function we wish to evaluate

```python
from openff.toolkit.typing.engines.smirnoff import ForceField
force_field = ForceField("openff_unconstrained-1.0.0.offxml")
```

The force field is applied to the molecule of interest using the `Interchange` object

```python
from openff.interchange.components.interchange import Interchange
openff_system = Interchange.from_smirnoff(force_field, molecule.to_topology())
```

The returned ``openff_system`` will contain the full set of parameters that have been applied to our molecule
in a format that is independant of any particular molecular simulation engine.

The energy of the molecule can then be directly be evaluated using the parameterized system and the confomrer of 
interest

```python
from smirnoffee.potentials import evaluate_system_energy
energy = evaluate_system_energy(openff_system, conformer)

print(f"Energy = {energy.item():.3f} kJ / mol")
```

> Energy = 137.622 kJ / mol

Because the ``evaluate_system_energy`` function will evaluate the system in a fully differentiable manner, we can
use to as part of an energy minimization loop

```python
from torch import optim

# Specify that we would like to compute the gradient of the energy with
# respect to this conformer.
conformer.requires_grad = True

# Minimize the conformer using the standard pytorch optimization loop.
optimizer = optim.Adam([conformer], lr=0.02)

for epoch in range(75):

    energy = evaluate_system_energy(openff_system, conformer)
    energy.backward()

    optimizer.step()
    optimizer.zero_grad()

    print(f"Epoch {epoch}: E={energy.item()} kJ / mol")

# Store the final conformer and save the molecule to file.
molecule.add_conformer(conformer.detach().numpy() * unit.angstrom)
molecule.to_file("molecule.xyz", "XYZ")
```

> Epoch 0:  E=137.62181091308594 kJ / mol
> 
> Epoch 25: E=112.00886535644531 kJ / mol
> 
> Epoch 50: E=110.65577697753906 kJ / mol
> 
> Epoch 74: E=110.43109130859375 kJ / mol

For more examples of using this framework, including how to compute the gradient of the energy with respect to a 
set of force field parameters, see the [examples](examples) directory.

### Copyright

Copyright (c) 2021, Simon Boothroyd
