SMIRNOFF Energy Evaluations
===========================
[![Test Status](https://github.com/simonboothroyd/smee/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/simonboothroyd/smee/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/simonboothroyd/smee/branch/main/graph/badge.svg)](https://codecov.io/gh/simonboothroyd/smee/branch/main)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The `smee` framework aims to offer a simple API for differentiably evaluating the energy of [SMIRNOFF](https://openforcefield.github.io/standards/standards/smirnoff/) 
force fields applied to molecules using `pytorch`.

The package currently supports evaluating the energy of force fields that contain: 

* `Bonds`, `Angles`, `ProperTorsions` and `ImproperTorsions` 
* `vdW`, `Electrostatics`, `ToolkitAM1BCC`, `LibraryCharges`
* `VirtualSites`

parameter handlers in addition to limited support for registering custom handlers.

***Warning**: This code is currently experimental and under active development. If you are using this it, please be 
aware that it is not guaranteed to provide correct results, the documentation and testing maybe be incomplete, and the
API can change without notice.*

## Installation

This package can be installed using `conda` (or `mamba`, a faster version of `conda`):

```shell
mamba install -c conda-forge smee
```

The example notebooks further require you install `jupyter`, `nglview`, `rdkit`, `ambertools` and `smirnoff-plugins`:

```shell
mamba install -c conda-forge jupyter nglview rdkit ambertools "smirnoff-plugins >=0.0.4"
```

## Getting Started

To get started, see the [examples](examples).

## Development

A development conda environment can be created and activated by running:

```shell
make env
conda activate smee
```

The environment will include all example and development dependencies, including linters and testing apparatus.

Unit / example tests can be run using:

```shell
make test
make test-examples
```

The codebase can be formatted by running:

```shell
make format
```

or checked for lint with:

```shell
make lint
```

## License

The main package is release under the [MIT license](LICENSE). 

## Copyright

Copyright (c) 2023, Simon Boothroyd
