SMIRNOFF Energy Evaluations
===========================
[![Test Status](https://github.com/simonboothroyd/smirnoffee/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/simonboothroyd/smirnoffee/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/simonboothroyd/smirnoffee/branch/main/graph/badge.svg)](https://codecov.io/gh/simonboothroyd/smirnoffee/branch/main)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The `smirnoffee` framework aims to offer a simple API for differentiably evaluating the energy of [SMIRNOFF](https://openforcefield.github.io/standards/standards/smirnoff/) 
force fields applied to molecules using `pytorch`.

The package currently supports evaluating the energy of force fields that contain: 

* `Bonds`, `Angles`, `ProperTorsions` and `ImproperTorsions` 
* `vdW`, `Electrostatics`, `ToolkitAM1BCC`, `LibraryCharges`
* `VirtualSites`

parameter handlers.

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

To get started, see the [examples](examples).

## Development

A development conda environment can be created and activated by running:

```shell
make env
conda activate smirnoffee
```

The environment will include all development dependencies, including linters and testing apparatus.

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
