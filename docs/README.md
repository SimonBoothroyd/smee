Building the Documentation
==========================

To build the documentation first build an appropriate conda environment:

```shell
conda env create --name smirnoffee-docs --file environment.yml
conda activate smirnoffee-docs

cd ..
python setup.py develop
cd -
```

The documentation is then compiled by running:

```shell
make html
```

and can be found in the generated `_build/html` directory.
