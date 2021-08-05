# Regression Testing

This directory contains scripts that evaluate the correctness of the outputs of this framework by
comparison with the more mature and heavily validated [OpenMM molecular simulation framework](
https://github.com/openmm/openmm).

In particular, these scripts compare the energies and forces computed by both `smirnoffee` and OpenMM
for a large set of molecules, and highlight molecules where there are large discrepancies.

## Retrieving the full molecule sets

The full suite of molecules used in the comparisons are currently sourced from the NCI 2012 Open set which contains
~250K drug like structures. Due to its size it is not stored in this repository and must be downloaded
and processed manually.

The raw archive of molecules can be retrieved directly from the NCI website:

```shell
curl https://cactus.nci.nih.gov/download/nci/NCI-Open_2012-05-01.sdf.gz --output NCI-molecules.sdf.gz
```

## Running the tests

1. (optional) split the tarball of molecules into chunks for easy parallel processing:

```shell
python 01-split-molecule-set.py -i small.sdf.gz
```

2. compute the energy of each SDF file:

```shell
python 02-compute-energies.py -i 01-split-molecules/small-1.sdf -o 02-energies-1.csv
```

3. concatenate the energy files together and sort by energy difference:

```shell
python 03-concat-energies.py -o 03-energies.csv 02-*.csv
```

4. inspect the energies for any outliers:

```shell
cat 03-energies.csv
```