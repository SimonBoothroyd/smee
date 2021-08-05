import click
import numpy
import pandas


@click.command()
@click.argument(
    "file_names",
    nargs=-1,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option(
    "-o",
    "--output",
    "output_path",
    show_default=True,
    help="The path to save the concatenated energies to (.csv).",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
)
def join_energies(file_names, output_path):
    """Concatenates a set of CSV FILES_NAMES together.

    The rows will be sorted by the difference in the OpenMM and OpenFF energies in
    descending order.
    """

    joined_data = pandas.concat(
        [pandas.read_csv(file_name) for file_name in file_names],
        ignore_index=True,
        sort=False,
    )
    joined_data["Delta E"] = numpy.abs(
        joined_data["E OpenMM"] - joined_data["E OpenFF"]
    )
    joined_data = joined_data.sort_values(by="Delta E", ascending=False)

    joined_data.to_csv(output_path, index=False)


if __name__ == "__main__":
    join_energies()
