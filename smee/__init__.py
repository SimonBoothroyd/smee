"""
smee

Differentiably evaluate energies of molecules using SMIRNOFF force fields
"""

from . import _version

__version__ = _version.get_versions()["version"]

__all__ = ["__version__"]


import inspect

# a hack to prevent pip failing as dependencies are 'not installed' in the build
# environment
if not any(
    "setuptools" in record.filename or "pip" in record.filename
    for record in inspect.stack()
):
    from .ff import convert_handlers, convert_interchange
    from .geometry import add_v_site_coords, compute_v_site_coords
    from .potentials import compute_energy, compute_energy_potential

    __all__ = [
        "__version__",
        "add_v_site_coords",
        "compute_v_site_coords",
        "convert_handlers",
        "convert_interchange",
        "compute_energy",
        "compute_energy_potential",
    ]

del inspect
