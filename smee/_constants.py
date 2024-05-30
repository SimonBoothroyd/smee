"""Constants used throughout the package."""

import enum

if hasattr(enum, "StrEnum"):
    _StrEnum = enum.StrEnum
else:
    import typing

    _S = typing.TypeVar("_S", bound="_StrEnum")

    class _StrEnum(str, enum.Enum):
        """TODO: remove when python 3.10 support is dropped."""

        def __new__(cls: typing.Type[_S], *values: str) -> _S:
            value = str(*values)

            member = str.__new__(cls, value)
            member._value_ = value

            return member

        __str__ = str.__str__


class PotentialType(_StrEnum):
    """An enumeration of the potential types supported by ``smee`` out of the box."""

    BONDS = "Bonds"
    ANGLES = "Angles"

    PROPER_TORSIONS = "ProperTorsions"
    IMPROPER_TORSIONS = "ImproperTorsions"

    VDW = "vdW"
    ELECTROSTATICS = "Electrostatics"


class EnergyFn(_StrEnum):
    """An enumeration of the energy functions supported by ``smee`` out of the box."""

    COULOMB = "coul"

    VDW_LJ = "4*epsilon*((sigma/r)**12-(sigma/r)**6)"
    VDW_DEXP = (
        "epsilon*("
        "beta/(alpha-beta)*exp(alpha*(1-r/r_min))-"
        "alpha/(alpha-beta)*exp(beta*(1-r/r_min)))"
    )
    # VDW_BUCKINGHAM = "a*exp(-b*r)-c*r^-6"

    BOND_HARMONIC = "k/2*(r-length)**2"

    ANGLE_HARMONIC = "k/2*(theta-angle)**2"

    TORSION_COSINE = "k*(1+cos(periodicity*theta-phase))"
