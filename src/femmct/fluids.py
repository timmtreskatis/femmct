from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass(frozen=True)
class FieldSpec:
    """Specification of fields to solve for."""

    name: str
    family: str = "P"
    degree_increment: int = 0
    value: str = "scalar"
    symmetry: bool = False


@dataclass(slots=True, frozen=True)
class Fluid(ABC):
    """Abstract base class for fluids."""

    density: float = 1.0  # Density [g/cm³]

    @abstractmethod
    def required_fields(self) -> dict[str, FieldSpec]:
        """Return a dictionary of field names with their specifications that this constitutive model requires."""
        pass

    def __str__(self):
        return f"Fluid with\n\tρ = {self.density}"


@dataclass(slots=True, frozen=True)
class NewtonianFluid(Fluid):
    viscosity: float = 1e-2  # Dynamic viscosity [P]

    def required_fields(self) -> dict[str, FieldSpec]:
        return {
            "velocity": FieldSpec(
                name="Velocity", family="P", degree_increment=0, value="vector"
            ),
            "pressure": FieldSpec(
                name="Pressure", family="DG", degree_increment=-1, value="scalar"
            )
        }

    def __str__(self):
        return (
            "Newtonian fluid with ρ = "
            + str(self.density)
            + ", µ = "
            + str(self.viscosity)
        )

@dataclass(slots=True, frozen=True)
class ThreeFieldNewtonianFluid(Fluid):
    solvent_viscosity: float = 1e-2  # Solvent viscosity [P]
    polymeric_viscosity: float = 1.0  # Polymeric viscosity [P]

    def required_fields(self) -> dict[str, FieldSpec]:
        return {
            "velocity": FieldSpec(
                name="Velocity", family="P", degree_increment=0, value="vector"
            ),
            "pressure": FieldSpec(
                name="Pressure", family="DG", degree_increment=-1, value="scalar"
            ),
            "stress": FieldSpec(
                name="Polymeric Stress",
                family="DG",
                degree_increment=-1,
                value="tensor",
                symmetry=True,
            ),
        }

    def __str__(self):
        return (
            "Newtonian fluid in three-field formulation with ρ = "
            + str(self.density)
            + ", µs = "
            + str(self.solvent_viscosity)
            + ", µp = "
            + str(self.polymeric_viscosity)
        )


@dataclass(slots=True, frozen=True)
class OldroydBFluid(Fluid):
    solvent_viscosity: float = 1e-2  # Solvent viscosity [P]
    polymeric_viscosity: float = 1.0  # Polymeric viscosity [P]
    relaxation_time: float = 1e-2  # Elastic relaxation time [s]

    def required_fields(self) -> dict[str, FieldSpec]:
        return {
            "velocity": FieldSpec(
                name="Velocity", family="P", degree_increment=0, value="vector"
            ),
            "pressure": FieldSpec(
                name="Pressure", family="DG", degree_increment=-1, value="scalar"
            ),
            "stress": FieldSpec(
                name="Polymeric Stress",
                family="DG",
                degree_increment=-1,
                value="tensor",
                symmetry=True,
            ),
        }

    def __str__(self):
        return (
            "Oldroyd-B fluid with ρ = "
            + str(self.density)
            + ", µs = "
            + str(self.solvent_viscosity)
            + ", µp = "
            + str(self.polymeric_viscosity)
            + ", λ = "
            + str(self.relaxation_time)
        )


@dataclass(slots=True, frozen=True)
class WhiteMetznerFluid(Fluid):
    """Physical parameters of a White-Metzner fluid with the constitutive law
        τ▿ + 1/λ τ = 2 G∞ Du
        1/λ = 1/λc + √(2) |Du|/γc
    for the polymeric stress tensor.

    G∞: elastic modulus [Ba]
    λc: characteristic elastic relaxation time [s]
    γc: characteristic strain [dimensionless]
    """

    solvent_viscosity: float = 1e-2  # Solvent viscosity [P]
    elastic_modulus: float = 1.0  # Elastic modulus [Ba]
    characteristic_strain: float = 1e-1  # Characteristic strain [dimensionless]
    relaxation_time: float = 1e2  # Elastic relaxation time [s]

    def required_fields(self) -> dict[str, FieldSpec]:
        return {
            "velocity": FieldSpec(
                name="Velocity", family="P", degree_increment=0, value="vector"
            ),
            "pressure": FieldSpec(
                name="Pressure", family="DG", degree_increment=-1, value="scalar"
            ),
            "stress": FieldSpec(
                name="Polymeric Stress",
                family="DG",
                degree_increment=-1,
                value="tensor",
                symmetry=True,
            ),
        }

    def __str__(self):
        return (
            "White-Metzner fluid with ρ = "
            + str(self.density)
            + ", µs = "
            + str(self.solvent_viscosity)
            + ", G∞ = "
            + str(self.elastic_modulus)
            + ", λc = "
            + str(self.relaxation_time)
            + ", γc = "
            + str(self.characteristic_strain)
        )
