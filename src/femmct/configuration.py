from dataclasses import dataclass, field
import numpy as np
from typing import Callable, List
import numpy.typing as npt

from dolfinx.io.gmsh import MeshData

from femmct.fluids import Fluid, NewtonianFluid

VectorValue = tuple[float, float] | tuple[float, float, float] | tuple[float, float, float, float]
ScalarValue = float
VectorFunction = Callable[[npt.NDArray[np.float64]], VectorValue]
ScalarFunction = Callable[[npt.NDArray[np.float64]], ScalarValue]


@dataclass
class SimulationData:
    """
    Dataclass holding the mesh and, optionally, custom PETSc. options
    """
    mesh_data: MeshData
    petsc_options: dict = None

@dataclass
class FlowData:
    """
    Dataclass holding specifications of the type of flow, boundary and right hand side data
    """
    inlet_marker: int | List[int]
    wall_marker: int | List[int]
    outlet_marker: int | List[int]
    u_in: VectorFunction | VectorValue
    u_wall: VectorFunction | VectorValue
    p_out: ScalarFunction | ScalarValue
    body_force: VectorFunction | VectorValue
    inertial: bool = False
    steady: bool = True
    tau_in: VectorFunction | VectorValue | None = None



@dataclass
class SimulationConfig:
    """
    Combined configuration object holding both numerical and physical parameters.
    This can be passed as a single argument to solvers.
    """

    simulation: SimulationData
    flow: FlowData = field(default_factory=FlowData)
    fluid: Fluid = field(default_factory=NewtonianFluid)

    # Global metadata
    case_name: str = "default_case"
    output_dir: str = "femmct_results"

    def summary(self):
        """Print a short summary of the current configuration."""
        print(f"Case name:        {self.case_name}")
        print(f"Output directory: {self.output_dir}")
        print(f"Fluid:            {self.fluid}")
