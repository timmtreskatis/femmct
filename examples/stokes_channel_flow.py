from femmct.configuration import (
    SimulationConfig,
    FlowData,
    SimulationData
)
from femmct.fluids import NewtonianFluid
from femmct.steady_solver import solve
from femmct.io import import_mesh, export_vtx


def inflow_velocity(x):
    dp = 1.0
    mu_p = 1.0
    mu_s = 1e-2
    return (
        -dp
        / (2.0 * (mu_p + mu_s))
        * (x[1] ** 2 - x[1]),
        0 * x[1],
    )


mesh_data = import_mesh("channel.geo")

simulation_data = SimulationData(mesh_data)

flow_data = FlowData(
    inlet_marker=1,
    wall_marker=2,
    outlet_marker=3,
    u_in=inflow_velocity,
    u_wall=(0.0, 0.0),
    p_out=1.0,
    body_force=(0.0, 0.0),
)


config = SimulationConfig(
    simulation=simulation_data, flow=flow_data, fluid=NewtonianFluid()
)
config.case_name = "Stokes_Channel"


config.summary()
results = solve(config)

for result in results:
    export_vtx(result, config)
