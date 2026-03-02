from mpi4py import MPI
from pathlib import Path

from dolfinx import fem
from ufl import div, dx, grad, inner, sym
import numpy as np

from femmct.configuration import (
    SimulationConfig,
    FlowData,
    SimulationData
)
from femmct.fluids import NewtonianFluid, ThreeFieldNewtonianFluid
from femmct.io import import_mesh
import femmct.steady_solver as steady_solver


def compute_error(numerical, exact=None, norm: str = "L2"):
    """Compute global error between two fields."""
    if exact is None:
        # If no exact solution is provided, compute the error against zero.
        exact = fem.Function(numerical.function_space)
    if norm == "L2":
        error_form = inner(numerical - exact, numerical - exact) * dx
    elif norm == "H1":
        error_form = (
            inner(numerical - exact, numerical - exact) * dx +
            inner(sym(grad(numerical - exact)),
                  sym(grad(numerical - exact))) * dx
        )
    elif norm == "div":
        error_form = div(numerical - exact)**2 * dx
    else:
        raise ValueError(f"Unsupported norm type: {norm}")

    local = fem.assemble_scalar(fem.form(error_form))
    comm = numerical.function_space.mesh.comm
    return np.sqrt(comm.allreduce(local, op=MPI.SUM))


def test_newtonian():
    """Test that the steady Newtonian Stokes solver reproduces Poiseuille flow."""
    viscosity = 2.0
    unit_pressure_drop = 3.0
    atmospheric_pressure = 4.0
    channel_length = 5.0

    def manufactured_velocity(x):
        return (-unit_pressure_drop / (2 * viscosity)
                * (x[1] ** 2 - x[1]), 0 * x[1])

    def manufactured_pressure(x):
        return atmospheric_pressure - unit_pressure_drop * (x[0] - channel_length)

    mesh_file = str(Path(__file__).parent / "test_macroelements.msh")
    mesh_data = import_mesh(mesh_file, apply_barycentric_refinement=False)
    simulation_data = SimulationData(mesh_data)

    fluid_data = NewtonianFluid(viscosity=viscosity)
    flow_data = FlowData(
        inlet_marker=1,
        wall_marker=2,
        outlet_marker=3,
        u_in=manufactured_velocity,
        u_wall=manufactured_velocity,
        p_out=manufactured_pressure,
        body_force=(0.0, 0.0),
    )
    config = SimulationConfig(
        simulation=simulation_data, flow=flow_data, fluid=fluid_data)

    u_num, p_num = steady_solver.solve(config)

    u_exact = fem.Function(u_num.function_space)
    u_exact.interpolate(manufactured_velocity)
    p_exact = fem.Function(p_num.function_space)
    p_exact.interpolate(manufactured_pressure)

    error_u = compute_error(u_num, u_exact, norm="H1")
    error_div = compute_error(u_num, norm="div")
    error_p = compute_error(p_num, p_exact, norm="L2")

    assert error_u < 1e-12, f"Velocity fields differ by {error_u}."
    assert error_div < 1e-14, f"The velocity is not divergence-free: the L2 norm of the divergence is {error_div}."
    assert error_p < 1e-12, f"Pressure fields differ by {error_p}."


def test_threefieldnewtonian():
    """Test that the steady Newtonian Stokes solver in three-field formulation reproduces Poiseuille flow."""
    solvent_viscosity = 2.0
    polymeric_viscosity = 3.0
    unit_pressure_drop = 4.0
    atmospheric_pressure = 6.0
    channel_length = 5.0

    total_viscosity = solvent_viscosity + polymeric_viscosity

    def manufactured_velocity(x):
        return (-unit_pressure_drop / (2 * total_viscosity)
                * (x[1] ** 2 - x[1]), 0 * x[1])

    def manufactured_pressure(x):
        return atmospheric_pressure - unit_pressure_drop * (x[0] - channel_length)

    def manufactured_stress(x):
        return (
            0 * x[1],
            unit_pressure_drop
            * polymeric_viscosity
            / (2 * total_viscosity)
            * (1 - 2 * x[1]),
            unit_pressure_drop
            * polymeric_viscosity
            / (2 * total_viscosity)
            * (1 - 2 * x[1]),
            0 * x[1],
        )

    mesh_file = str(Path(__file__).parent / "test_macroelements.msh")
    mesh_data = import_mesh(mesh_file, apply_barycentric_refinement=False)
    simulation_data = SimulationData(mesh_data)

    fluid_data = ThreeFieldNewtonianFluid(solvent_viscosity=solvent_viscosity,
                                          polymeric_viscosity=polymeric_viscosity)
    flow_data = FlowData(
        inlet_marker=1,
        wall_marker=2,
        outlet_marker=3,
        u_in=manufactured_velocity,
        u_wall=manufactured_velocity,
        p_out=manufactured_pressure,
        body_force=(0.0, 0.0),
    )
    config = SimulationConfig(
        simulation=simulation_data, flow=flow_data, fluid=fluid_data)

    u_num, p_num, tau_num = steady_solver.solve(config)

    u_exact = fem.Function(u_num.function_space)
    u_exact.interpolate(manufactured_velocity)
    p_exact = fem.Function(p_num.function_space)
    p_exact.interpolate(manufactured_pressure)
    tau_exact = fem.Function(tau_num.function_space)
    tau_exact.interpolate(manufactured_stress)

    error_u = compute_error(u_num, u_exact, norm="H1")
    error_div = compute_error(u_num, norm="div")
    error_p = compute_error(p_num, p_exact, norm="L2")
    error_tau = compute_error(tau_num, tau_exact, norm="L2")

    assert error_u < 1e-11, f"Velocity fields differ by {error_u}."
    assert error_div < 1e-14, f"Velocity field is not divergence-free: the L2 norm of the divergence is {error_div}."
    assert error_p < 1e-12, f"Pressure fields differ by {error_p}."
    assert error_tau < 1e-10, f"Stress fields differ by {error_tau}."
