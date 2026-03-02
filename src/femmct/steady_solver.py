import numpy as np

from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from basix.ufl import element
from ufl import (
    dx,
    sym,
    grad,
    div,
    inner,
    FacetNormal,
    Measure,
    TrialFunction,
    TestFunction,
    ZeroBaseForm
)

from femmct.configuration import SimulationConfig
from femmct.fluids import Fluid, NewtonianFluid, ThreeFieldNewtonianFluid


def setup_function_spaces(domain, fluid_model: Fluid, degree: int = 2):
    """
    Create function spaces depending on required fields.

    Args:
        domain : dolfinx.mesh.Mesh
        fluid_model : instance of Fluid subclass.

    Returns:
        spaces : function spaces containing all required fields.
        field_names : list of subspace labels.
    """

    num_fields = len(fluid_model.required_fields())

    cell = domain.topology.cell_name()
    gdim = domain.geometry.dim

    V_spec = fluid_model.required_fields().get("velocity")
    V_el = element(V_spec.family, cell, degree +
                   V_spec.degree_increment, shape=(gdim,))
    V = fem.functionspace(domain, V_el)

    Q_spec = fluid_model.required_fields().get("pressure")
    Q_el = element(Q_spec.family, cell, degree + Q_spec.degree_increment)
    Q = fem.functionspace(domain, Q_el)

    spaces = [V, Q]
    field_names = ["velocity", "pressure"]

    if num_fields > 2:
        # Add extra fields for viscoelastic stresses or other variables
        for field_name, spec in fluid_model.required_fields().items():
            if field_name in ["velocity", "pressure"]:
                continue
            if spec.value == "scalar":
                S_el = element(spec.family, cell, degree +
                               spec.degree_increment)
            elif spec.value == "vector":
                S_el = element(spec.family, cell, degree +
                               spec.degree_increment, shape=(gdim,))
            elif spec.value == "tensor":
                S_el = element(spec.family, cell, degree + spec.degree_increment,
                               shape=(gdim, gdim), symmetry=spec.symmetry)
            else:
                raise ValueError(f"Unknown field value type: {spec.value}")

            S = fem.functionspace(domain, S_el)
            spaces.append(S)
            field_names.append(field_name)

    return spaces, field_names


def setup_velocity_bc(V, facet_tags, marker, value, component=None):
    gdim = V.mesh.geometry.dim
    if component is None:
        f = fem.Function(V)
    else:
        V_comp = V.sub(component).collapse()[0]
        f = fem.Function(V_comp)

    if callable(value):
        f.interpolate(value)
    else:
        # Broadcast constant value to all dofs
        f.interpolate(lambda x: np.tile(
            np.array(value).reshape(-1, 1), x.shape[1]))

    if component is None:
        dofs = fem.locate_dofs_topological(V, gdim-1, facet_tags.find(marker))
        return fem.dirichletbc(f, dofs)
    else:
        dofs = fem.locate_dofs_topological(
            (V.sub(component), V_comp), gdim-1, facet_tags.find(marker))
        return fem.dirichletbc(f, dofs, V.sub(component))


def setup_boundary_conditions(function_spaces, config: SimulationConfig):

    bcs = []

    # Inflow boundary conditions
    if type(config.flow.inlet_marker) is int:
        boundary_segments = [config.flow.inlet_marker]
    else:
        boundary_segments = config.flow.inlet_marker
    for marker in boundary_segments:
        bc = setup_velocity_bc(function_spaces[0], config.simulation.mesh_data.facet_tags,
                               marker, config.flow.u_in)
        bcs.append(bc)

    # Wall boundary conditions
    if type(config.flow.wall_marker) is int:
        boundary_segments = [config.flow.wall_marker]
    else:
        boundary_segments = config.flow.wall_marker
    for marker in boundary_segments:
        bc = setup_velocity_bc(function_spaces[0], config.simulation.mesh_data.facet_tags,
                               marker, config.flow.u_wall)
        bcs.append(bc)

    # Outflow boundary conditions
    if type(config.flow.outlet_marker) is int:
        boundary_segments = [config.flow.outlet_marker]
    else:
        boundary_segments = config.flow.outlet_marker
    for marker in boundary_segments:
        bc = setup_velocity_bc(function_spaces[0], config.simulation.mesh_data.facet_tags,
                               marker, 0.0, component=1)  # Fix y-velocity at outflow
        bcs.append(bc)

    return bcs


def evaluate_rhs(rhs_data, function_space):
    if callable(rhs_data):
        rhs_func = fem.Function(function_space)
        rhs_func.interpolate(rhs_data)
        return rhs_func
    else:
        return fem.Constant(function_space.mesh, rhs_data)


def setup_newtonian_problem(function_spaces, bcs, config: SimulationConfig):

    V, Q = function_spaces[:]
    u, p = TrialFunction(V), TrialFunction(Q)
    v, q = TestFunction(V), TestFunction(Q)

    # RHS terms from body force and outflow boundary conditions
    f = evaluate_rhs(config.flow.body_force, V)
    p_atm = evaluate_rhs(config.flow.p_out, Q)

    # Weak form of the Stokes equations
    a = [[fem.Constant(config.simulation.mesh_data.mesh,
                       2.0 * config.fluid.viscosity)
          * inner(sym(grad(u)), sym(grad(v)))
          * dx, - p * div(v) * dx], [- q * div(u) * dx, None]]

    L = [inner(f, v) * dx, ZeroBaseForm((q,))]

    # Add outflow pressure boundary condition as a natural condition in the weak form
    if type(config.flow.outlet_marker) is int:
        outlet_segments = [config.flow.outlet_marker]
    else:
        outlet_segments = config.flow.outlet_marker
    n = FacetNormal(config.simulation.mesh_data.mesh)
    ds = Measure("ds", domain=config.simulation.mesh_data.mesh,
                 subdomain_data=config.simulation.mesh_data.facet_tags)
    L[0] -= sum(p_atm * inner(v, n) * ds(tag) for tag in outlet_segments)

    # Use direct solver for steady Stokes problems by default, but allow override via config
    default_options = {
        "ksp_type": "none",
        "ksp_error_if_not_converged": True,
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    petsc_options = config.simulation.petsc_options or default_options

    problem = LinearProblem(
        a, L, bcs=bcs, kind="nest", petsc_options=petsc_options, petsc_options_prefix="steady_newtonian_"
    )

    return problem


def setup_threefieldnewtonian_problem(function_spaces, bcs, config: SimulationConfig):

    V, Q, S = function_spaces[:]
    u, p, tau = TrialFunction(V), TrialFunction(Q), TrialFunction(S)
    v, q, sig = TestFunction(V), TestFunction(Q), TestFunction(S)

    # RHS terms from body force and outflow boundary conditions
    f = evaluate_rhs(config.flow.body_force, V)
    p_atm = evaluate_rhs(config.flow.p_out, Q)

    domain = config.simulation.mesh_data.mesh

    n = FacetNormal(domain)
    ds = Measure("ds", domain=domain,
                 subdomain_data=config.simulation.mesh_data.facet_tags)

    if type(config.flow.outlet_marker) is int:
        outlet_segments = [config.flow.outlet_marker]
    else:
        outlet_segments = config.flow.outlet_marker

    # Weak form of the Stokes equations
    mu_s = config.fluid.solvent_viscosity
    mu_p = config.fluid.polymeric_viscosity
    a_00 = fem.Constant(domain, 2.0 * mu_s) * \
        inner(sym(grad(u)), sym(grad(v))) * dx
    a_01 = - p * div(v) * dx
    a_02 = inner(tau, sym(grad(v))) * dx - sum(inner(tau * n, v)
                                               * ds(tag) for tag in outlet_segments)
    a_10 = - q * div(u) * dx
    a_20 = inner(sig, sym(grad(u))) * dx
    a_22 = - fem.Constant(domain, 1 / (2.0 * mu_p)) * inner(tau, sig) * dx
    
    a = [[a_00, a_01, a_02], [a_10, None, None], [a_20, None, a_22]]
    L = [inner(f, v) * dx - sum(p_atm * inner(v, n) * ds(tag)
                                for tag in outlet_segments), ZeroBaseForm((q,)), ZeroBaseForm((sig,))]

    default_options = {
        "ksp_type": "gmres",
        "ksp_error_if_not_converged": True,
        "ksp_monitor": None,
        "ksp_rtol": 1e-14,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "additive",
        "pc_fieldsplit_block_size": 3,
        "pc_fieldsplit_0_fields": [0, 1],
        "pc_fieldsplit_1_fields": 2,
        "fieldsplit_0_ksp_type": "none",
        "fieldsplit_0_pc_type": "lu",
        "fieldsplit_0_pc_factor_mat_solver_type": "mumps",
        "fieldsplit_1_ksp_type": "none",
        "fieldsplit_1_pc_type": "lu",
        "fieldsplit_1_pc_factor_mat_solver_type": "mumps",
    }

    petsc_options = config.simulation.petsc_options or default_options

    problem = LinearProblem(
        a, L, bcs=bcs, kind="nest", petsc_options=petsc_options, petsc_options_prefix="steady_newtonian_"
    )

    return problem


def setup_problem(function_spaces, bcs, config: SimulationConfig):
    if isinstance(config.fluid, NewtonianFluid):
        return setup_newtonian_problem(function_spaces, bcs, config)
    elif isinstance(config.fluid, ThreeFieldNewtonianFluid):
        return setup_threefieldnewtonian_problem(function_spaces, bcs, config)
    else:
        raise NotImplementedError(
            f"Solver for fluid type {type(config.fluid)} is not implemented."
        )


def solve(config: SimulationConfig):
    function_spaces, field_names = setup_function_spaces(
        config.simulation.mesh_data.mesh, config.fluid)

    bcs = setup_boundary_conditions(function_spaces, config)

    problem = setup_problem(function_spaces, bcs, config)

    solutions = problem.solve()
    for solution, name in zip(solutions, field_names):
        solution.name = name

    return solutions
