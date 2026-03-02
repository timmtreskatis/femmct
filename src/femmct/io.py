from mpi4py import MPI
from pathlib import Path
import subprocess

from dolfinx import fem, io

from femmct.configuration import SimulationConfig


def import_mesh(filename: str, apply_barycentric_refinement: bool = True):
    """
    Import or generate a mesh depending on file extension.

    Args:
        filename: Name of a GEO or MSH file.
        apply_barycentric_refinement: Whether triangles should be subdivided by connecting vertices with the centre of mass for LBB stability of lowest-order Scott-Vogelius elements.

    Returns:
        mesh_data: includes mesh, cell_tags and facet_tags
    """

    path = Path(filename)
    ext = path.suffix.lower()

    if ext not in [".geo", ".msh"]:
        raise ValueError(f"Unsupported mesh format '{ext}'. Expected .geo or .msh.")

    if ext == ".geo":
        cmd = ["gmsh", "-2", str(path), "-o", str(path.with_suffix(".msh"))]
        try:
            subprocess.run(cmd, check=True)
            print("Meshing of GEO file completed successfully.")

            path = path.with_suffix(".msh")

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Meshing of GEO file failed: {e}")

    if apply_barycentric_refinement:
        cmd = [
            "gmsh",
            str(path),
            "-barycentric_refine",
            "-o",
            str(path.parent / (path.stem + "_macroelements.msh")),
        ]
        try:
            subprocess.run(cmd, check=True)
            print("Barycentric refinement completed successfully.")

            path = (path.parent / (path.stem + "_macroelements")).with_suffix(".msh")

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Barycentric refinement failed: {e}")

    mesh_data = io.gmsh.read_from_msh(filename=path, comm=MPI.COMM_WORLD, gdim=2)
    return mesh_data


def export_vtx(
    f: fem.Function, config: SimulationConfig, *, filename: str = None, t: float = 0.0
):
    """
    Write a finite element field to disk in ADIOS2 / VTX format for visualization.

    This utility exports a FEniCSx `Function` as a `.bp` file that can be opened
    directly in ParaView.

    Args:
        f: Finite element function to be written.
        config: Simulation configuration containing an ``output_dir`` attribute.
        filename: File name (without suffix) of the output file. If ``None``, uses ``f.name``.
        t: Time step associated with this data set (default = 0.0).

    Note:
        Existing files with the same name will be overwritten.
    """
    
    if filename is None:
        filename = f.name
    comm = f.function_space.mesh.comm

    results_folder = Path(config.output_dir)
    results_folder.mkdir(exist_ok=True, parents=True)

    with io.VTXWriter(comm, (results_folder / filename).with_suffix(".bp"), [f]) as vtx:
        vtx.write(t)
