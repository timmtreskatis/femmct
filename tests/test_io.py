import pytest
from mpi4py import MPI
from pathlib import Path

from dolfinx import fem, io, mesh

import femmct.io as io_module


def test_import_mesh_invalid_extension(tmp_path):
    """Test that unsupported file extension raises ValueError."""

    bad_file = tmp_path / "mesh.txt"
    with pytest.raises(ValueError):
        io_module.import_mesh(str(bad_file))


@pytest.mark.parametrize(
    "filename, apply_refine, expected_shape",
    [
        ("test.geo", False, (40, 3)),
        ("test.msh", True, (120, 3)),
    ],
)
def test_import_mesh(tmp_path, filename, apply_refine, expected_shape):
    """Test that sample GEO or MSH files are imported as MeshData with a 2-D mesh that has the expected number of triangles."""

    src = Path(__file__).parent / filename
    dst = src.copy_into(tmp_path)

    mesh_data = io_module.import_mesh(
        str(dst), apply_barycentric_refinement=apply_refine
    )

    assert type(mesh_data) is io.gmsh.MeshData
    assert mesh_data.mesh.topology.dim == 2
    assert mesh_data.mesh.geometry.dofmap.shape == expected_shape


def test_export_vtx(tmp_path):
    """Test that export_vtx writes a BP file."""

    domain = mesh.create_unit_square(MPI.COMM_SELF, 1, 1)
    V = fem.functionspace(domain, ("P", 1))
    f = fem.Function(V)
    f.name = "Velocity"

    class SampleConfig:
        output_dir = str(tmp_path)

    io_module.export_vtx(f, SampleConfig(), filename="velocity_test", t=0.0)

    output_files = list(tmp_path.glob("*.bp"))

    assert len(output_files) == 1
