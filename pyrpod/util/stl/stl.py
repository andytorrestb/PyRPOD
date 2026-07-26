from stl import mesh
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
import numpy as np
from numpy.typing import NDArray
from pyevtk.hl import unstructuredGridToVTK
from pyevtk.vtk import VtkTriangle
from pyrpod.util.io.fs import ensure_parent_dir

import argparse
import json


def load_stl(file_path: str) -> mesh.Mesh:
    """
    Load an STL file and return a mesh object.

    Parameters
    ----------
    file_path : str
        Path to the STL file.

    Returns
    -------
    mesh.Mesh
        The loaded STL mesh object.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"STL file not found: {file_path}")
    return mesh.Mesh.from_file(file_path)

def transform_mesh(
    mesh_obj: mesh.Mesh,
    rotation_matrix: NDArray[np.float64] | None = None,
    translation_vector: Sequence[float] | NDArray[np.float64] | None = None,
    scale_factor: float | None = None,
) -> mesh.Mesh:
    """
    Apply transformations to a mesh object.

    Parameters
    ----------
    mesh_obj : mesh.Mesh
        The mesh object to transform.
    rotation_matrix : np.ndarray, optional
        A 3x3 rotation matrix to apply to the mesh.
    translation_vector : list or np.ndarray, optional
        A 3-element vector to translate the mesh.
    scale_factor : float, optional
        A scaling factor to apply to the mesh.

    Returns
    -------
    mesh.Mesh
        The transformed mesh object.
    """
    if scale_factor:
        mesh_obj.points *= scale_factor
    if rotation_matrix is not None:
        mesh_obj.rotate_using_matrix(rotation_matrix)
    if translation_vector is not None:
        mesh_obj.translate(translation_vector)
    return mesh_obj


def transform_mesh_from_file(
    input_file: str,
    output_file: str,
    scale: float,
    translate: Sequence[float] | NDArray[np.float64],
) -> None:
    """
    Transform an STL mesh with scaling and translation and save it to a file.

    Parameters
    ----------
    input_file : str
        Path to the input STL file.
    output_file : str
        Path to the output STL file.
    scale : float
        Scaling factor to apply to the mesh.
    translate : list or np.ndarray
        Translation vector to apply to the mesh.
    """
    # Load the mesh from the file
    model = load_stl(input_file)

    # Apply transformations
    model = transform_mesh(model, scale_factor=scale, translation_vector=translate)

    # Save the transformed mesh to the output file
    model.save(output_file)
    print(f"Mesh saved to {output_file}")


def compose_meshes(meshes: Iterable[mesh.Mesh]) -> mesh.Mesh:
    """Concatenate an ordered collection of numpy-stl meshes into a single mesh.

    This is the canonical home for generic mesh composition used by the RPOD
    visualization pipeline (visiting-vehicle body + active plume cones +
    optional cluster geometry). It replaces repeated incremental
    ``np.concatenate`` calls that were duplicated across the study methods.

    Parameters
    ----------
    meshes : iterable of stl.mesh.Mesh
        Ordered meshes to concatenate. Ordering is preserved exactly in the
        output and all face data is concatenated in a single operation.

    Returns
    -------
    stl.mesh.Mesh
        A single mesh containing every input mesh's faces, in order. When
        exactly one mesh is supplied it is returned unchanged (the same
        object), matching the legacy incremental-composition behavior. The
        input meshes are never mutated.

    Raises
    ------
    ValueError
        If no meshes are supplied, or if any supplied element is ``None``.
        Composition never silently returns ``None`` nor drops a component.
    """
    meshes = list(meshes)
    if not meshes:
        raise ValueError("compose_meshes requires at least one mesh; got none.")
    if any(m is None for m in meshes):
        raise ValueError(
            "compose_meshes received a None mesh; every component must be a "
            "valid stl.mesh.Mesh.")
    if len(meshes) == 1:
        return meshes[0]
    return mesh.Mesh(np.concatenate([m.data for m in meshes]))


def convert_stl_to_vtk(
    surface: mesh.Mesh | str | Path,
    out_path: str | os.PathLike[str],
    *,
    filename: str | None = None,
    cellData: Mapping[str, NDArray[Any]] | None = None,
) -> None:
    """
    Convert an STL mesh (or path to an STL) to a VTK unstructured grid file.

    Parameters
    ----------
    surface : stl.mesh.Mesh or str or pathlib.Path
        The mesh instance or a path to an STL file.
    out_path : str or pathlib.Path
        Directory or file path where VTK output should be written. If a directory is
        provided, filename will be appended.
    filename : str, optional
        If provided and out_path is a directory, use this as the base filename
        (without extension). If out_path is a file, this is ignored.
    cellData : dict of numpy arrays, optional
        Optional cell (face) data to attach to the VTK cells.
    """
    # Accept either a mesh instance or a path.
    if isinstance(surface, (str, Path)):
        surface = mesh.Mesh.from_file(str(surface))
    elif not isinstance(surface, mesh.Mesh):
        raise TypeError("surface must be stl.mesh.Mesh or path to STL")

    out_path = Path(out_path)

    # If out_path is a directory, build file path from filename or from mesh name
    if out_path.is_dir() or out_path.suffix == "":
        if filename:
            base = filename
        else:
            base = getattr(surface, 'name', 'mesh')
        out_file_stem = out_path / base
    else:
        out_file_stem = out_path.with_suffix('')

    # Ensure parent directory exists
    ensure_parent_dir(str(out_file_stem) + '.vtu')

    # Use numpy operations to build vertex arrays and connectivity
    faces = surface.vectors  # shape (F,3,3)
    n_faces = faces.shape[0]

    coords = faces.reshape(-1, 3)

    # Ensure coordinates are contiguous float arrays (pyevtk requires C/F contiguous)
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    x = np.ascontiguousarray(coords[:, 0])
    y = np.ascontiguousarray(coords[:, 1])
    z = np.ascontiguousarray(coords[:, 2])

    conn = np.ascontiguousarray(np.arange(coords.shape[0], dtype=np.int32))

    offsets = np.ascontiguousarray(np.arange(3, 3 * n_faces + 1, 3, dtype=np.int32))

    ctype = np.ascontiguousarray(np.full(n_faces, VtkTriangle.tid, dtype=np.uint8))

    if cellData is None:
        cellData = {}

    unstructuredGridToVTK(str(out_file_stem), x, y, z, connectivity=conn, offsets=offsets, cell_types=ctype, cellData=cellData)


import argparse
import json
from stl import mesh
import numpy as np

def _transform_stl_file_cli(
    input_file: str,
    output_file: str,
    scale: float,
    translate: Sequence[float] | NDArray[np.float64],
) -> None:
    """Scale + translate an STL file in place from the command line.

    This is the file-based command-line helper invoked from ``__main__``. It
    is deliberately kept distinct from the reusable, in-memory
    ``transform_mesh(mesh_obj, ...)`` API above: the two previously shared the
    name ``transform_mesh``, and the later (file-based) definition silently
    shadowed the mesh-object API on import, breaking callers that passed a mesh
    plus ``rotation_matrix``/``scale_factor``. Only the CLI entry point below
    uses this function.
    """
    # Load the mesh from the file
    model = mesh.Mesh.from_file(input_file)

    # Scale the mesh
    model.points *= scale

    # Translate the mesh
    model.translate(translate)

    # Save the transformed mesh to the output file
    model.save(output_file)
    print(f"Mesh saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform an STL mesh with scaling and translation using a configuration file.")

    # Configuration file
    parser.add_argument("config_file", type=str, help="Path to the configuration JSON file.")

    args = parser.parse_args()

    # Load configuration from the file
    with open(args.config_file, 'r') as config_file:
        config = json.load(config_file)

    input_file = config.get("input_file")
    output_file = config.get("output_file")
    scale = config.get("scale", 1.0)
    translate = config.get("translate", [0.0, 0.0, 0.0])

    # Transform the mesh using the configuration
    _transform_stl_file_cli(
        input_file=input_file,
        output_file=output_file,
        scale=scale,
        translate=translate
    )
