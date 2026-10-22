import numpy
from .. import _CapsuleAsDLPack


def load_tet_mesh(path_file: str):
    """Load a tet mesh from a VTK mixed-cell file, extracting only tetrahedra.

    Args:
        path_file: path to the .vtk file
    Returns:
        tet2vtx: (num_tet, 4) uint32 - tetrahedron connectivity
        vtx2xyz: (num_vtx, 3) float32 - vertex positions
    """
    from .. import IoVtk

    cap_tet2vtx, cap_vtx2xyz = IoVtk.load_tet_mesh(path_file)
    tet2vtx = numpy.from_dlpack(_CapsuleAsDLPack(cap_tet2vtx)).copy()
    vtx2xyz = numpy.from_dlpack(_CapsuleAsDLPack(cap_vtx2xyz)).copy()
    return tet2vtx, vtx2xyz


def write_mix_mesh(
    path_file: str,
    vtx2xyz: numpy.ndarray,
    tet2vtx: numpy.ndarray,
    pyrmd2vtx: numpy.ndarray,
    prism2vtx: numpy.ndarray,
    hex2vtx: numpy.ndarray,
):
    """Save a mixed-element mesh to a VTK file.

    Args:
        path_file: output file path
        vtx2xyz: (num_vtx, 3) float32 - vertex positions
        tet2vtx: (num_tet, 4) uint32 - tetrahedron connectivity
        pyrmd2vtx: (num_pyrmd, 5) uint32 - pyramid connectivity
        prism2vtx: (num_prism, 6) uint32 - prism connectivity
        hex2vtx: (num_hex, 8) uint32 - hexahedron connectivity
    """
    from .. import util_numpy
    from .. import IoVtk

    num_vtx = vtx2xyz.shape[0]
    num_tet = tet2vtx.shape[0]
    num_pyrmd = pyrmd2vtx.shape[0]
    num_prism = prism2vtx.shape[0]
    num_hex = hex2vtx.shape[0]
    util_numpy.assert_shape_dtype(vtx2xyz, (num_vtx, 3), numpy.float32)
    util_numpy.assert_shape_dtype(tet2vtx, (num_tet, 4), numpy.uint32)
    util_numpy.assert_shape_dtype(pyrmd2vtx, (num_pyrmd, 5), numpy.uint32)
    util_numpy.assert_shape_dtype(prism2vtx, (num_prism, 6), numpy.uint32)
    util_numpy.assert_shape_dtype(hex2vtx, (num_hex, 8), numpy.uint32)
    IoVtk.write_mix_mesh(
        path_file,
        vtx2xyz.__dlpack__(),
        tet2vtx.__dlpack__(),
        pyrmd2vtx.__dlpack__(),
        prism2vtx.__dlpack__(),
        hex2vtx.__dlpack__(),
    )
