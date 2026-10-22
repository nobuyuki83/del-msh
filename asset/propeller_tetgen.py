import os, pathlib

import numpy as np
import tetgen


def tetrahedralize(tri2vtx, vtx2xyz):
    """
    入力:
        vtx2xyz: (N, 3) 頂点座標
        tri2vtx: (M, 3) 三角形の頂点インデックス（0始まり）

    出力:
        node2xyz: (K, 3) 四面体メッシュの頂点座標
        tet2node: (L, 4) 四面体の頂点インデックス（0始まり）
    """
    vertices = np.ascontiguousarray(vtx2xyz, dtype=np.float64)
    faces = np.ascontiguousarray(tri2vtx, dtype=np.int32)

    tgen = tetgen.TetGen(vertices, faces)
    tgen.tetrahedralize(
        order=1,  # 4頂点の一次四面体
        quality=True,
        minratio=1.5,
        mindihedral=10,
    )

    return tgen.elem.copy(), tgen.node.copy()


def tetrahedralize_exterior(
        tri2vtx,
        vtx2xyz,
        hole_point,
        padding=0.5,
):
    """
    Generate tetrahedra between a closed surface and an enclosing box.

    Parameters
    ----------
    vtx2xyz : (N, 3) array
        Surface vertex positions.
    tri2vtx : (M, 3) integer array
        Surface triangle indices (zero-based).
    hole_point : (3,) array
        Point strictly inside the object to exclude.
    padding : float
        Box margin relative to the object's largest bounding-box extent.

    Returns
    -------
    node2xyz : (K, 3) array
        Output vertex positions.
    tet2node : (L, 4) integer array
        Tetrahedron indices into node2xyz.
    """
    vertices = np.asarray(vtx2xyz, dtype=np.float64)
    faces = np.asarray(tri2vtx, dtype=np.int32)

    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("vtx2xyz must have shape (N, 3), with N > 0.")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("tri2vtx must have shape (M, 3), with M > 0.")
    if not np.isfinite(vertices).all():
        raise ValueError("Vertex coordinates must be finite.")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Triangle indices are out of bounds.")
    if not np.isfinite(padding) or padding <= 0:
        raise ValueError("padding must be positive and finite.")

    hole_point = np.asarray(hole_point, dtype=np.float64)
    if hole_point.shape != (3,) or not np.isfinite(hole_point).all():
        raise ValueError("hole_point must contain three finite coordinates.")

    xyz_min = vertices.min(axis=0)
    xyz_max = vertices.max(axis=0)
    extent = (xyz_max - xyz_min).max()
    if extent <= 0:
        raise ValueError("The input surface has zero extent.")

    lo = xyz_min - padding * extent
    hi = xyz_max + padding * extent
    x0, y0, z0 = lo
    x1, y1, z1 = hi

    box_vertices = np.array([
        [x0, y0, z0],  # 0
        [x1, y0, z0],  # 1
        [x1, y1, z0],  # 2
        [x0, y1, z0],  # 3
        [x0, y0, z1],  # 4
        [x1, y0, z1],  # 5
        [x1, y1, z1],  # 6
        [x0, y1, z1],  # 7
    ], dtype=np.float64)

    # Two outward-facing triangles per box face.
    box_faces = np.array([
        [0, 2, 1], [0, 3, 2],  # -Z
        [4, 5, 6], [4, 6, 7],  # +Z
        [0, 1, 5], [0, 5, 4],  # -Y
        [3, 7, 6], [3, 6, 2],  # +Y
        [0, 4, 7], [0, 7, 3],  # -X
        [1, 2, 6], [1, 6, 5],  # +X
    ], dtype=np.int32)

    combined_vertices = np.vstack([vertices, box_vertices])
    combined_faces = np.vstack([faces, box_faces + len(vertices)])

    tgen = tetgen.TetGen(combined_vertices, combined_faces)
    tgen.add_hole(hole_point.tolist())
    tgen.tetrahedralize(
        order=1,
        quality=True,
        minratio=1.5,
        mindihedral=10,
    )

    return tgen.elem.copy(), tgen.node.copy()


if __name__ == "__main__":
    import del_msh_dlpack.IoOff.numpy as IoOff

    path_dir = pathlib.Path(os.path.dirname(__file__))
    print(path_dir)
    tri2vtx, vtx2xyz = IoOff.load_tri_mesh(str(path_dir / "propeller0.off"))
    tet2wtx, wtx2xyz = tetrahedralize_exterior(tri2vtx, vtx2xyz, hole_point=[3.1, 0.1, 5.1])
    print(tet2wtx.shape, wtx2xyz.shape)
    import del_msh_dlpack.IoVtk.numpy as IoVtk

    IoVtk.write_mix_mesh(
        "propeller0_exterior.vtk",
        wtx2xyz.astype(np.float32),
        tet2wtx.astype(np.uint32),
        np.zeros([0, 5], dtype=np.uint32),
        np.zeros([0, 6], dtype=np.uint32),
        np.zeros([0, 8], dtype=np.uint32)
    )
