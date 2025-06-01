import typing
#
import numpy
import numpy.typing

import del_msh_numpy

# ------------------------------
# below: vtx2***

def vtx2vtx(
        tri2vtx: numpy.typing.NDArray,
        num_vtx: int,
        is_self = False) \
        -> (numpy.typing.NDArray, numpy.typing.NDArray):
    assert len(tri2vtx.shape) == 2
    assert tri2vtx.shape[1] == 3
    from .del_msh_numpy import vtx2vtx_trimesh
    return vtx2vtx_trimesh(tri2vtx, num_vtx, is_self)


def vtx2area(
        tri2vtx: numpy.typing.NDArray,
        vtx2xyz: numpy.typing.NDArray):
   from .del_msh_numpy import vtx2area_from_uniformmesh
   return vtx2area_from_uniformmesh(tri2vtx, vtx2xyz)

# above: vtx2***
# --------------------------------------
# below: edge2***

def edge2vtx(
        tri2vtx: numpy.typing.NDArray,
        num_vtx: int) \
        -> numpy.typing.NDArray:
    """
    compute line mesh connectivity from a triangle connectivity
    :param tri2vtx: triangle mesh connectivity
    :param num_vtx: number of vertex
    :return: 2D numpy.ndarray showing the vertex index for each edge
    """
    assert len(tri2vtx.shape) == 2
    assert tri2vtx.shape[1] == 3
    assert not numpy.isfortran(tri2vtx)
    from .del_msh_numpy import edge2vtx_uniform_mesh
    return edge2vtx_uniform_mesh(tri2vtx, num_vtx)


def boundaryedge2vtx(
        tri2vtx: numpy.typing.NDArray,
        num_vtx: int ) -> (numpy.typing.NDArray, numpy.typing.NDArray):
   assert len(tri2vtx.shape) == 2
   assert tri2vtx.shape[1] == 3
   assert not numpy.isfortran(tri2vtx)
   from .del_msh_numpy import boundaryedge2vtx_triangle_mesh
   return boundaryedge2vtx_triangle_mesh(tri2vtx, num_vtx)


# above: edge2***
# ---------------------------------------
# below: tri2***

def tri2tri(
        tri2vtx: numpy.typing.NDArray,
        num_vtx: int) \
        -> numpy.typing.NDArray:
    from .del_msh_numpy import elem2elem_uniform_mesh_polygon_indexing
    return elem2elem_uniform_mesh_polygon_indexing(tri2vtx, num_vtx)


def tri2distance(
        idx_tri: int,
        tri2tri: numpy.typing.NDArray) -> numpy.typing.NDArray:
    from .del_msh_numpy import topological_distance_on_uniform_mesh
    return topological_distance_on_uniform_mesh(idx_tri, tri2tri)


def tri2area(
        tri2vtx: numpy.typing.NDArray,
        vtx2xyz: numpy.typing.NDArray) \
        -> numpy.typing.NDArray:
    """
    Areas of the triangles in a 2D/3D mesh.
    :param tri2vtx:
    :param vtx2xyz:
    :return:
    """
    assert vtx2xyz.shape[1] == 2 or vtx2xyz.shape[1] == 3, "the dimension should be 2 or 3"
    from .del_msh_numpy import areas_of_triangles_of_mesh
    return areas_of_triangles_of_mesh(tri2vtx, vtx2xyz)


def tri2circumcenter(
        tri2vtx: numpy.typing.NDArray,
        vtx2xyz: numpy.typing.NDArray):
   from .del_msh_numpy import circumcenters_of_triangles_of_mesh
   return circumcenters_of_triangles_of_mesh(tri2vtx, vtx2xyz)

# above: tri2***
# ------------------------
# below: primitive

def torus(
        major_radius=1.0,
        minor_radius=0.4,
        ndiv_major_radius=32,
        ndiv_minor_radius=32) \
        -> (numpy.typing.NDArray, numpy.typing.NDArray):
    from .del_msh_numpy import torus_meshtri3
    return torus_meshtri3(major_radius, minor_radius, ndiv_major_radius, ndiv_minor_radius)


def capsule(
        radius: float = 1.0,
        height: float = 1.,
        ndiv_theta: int = 32,
        ndiv_height: int = 32,
        ndiv_longtitude: int = 32) \
        -> (numpy.typing.NDArray, numpy.typing.NDArray):
    from .del_msh_numpy import capsule_meshtri3
    return capsule_meshtri3(radius, height, ndiv_theta, ndiv_longtitude, ndiv_height)


def cylinder(
        radius: float = 1.,
        height: float = 1.,
        ndiv_circumference: int = 32,
        ndiv_height: int = 8,
        is_closed_end = True,
        is_center = True) \
        -> (numpy.typing.NDArray, numpy.typing.NDArray):
   from .del_msh_numpy import cylinder_closed_end_meshtri3
   return cylinder_closed_end_meshtri3(
        radius, height,
        ndiv_circumference, ndiv_height,
        is_closed_end,
        is_center)


def sphere(
        radius: float = 1.,
        ndiv_latitude: int = 32,
        ndiv_longtitude: int = 32) \
        -> (numpy.typing.NDArray, numpy.typing.NDArray):
    from .del_msh_numpy import sphere_meshtri3
    return sphere_meshtri3(radius, ndiv_latitude, ndiv_longtitude)


def hemisphere(
        radius: float = 1.,
        ndiv_longtitude: int = 32,
        ndiv_latitude: int = 32) \
        -> (numpy.typing.NDArray, numpy.typing.NDArray):
    assert ndiv_longtitude > 0
    assert ndiv_latitude > 2
    from .del_msh_numpy import trimesh3_hemisphere_zup
    return trimesh3_hemisphere_zup(radius, ndiv_longtitude, ndiv_latitude)


# above: primitive
# ------------------------------
# below: io

def load_wavefront_obj(
        path_file: str,
        is_centerize=False,
        normalized_size: typing.Optional[float] = None):
    from .del_msh_numpy import load_wavefront_obj_as_triangle_mesh
    tri2vtx, vtx2xyz = load_wavefront_obj_as_triangle_mesh(path_file)
    if is_centerize:
        vtx2xyz[:] -= (vtx2xyz.max(axis=0) + vtx2xyz.min(axis=0)) * 0.5
    if type(normalized_size) == float:
        vtx2xyz *= normalized_size / (vtx2xyz.max(axis=0) - vtx2xyz.min(axis=0)).max()
    return tri2vtx, vtx2xyz


def load_nastran(
        path_file: str):
    from .del_msh_numpy import load_nastran_as_triangle_mesh
    return load_nastran_as_triangle_mesh(path_file)


def load_off(
        path_file: str,
        is_centerize=False,
        normalized_size: typing.Optional[float] = None):
    from .del_msh_numpy import load_off_as_triangle_mesh
    tri2vtx, vtx2xyz = load_off_as_triangle_mesh(path_file)
    if is_centerize:
        vtx2xyz[:] -= (vtx2xyz.max(axis=0) + vtx2xyz.min(axis=0)) * 0.5
    if type(normalized_size) == float:
        vtx2xyz *= normalized_size / (vtx2xyz.max(axis=0) - vtx2xyz.min(axis=0)).max()
    return tri2vtx, vtx2xyz


# above: io
# --------------------------------
# below: misc

def unindexing(
        tri2vtx: numpy.typing.NDArray,
        vtx2xyz: numpy.typing.NDArray) \
        -> numpy.typing.NDArray:
    from .del_msh_numpy import unidex_vertex_attribute_for_triangle_mesh
    return unidex_vertex_attribute_for_triangle_mesh(tri2vtx, vtx2xyz)




def merge(
        tri2vtx0: numpy.typing.NDArray,
        vtx2xyz0: numpy.typing.NDArray,
        tri2vtx1: numpy.typing.NDArray,
        vtx2xyz1: numpy.typing.NDArray):
    num_vtx0 = vtx2xyz0.shape[0]
    tri2vtx = numpy.vstack([tri2vtx0, tri2vtx1 + num_vtx0])
    vtx2xyz = numpy.vstack([vtx2xyz0, vtx2xyz1])
    return tri2vtx, vtx2xyz


# above: property
# ------------------------
# below: sampling

def position(
        tri2vtx: numpy.typing.NDArray,
        vtx2xyz: numpy.typing.NDArray,
        idx_tri: int, r0: float, r1: float) -> numpy.typing.NDArray:
    i0 = tri2vtx[idx_tri][0]
    i1 = tri2vtx[idx_tri][1]
    i2 = tri2vtx[idx_tri][2]
    p0 = vtx2xyz[i0]
    p1 = vtx2xyz[i1]
    p2 = vtx2xyz[i2]
    return r0 * p0 + r1 * p1 + (1. - r0 - r1) * p2


def sample(
        cumsum_area: numpy.ndarray,
        r0: float,
        r1: float):
    from .del_msh_numpy import sample_uniformly_trimesh
    return sample_uniformly_trimesh(cumsum_area, r0, r1)


def sample_many(
        tri2vtx: numpy.typing.NDArray,
        vtx2xy, num_sample: int) -> numpy.ndarray:
    import random
    tri2area_ = tri2area(tri2vtx, vtx2xy)
    cumsum_area = numpy.cumsum(numpy.append(numpy.zeros(1, dtype=numpy.float32), tri2area_))
    num_dim = vtx2xy.shape[1]
    xys = numpy.zeros([num_sample, num_dim], numpy.float32)
    for i in range(num_sample):
        smpl_i = sample(cumsum_area, random.random(), random.random())
        xys[i] = position(tri2vtx, vtx2xy, *smpl_i)
    return xys


# ------------------------------------
# below: BVH related functions

def bvhnodes_tri(
        tri2vtx: numpy.typing.NDArray,
        vtx2xyz: numpy.typing.NDArray,
        is_morton=True) \
        -> (numpy.typing.NDArray, numpy.typing.NDArray):
    """
    2D and 3D bvh tree topoloty geneartion
    :param tri2vtx:
    :param vtx2xyz:
    :return: array of bvh nodes (X-by-3 matrix)
    """
    if is_morton:
        tri2center = (vtx2xyz[tri2vtx[:, 0], :] + vtx2xyz[tri2vtx[:, 1], :] + vtx2xyz[tri2vtx[:, 2], :]) / 3
        del_msh_numpy.fit_into_unit_cube(tri2center)  # fit the points inside unit cube [0,1]^3
        from .del_msh_numpy import build_bvh_topology_morton
        return build_bvh_topology_morton(tri2center)
    else:
        from .del_msh_numpy import build_bvh_topology_topdown
        return build_bvh_topology_topdown(tri2vtx, vtx2xyz)


def aabbs_tri(
        tri2vtx: numpy.typing.NDArray,
        vtx2xyz0: numpy.typing.NDArray,
        bvhnodes: numpy.typing.NDArray,
        aabbs=None,
        i_bvhnode_root=0):
    from del_msh_numpy.BVH import aabb_uniform_mesh
    return aabb_uniform_mesh(tri2vtx, vtx2xyz0, bvhnodes, aabbs, i_bvhnode_root)


def bvhnodes_vtxedgetri(
        edge2vtx: numpy.typing.NDArray,
        tri2vtx: numpy.typing.NDArray,
        vtx2xyz: numpy.typing.NDArray) \
        -> (numpy.typing.NDArray, typing.List[int]):
    vtx2center = vtx2xyz.copy()
    edge2center = (vtx2xyz[edge2vtx[:, 0], :] + vtx2xyz[edge2vtx[:, 1], :]) / 2
    tri2center = (vtx2xyz[tri2vtx[:, 0], :] + vtx2xyz[tri2vtx[:, 1], :] + vtx2xyz[tri2vtx[:, 2], :]) / 3
    del_msh_numpy.fit_into_unit_cube(vtx2center)
    del_msh_numpy.fit_into_unit_cube(edge2center)
    del_msh_numpy.fit_into_unit_cube(tri2center)
    from .del_msh_numpy import build_bvh_topology_morton
    bvhnodes_vtx = build_bvh_topology_morton(vtx2center)
    bvhnodes_edge = build_bvh_topology_morton(edge2center)
    bvhnodes_tri = build_bvh_topology_morton(tri2center)
    # print(vtx2xyz.shape, edge2vtx.shape, tri2vtx.shape)
    # print(bvhnodes_vtx.shape, bvhnodes_edge.shape, bvhnodes_tri.shape)
    from .del_msh_numpy import shift_bvhnodes
    shift_bvhnodes(bvhnodes_edge, bvhnodes_vtx.shape[0], 0)
    shift_bvhnodes(bvhnodes_tri, bvhnodes_vtx.shape[0] + bvhnodes_edge.shape[0], 0)
    bvhnodes = numpy.vstack([bvhnodes_vtx, bvhnodes_edge, bvhnodes_tri])
    return bvhnodes, [0, bvhnodes_vtx.shape[0], bvhnodes_vtx.shape[0] + bvhnodes_edge.shape[0]]


def aabb_vtxedgetri(
        edge2vtx,
        tri2vtx,
        vtx2xyz0,
        bvhnodes,
        roots: typing.List[int],
        aabbs=None,
        vtx2xyz1=None):
    """
    :param edge2vtx:
    :param tri2vtx:
    :param vtx2xyz0:
    :param bvhnodes:
    :param roots: list of root bvhnode indices (vertex, edge, tri)
    :param aabbs: (optional) provide numpy array if you want to update
    :param vtx2xyz1: (optinoal) for Continuous Collision Detection (CCD)
    :return:
    """
    assert len(roots) == 3
    from del_msh_numpy.BVH import aabb_uniform_mesh
    # vertex
    aabbs = aabb_uniform_mesh(
        numpy.zeros((0, 0), dtype=numpy.uint64), vtx2xyz0, bvhnodes,
        aabbs=aabbs, root=roots[0], vtx2xyz1=vtx2xyz1)
    # edge
    aabbs = aabb_uniform_mesh(
        edge2vtx, vtx2xyz0, bvhnodes,
        aabbs=aabbs, root=roots[0], vtx2xyz1=vtx2xyz1)
    # triangle
    aabbs = aabb_uniform_mesh(
        tri2vtx, vtx2xyz0, bvhnodes,
        aabbs=aabbs, root=roots[0], vtx2xyz1=vtx2xyz1)
    return aabbs


# -------------------------------------
# search related

# TODO: make brute force version
def ccd_intersection_time(
        edge2vtx,
        tri2vtx,
        vtx2xyz0,
        vtx2xyz1,
        bvhnodes: typing.Optional[numpy.typing.NDArray] = None,
        aabbs: typing.Optional[numpy.typing.NDArray] = None,
        roots: typing.Optional[typing.List[int]] = None):
    from .del_msh_numpy import ccd_intersection_time
    if not bvhnodes is None:
        assert bvhnodes.shape[0] == aabbs.shape[0]
        assert vtx2xyz0.shape == vtx2xyz1.shape
        assert bvhnodes.shape[0] == vtx2xyz0.shape[0] * 2 - 1 + edge2vtx.shape[0] * 2 - 1 + tri2vtx.shape[0] * 2 - 1
        return ccd_intersection_time(
            edge2vtx, tri2vtx, vtx2xyz0, vtx2xyz1,
            bvhnodes, aabbs, roots)
    else:
        return ccd_intersection_time(
            edge2vtx, tri2vtx, vtx2xyz0, vtx2xyz1,
            numpy.zeros((0, 3), dtype=numpy.uint64),
            numpy.zeros((0, 6), dtype=vtx2xyz0.dtype), [])


def first_intersection_ray(
        src,
        dir,
        tri2vtx: numpy.typing.NDArray,
        vtx2xyz: numpy.typing.NDArray) -> [numpy.typing.NDArray,int]:
    """
    compute first intersection of a ray against a triangle mesh
    :param src: source of ray
    :param dir: direction of ray (can be un-normalized vector)
    :param tri2vtx: triangle index
    :param vtx2xyz: vertex positions
    :return: position and triangle index
    """
    from .del_msh_numpy import first_intersection_ray_meshtri3
    return first_intersection_ray_meshtri3(src, dir, tri2vtx, vtx2xyz)


def pick_vertex(
        tri2vtx: numpy.typing.NDArray,
        vtx2xyz: numpy.typing.NDArray,
        src: numpy.typing.NDArray,
        dir: numpy.typing.NDArray):
    from .del_msh_numpy import pick_vertex_meshtri3
    return pick_vertex_meshtri3(tri2vtx, vtx2xyz, src, dir)


def self_intersection(
        tri2vtx: numpy.typing.NDArray,
        vtx2xyz: numpy.typing.NDArray,
        bvhnodes: typing.Optional[numpy.typing.NDArray] = None,
        aabbs: typing.Optional[numpy.typing.NDArray] = None,
        i_bvhnode_root: typing.Optional[int] = 0):
    if vtx2xyz.shape[1] == 3:
        from .del_msh_numpy import intersection_trimesh3
        if bvhnodes is None:
            return intersection_trimesh3(
                tri2vtx, vtx2xyz,
                numpy.zeros((0,3), dtype=numpy.uint64),
                numpy.zeros((0,6), dtype=vtx2xyz.dtype),
                0)
        else:
            assert bvhnodes.shape[1] == 3
            assert aabbs.shape[1] == 6
            return intersection_trimesh3(
                tri2vtx, vtx2xyz,
                bvhnodes, aabbs, i_bvhnode_root)


def contacting_pair(
        tri2vtx: numpy.typing.NDArray,
        vtx2xyz: numpy.typing.NDArray,
        edge2vtx: numpy.typing.NDArray,
        threshold: float):
    from .del_msh_numpy import contacting_pair
    return contacting_pair(tri2vtx, vtx2xyz, edge2vtx, threshold)

# above: search intersection
# --------------------------------------