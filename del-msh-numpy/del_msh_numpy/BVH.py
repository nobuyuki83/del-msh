import numpy
import numpy.typing

def edges_of_aabb(aabb: numpy.typing.NDArray) \
        -> numpy.typing.NDArray:
    num_dim = aabb.shape[1] // 2
    assert aabb.shape[1] == num_dim * 2
    num_aabb = aabb.shape[0]
    if num_dim == 3:
        edge2node2xyz = numpy.zeros((num_aabb, 12, 2, num_dim), dtype=aabb.dtype)
        edge2node2xyz[:, 0, 0, :] = edge2node2xyz[:, 3, 1, :] = edge2node2xyz[:, 8, 0, :] = aabb[:, [0, 1, 2]]
        edge2node2xyz[:, 0, 1, :] = edge2node2xyz[:, 1, 0, :] = edge2node2xyz[:, 9, 0, :] = aabb[:, [3, 1, 2]]
        edge2node2xyz[:, 2, 1, :] = edge2node2xyz[:, 3, 0, :] = edge2node2xyz[:, 10, 0, :] = aabb[:, [0, 4, 2]]
        edge2node2xyz[:, 1, 1, :] = edge2node2xyz[:, 2, 0, :] = edge2node2xyz[:, 11, 0, :] = aabb[:, [3, 4, 2]]
        edge2node2xyz[:, 4, 0, :] = edge2node2xyz[:, 7, 1, :] = edge2node2xyz[:, 8, 1, :] = aabb[:, [0, 1, 5]]
        edge2node2xyz[:, 4, 1, :] = edge2node2xyz[:, 5, 0, :] = edge2node2xyz[:, 9, 1, :] = aabb[:, [3, 1, 5]]
        edge2node2xyz[:, 6, 1, :] = edge2node2xyz[:, 7, 0, :] = edge2node2xyz[:, 10, 1, :] = aabb[:, [0, 4, 5]]
        edge2node2xyz[:, 5, 1, :] = edge2node2xyz[:, 6, 0, :] = edge2node2xyz[:, 11, 1, :] = aabb[:, [3, 4, 5]]
        edge2node2xyz = edge2node2xyz.reshape(num_aabb * 12, 2, num_dim)
        return edge2node2xyz



def aabb_uniform_mesh(
        elem2vtx: numpy.typing.NDArray | None,
        vtx2xyz0: numpy.typing.NDArray,
        bvhnodes: numpy.typing.NDArray,
        aabbs: numpy.typing.NDArray | None,
        vtx2xyz1: numpy.typing.NDArray | None,
        root=0):
    """ compute Axis-Aligned Bounding Box (AABB) for elements of an uniform mesh
    :param elem2vtx: if `None` is provided, build aabb for vertices
    :param vtx2xyz0: list of vertex coordinate
    :param bvhnodes: BVH tree structure
    :param aabbs: (optional) provide numpy array if you want to update values
    :param vtx2xyz1: (optional) for Continuous Collision Detection (CCD)
    :param root: (optional) default is zero
    :return:
    """
    if aabbs is None:
        if vtx2xyz0.shape[1] == 3:
            aabbs = numpy.zeros((bvhnodes.shape[0], 6), dtype=vtx2xyz0.dtype)
        else:
            print("TODO", vtx2xyz0.shape)
    if vtx2xyz1 is None:
        vtx2xyz1 = numpy.zeros((0, 0), dtype=vtx2xyz0.dtype)
    if vtx2xyz0.dtype == numpy.float32:
        from .del_msh_numpy import build_bvh_geometry_aabb_uniformmesh_f32
        build_bvh_geometry_aabb_uniformmesh_f32(aabbs, bvhnodes, elem2vtx, vtx2xyz0, root, vtx2xyz1)
    elif vtx2xyz0.dtype == numpy.float64:
        from .del_msh_numpy import build_bvh_geometry_aabb_uniformmesh_f64
        build_bvh_geometry_aabb_uniformmesh_f64(aabbs, bvhnodes, elem2vtx, vtx2xyz0, root, vtx2xyz1)
    else:
        pass
    return aabbs
