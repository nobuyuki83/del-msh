import numpy


def barycentric_coord(
        p0,
        p1,
        p2,
        q0):
    """
    barycentric coordinate of a 3D point on a 3D triangle
    :param p0: 0th vtx
    :param p1: 1st vtx
    :param p2: 2nd vtx
    :param q0: point on the triangle
    :return: barycentric coordinate as numpy.ndarray
    """
    from .Tet import tet_volume
    n = numpy.cross(p1 - p0, p2 - p0)
    q1 = n + q0
    v0 = tet_volume(q0, p1, p2, q1)
    v1 = tet_volume(q0, p2, p0, q1)
    v2 = tet_volume(q0, p0, p1, q1)
    inv_v012 = 1.0 / (v0 + v1 + v2)
    return numpy.array([v0 * inv_v012, v1 * inv_v012, v2 * inv_v012])
