import numpy


def tet_volume(
        p0,
        p1,
        p2,
        p3) -> float:
    """
    volume of tetrahedron in the right hand coordinate
    :param p0:
    :param p1:
    :param p2:
    :param p3:
    :return:
    """
    return (numpy.cross(p1 - p2, p2 - p0)).dot(p3 - p0) / 6.0
