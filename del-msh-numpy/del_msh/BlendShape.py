import numpy
import cvxopt
import pyrr
from cvxopt import matrix


def direct_manipulation_delta(shape2pos, markers: dict[int, pyrr.Matrix44]):
    """
    :param shape2pos:
    :param markers: ivtx -> [mvp, screen pos]
    :return:
    """
    num_shape = shape2pos.shape[0]
    assert num_shape >= 1
    num_dweight = num_shape - 1
    if len(markers) == 0:
        weights = numpy.zeros(shape=[num_dweight], dtype=numpy.float32)
        return weights
    #
    B = []
    for vtx_marker in markers.keys():
        mvp44 = markers[vtx_marker][0]
        pos_0 = shape2pos[0].reshape(-1, 3)[vtx_marker].copy()
        pos_0 = mvp44.dot(numpy.append(pos_0, 1.0))[0:2]  # screen position of the marker
        for idx_weight in range(num_dweight):
            idx_shape = idx_weight + 1
            pos_i = shape2pos[idx_shape].reshape(-1, 3)[vtx_marker].copy()  # 3D position of the marker
            pos_i = mvp44.dot(numpy.append(pos_i, 1.0))[0:2]  # screen position of the marker
            B.append(pos_i - pos_0)
    B = numpy.stack(B).transpose().reshape(-1, num_dweight)
    #
    T = []
    for vtx_marker in markers.keys():
        mvp44 = markers[vtx_marker][0]
        pos_0 = shape2pos[0].reshape(-1, 3)[vtx_marker].copy()
        pos_0 = mvp44.dot(numpy.append(pos_0, 1.0))[0:2]  # screen position of the marker
        T.append(markers[vtx_marker][1]-pos_0)
    T = numpy.array(T).transpose().flatten()
    #
    P = B.transpose().dot(B) + numpy.eye(num_dweight) * 0.001
    q = -B.transpose().dot(T)
    A = numpy.ones((0, num_dweight)).astype(numpy.double)
    b = numpy.ones((0,)).reshape(0, 1)
    G = numpy.vstack([numpy.eye(num_dweight), -numpy.eye(num_dweight)]).astype(numpy.double)
    h = numpy.vstack([numpy.ones((num_dweight, 1)), numpy.zeros((num_dweight, 1))]).astype(numpy.double)
    sol = cvxopt.solvers.qp(P=matrix(P), q=matrix(q),
                            A=matrix(A), b=matrix(b),
                            G=matrix(G), h=matrix(h))
    return numpy.array(sol['x'], dtype=numpy.float32).transpose().flatten()


def direct_manipulation_absolute(shape2pos, markers):
    num_shape = shape2pos.shape[0]
    if len(markers) == 0:
        weights = numpy.zeros(shape=[num_shape], dtype=numpy.float32)
        weights[0] = 1.
        return weights
    #
    B = []
    for vtx_marker in markers.keys():
        for idx_shape in range(num_shape):
            pos0 = shape2pos[idx_shape].reshape(-1, 3)[vtx_marker].copy()
            pos0 = markers[vtx_marker][0].dot(numpy.append(pos0, 1.0))[0:2]
            B.append(pos0)
    B = numpy.vstack(B).transpose().reshape(-1, num_shape)
    #
    T = []
    for vtx_marker in markers.keys():
        T.append(markers[vtx_marker][1])
    T = numpy.array(T).transpose().flatten()
    #
    P = B.transpose().dot(B) + numpy.eye(num_shape) * 0.001
    q = -B.transpose().dot(T)
    A = numpy.ones((1, num_shape)).astype(numpy.double)
    b = numpy.array([1.]).reshape(1, 1)
    G = numpy.vstack([numpy.eye(num_shape), -numpy.eye(num_shape)]).astype(numpy.double)
    h = numpy.vstack([numpy.ones((num_shape, 1)), numpy.zeros((num_shape, 1))]).astype(numpy.double)
    sol = cvxopt.solvers.qp(P=matrix(P), q=matrix(q),
                            A=matrix(A), b=matrix(b),
                            G=matrix(G), h=matrix(h))
    return numpy.array(sol['x'], dtype=numpy.float32)