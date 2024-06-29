import numpy


''' 
// exact computation looks like:

hoge = numpy.zeros_like(self.vtx2xyz_new)
for i_vtx in range(self.vtx2xyz.shape[0]):
    w = self.weights[i_vtx]
    q0 = w.dot(self.samples_old)
    q1 = w.dot(self.samples_new)
    dq0 = self.samples_old - q0
    dq1 = self.samples_new - q1
    diag_w = numpy.diag(w)
    M = dq0.transpose().dot(diag_w.dot(dq0))
    M = numpy.linalg.inv(M).dot(dq0.transpose()).dot(diag_w).dot(dq1)
    p1 = (self.vtx2xyz[i_vtx]-q0).dot(M)+q1
    hoge[i_vtx] = p1
'''

def kernel(
        samples: numpy.typing.NDArray,
        vtx2xyz: numpy.typing.NDArray,
        p: int=2,
        eps: float=1.0e-3) -> numpy.ndarray:
    """
    compute the kernel value for each vertex against each sample point
    :param samples:
    :param vtx2xyz:
    :param p: the degree of the polynominal
    :param eps: epsilon value to avoid zero-division
    :return:
    """
    ws = numpy.ndarray((vtx2xyz.shape[0], samples.shape[0]))
    for i_vtx in range(vtx2xyz.shape[0]):
        p0 = vtx2xyz[i_vtx].copy()
        w = numpy.linalg.norm(samples[:] - p0, axis=1)
        w = 1.0 / (numpy.power(w, p) + eps)
        ws[i_vtx] = w / w.sum()
    return ws


def precomp(
        samples: numpy.typing.NDArray,
        vtx2xyz: numpy.typing.NDArray,
        weights: numpy.typing.NDArray):
    precomp = numpy.ndarray((vtx2xyz.shape[0], samples.shape[0]))
    if samples.shape[0] < 4:
        return
    for i_vtx in range(vtx2xyz.shape[0]):
        p0 = vtx2xyz[i_vtx].copy()
        w = weights[i_vtx]
        q0 = w.dot(samples)
        dq0 = samples - q0
        diag_w = numpy.diag(w)
        M = dq0.transpose().dot(diag_w.dot(dq0))
        precomp[i_vtx] = (p0 - q0).dot(numpy.linalg.inv(M)).dot(dq0.transpose()).dot(diag_w) + w
    return precomp
