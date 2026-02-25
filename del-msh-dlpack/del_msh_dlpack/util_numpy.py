import numpy


def assert_shape_dtype(t: numpy.ndarray, shape: tuple[int,...], dtype: numpy.dtype):
    assert t.shape == shape
    assert t.dtype == dtype
    assert t.flags['C_CONTIGUOUS']