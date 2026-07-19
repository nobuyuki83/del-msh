import torch


def to_dlpack_safe(t: torch.Tensor, stream_ptr: int):
    assert t.is_contiguous()

    dev = t.device.type
    if dev == "cuda":
        if stream_ptr == 0:
            # In the Python DLPack protocol, CUDA legacy default stream is
            # represented by 1, although cudaStream_t itself is 0.
            stream_ptr = 1
        return t.__dlpack__(stream=stream_ptr)
    elif dev == "cpu":
        return t.__dlpack__()
    else:
        raise RuntimeError(f"Unsupported device for DLPack export: {t.device}")


def assert_shape_dtype_device(
    t: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
):
    assert t.shape == shape
    assert t.dtype == dtype
    assert t.device == device
    assert t.is_contiguous()
    assert not t.requires_grad
