import torch

def to_dlpack_safe(t: torch.Tensor, stream_ptr: int):
    assert not t.requires_grad
    assert t.is_contiguous()

    dev = t.device.type
    if dev == "cuda":
        return t.__dlpack__(stream=stream_ptr)
    elif dev == "cpu":
        return t.__dlpack__()
    else:
        raise RuntimeError(f"Unsupported device for DLPack export: {t.device}")