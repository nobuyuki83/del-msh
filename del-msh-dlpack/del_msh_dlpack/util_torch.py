import torch

def to_dlpack_safe(t: torch.Tensor, stream_ptr: int | None = None):
    # DLPack側の制約を満たす
    assert not t.requires_grad
    assert t.is_contiguous()

    dev = t.device.type
    if dev == "cuda":
        # stream_ptr は「整数の CUDA ストリームハンドル」を期待
        # 既定ストリームで良ければ stream=1（CUDA）/0（ROCm）相当
        s = int(stream_ptr) if stream_ptr is not None else 1
        return t.__dlpack__(stream=s)
    elif dev == "cpu":
        # CPU では stream を渡してはいけない
        return t.__dlpack__()
    else:
        raise RuntimeError(f"Unsupported device for DLPack export: {t.device}")