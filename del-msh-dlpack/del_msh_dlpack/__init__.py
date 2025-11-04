class _CapsuleAsDLPack:
    def __init__(self, cap):
        self._cap = cap
        self._used = False

    def __dlpack__(self, *, max_version=None, stream=None):
        if self._used:
            raise RuntimeError("DLPack capsule already consumed")
        self._used = True
        return self._cap

    def __dlpack_device__(self):
        # CPU: (kDLCPU=1, device_id=0)
        return (1, 0)


def get_cuda_driver_version():
    from .del_msh_dlpack import get_cuda_driver_version

    return get_cuda_driver_version()





