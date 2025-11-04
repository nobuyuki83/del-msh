import del_msh_dlpack

def test_0():
    a = del_msh_dlpack.get_cuda_driver_version()
    print("cuda_version:",a)