use del_cudarc_sys::{cu, CuVec, LaunchConfig};
pub fn from_uniform_mesh(
    stream: cu::CUstream,
    elem2vtx: &CuVec<u32>,
    num_elem: usize,
    num_vtx: usize,
) -> (CuVec<u32>, CuVec<u32>) {
    let num_node = elem2vtx.n / num_elem;
    assert_eq!(elem2vtx.n, num_elem * num_node);
    let vtx2valence: CuVec<u32> = CuVec::with_capacity(num_vtx + 1).unwrap();
    vtx2valence.set_zeros(stream).unwrap();
    {
        /*
        let (func, _mdl) =
            del_cudarc_sys::load_function_in_module(del_msh_cuda_kernel::UNIFORM_MESH, "vtx2nelem")
                .unwrap();
         */
        let func = crate::load_get_function("uniform_mesh", "vtx2nelem").unwrap();
        let mut builder = del_cudarc_sys::Builder::new(stream);
        builder.arg_u32(num_elem as u32);
        builder.arg_dptr(elem2vtx.dptr);
        builder.arg_u32(num_node as u32);
        builder.arg_dptr(vtx2valence.dptr);
        builder
            .launch_kernel(func, LaunchConfig::for_num_elems(num_elem as u32))
            .unwrap();
    }
    let vtx2idx0 = CuVec::<u32>::with_capacity(vtx2valence.n).unwrap();
    del_cudarc_sys::cumsum::exclusive_scan(stream, &vtx2valence, &vtx2idx0);
    // dbg!(vtx2idx0.copy_to_host());
    let num_idx = vtx2idx0.last().unwrap();
    let idx2elem: CuVec<u32> = CuVec::with_capacity(num_idx as usize).unwrap();
    {
        /*
        let (func, _mdl) = del_cudarc_sys::load_function_in_module(
            del_msh_cuda_kernel::UNIFORM_MESH,
            "fill_idx2elem",
        )
        .unwrap();
         */
        let func = crate::load_get_function("uniform_mesh", "fill_idx2elem").unwrap();
        let mut builder = del_cudarc_sys::Builder::new(stream);
        builder.arg_u32(num_elem as u32);
        builder.arg_dptr(elem2vtx.dptr);
        builder.arg_u32(num_node as u32);
        builder.arg_dptr(vtx2idx0.dptr);
        builder.arg_dptr(idx2elem.dptr);
        builder
            .launch_kernel(func, LaunchConfig::for_num_elems(num_elem as u32))
            .unwrap();
    }
    // dbg!(vtx2idx0.copy_to_host());
    let vtx2idx = del_cudarc_sys::array1d::shift_array_right(stream, &vtx2idx0);
    del_cudarc_sys::offset_array::sort(stream, &vtx2idx, &idx2elem);
    (vtx2idx, idx2elem)
}
