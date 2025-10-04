use del_cudarc_sys::{cu, CuVec, LaunchConfig};
pub fn from_uniform_mesh(
    stream: cu::CUstream,
    elem2vtx: &CuVec<u32>,
    num_elem: usize,
    num_vtx: usize,
) -> (CuVec<u32>, CuVec<u32>) {
    let num_node = elem2vtx.n / num_elem;
    assert_eq!(elem2vtx.n, num_elem * num_node);
    let vtx2valence: CuVec<u32> = CuVec::with_capacity(num_vtx + 1);
    vtx2valence.set_zeros(stream);
    {
        let (func, _mdl) =
            del_cudarc_sys::load_function_in_module(del_msh_cuda_kernel::UNIFORM_MESH, "vtx2nelem");
        let mut builder = del_cudarc_sys::Builder::new(stream);
        builder.arg_i32(num_elem as i32);
        builder.arg_dptr(elem2vtx.dptr);
        builder.arg_i32(num_node as i32);
        builder.arg_dptr(vtx2valence.dptr);
        builder.launch_kernel(func, LaunchConfig::for_num_elems(num_elem as u32));
    }
    let vtx2idx0 = del_cudarc_sys::cumsum::cumsum(stream, &vtx2valence);
    // dbg!(vtx2idx0.copy_to_host());
    let num_idx = vtx2idx0.last();
    let idx2elem: CuVec<u32> = CuVec::with_capacity(num_idx as usize);
    {
        let (func, _mdl) = del_cudarc_sys::load_function_in_module(
            del_msh_cuda_kernel::UNIFORM_MESH,
            "fill_idx2elem",
        );
        let mut builder = del_cudarc_sys::Builder::new(stream);
        builder.arg_i32(num_elem as i32);
        builder.arg_dptr(elem2vtx.dptr);
        builder.arg_i32(num_node as i32);
        builder.arg_dptr(vtx2idx0.dptr);
        builder.arg_dptr(idx2elem.dptr);
        builder.launch_kernel(func, LaunchConfig::for_num_elems(num_elem as u32));
    }
    // dbg!(vtx2idx0.copy_to_host());
    let vtx2idx = del_cudarc_sys::util::shift_array_right(stream, &vtx2idx0);
    del_cudarc_sys::util::sort_indexed_array(stream, &vtx2idx, &idx2elem);
    (vtx2idx, idx2elem)
}
