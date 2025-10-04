use del_cudarc_sys::{cu, CuVec, LaunchConfig};
pub fn from_uniform_mesh(
    stream: cu::CUstream,
    elem2vtx: &CuVec<u32>,
    num_elem: usize,
    num_vtx: usize,
    is_self: bool,
) -> (CuVec<u32>, CuVec<u32>) {
    let num_node = elem2vtx.n / num_elem;
    assert_eq!(elem2vtx.n % num_vtx, 0);
    let (vtx2jdx, jdx2elem) =
        crate::vtx2elem::from_uniform_mesh(stream, elem2vtx, num_elem, num_vtx);
    let num_buff = if is_self {
        jdx2elem.n * num_node
    } else {
        jdx2elem.n * (num_node - 1)
    };
    let jdx2buff = CuVec::<u32>::with_capacity(num_buff);
    let vtx2nvtx = CuVec::<u32>::with_capacity(num_vtx + 1);
    vtx2nvtx.set_zeros(stream);
    {
        let (func, _mdl) = del_cudarc_sys::load_function_in_module(
            del_msh_cuda_kernel::VTX2VTX,
            "vtx2nvtx_from_uniform_mesh",
        );
        let mut builder = del_cudarc_sys::Builder::new(stream);
        builder.arg_i32(num_vtx as i32);
        builder.arg_dptr(elem2vtx.dptr);
        builder.arg_i32(num_node as i32);
        builder.arg_bool(is_self);
        builder.arg_dptr(vtx2jdx.dptr);
        builder.arg_dptr(jdx2elem.dptr);
        builder.arg_dptr(vtx2nvtx.dptr);
        builder.arg_dptr(jdx2buff.dptr);
        builder.launch_kernel(func, LaunchConfig::for_num_elems(num_vtx as u32));
    }
    let vtx2idx = del_cudarc_sys::cumsum::cumsum(stream, &vtx2nvtx);
    let num_idx = vtx2idx.last();
    let idx2vtx: CuVec<u32> = CuVec::with_capacity(num_idx as usize);
    {
        let (func, _mdl) = del_cudarc_sys::load_function_in_module(
            del_msh_cuda_kernel::VTX2VTX,
            "idx2vtx_from_vtx2buff_for_uniform_mesh",
        );
        let mut builder = del_cudarc_sys::Builder::new(stream);
        builder.arg_i32(num_vtx as i32);
        builder.arg_dptr(vtx2jdx.dptr);
        builder.arg_i32(num_node as i32);
        builder.arg_bool(is_self);
        builder.arg_dptr(vtx2idx.dptr);
        builder.arg_dptr(jdx2buff.dptr);
        builder.arg_dptr(idx2vtx.dptr);
        builder.launch_kernel(func, LaunchConfig::for_num_elems(num_vtx as u32));
    }
    (vtx2idx, idx2vtx)
}
