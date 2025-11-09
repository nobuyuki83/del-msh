use del_cudarc_sys::{cu, CuVec, LaunchConfig};
pub fn from_uniform_mesh(
    stream: cu::CUstream,
    elem2vtx: &CuVec<u32>,
    num_elem: usize,
    num_vtx: usize,
    is_self: bool,
) -> (CuVec<u32>, CuVec<u32>) {
    let num_node = elem2vtx.n / num_elem;
    assert_eq!(elem2vtx.n, num_elem * num_node);
    let (vtx2jdx, jdx2elem) =
        crate::vtx2elem::from_uniform_mesh(stream, elem2vtx, num_elem, num_vtx);
    let num_buff = if is_self {
        jdx2elem.n * num_node
    } else {
        jdx2elem.n * (num_node - 1)
    };
    let jdx2buff = CuVec::<u32>::with_capacity(num_buff).unwrap();
    let vtx2nvtx = CuVec::<u32>::with_capacity(num_vtx + 1).unwrap();
    vtx2nvtx.set_zeros(stream).unwrap();
    {
        /*
        let (func, _mdl) = del_cudarc_sys::load_function_in_module(
            del_msh_cuda_kernel::VTX2VTX,
            "vtx2nvtx_from_uniform_mesh",
        )
        .unwrap();
         */
        // let func = crate::load_get_function("vtx2vtx", "vtx2nvtx_from_uniform_mesh").unwrap();
        let func = del_cudarc_sys::cache_func::get_function_cached(
            "del_msh::vtx2vtx",
            del_msh_cuda_kernels::get("vtx2vtx").unwrap(),
            "vtx2nvtx_from_uniform_mesh",
        )
        .unwrap();
        let mut builder = del_cudarc_sys::Builder::new(stream);
        builder.arg_u32(num_vtx as u32);
        builder.arg_dptr(elem2vtx.dptr);
        builder.arg_u32(num_node as u32);
        builder.arg_bool(is_self);
        builder.arg_dptr(vtx2jdx.dptr);
        builder.arg_dptr(jdx2elem.dptr);
        builder.arg_dptr(vtx2nvtx.dptr);
        builder.arg_dptr(jdx2buff.dptr);
        builder
            .launch_kernel(func, LaunchConfig::for_num_elems(num_vtx as u32))
            .unwrap();
    }
    let vtx2idx = CuVec::<u32>::with_capacity(vtx2nvtx.n).unwrap();
    del_cudarc_sys::cumsum::exclusive_scan(stream, &vtx2nvtx, &vtx2idx);
    let num_idx = vtx2idx.last().unwrap();
    let idx2vtx: CuVec<u32> = CuVec::with_capacity(num_idx as usize).unwrap();
    {
        /*
        let (func, _mdl) = del_cudarc_sys::load_function_in_module(
            del_msh_cuda_kernel::VTX2VTX,
            "idx2vtx_from_vtx2buff_for_uniform_mesh",
        )
        .unwrap();
         */
        /*
        let func =
            crate::load_get_function("vtx2vtx", "idx2vtx_from_vtx2buff_for_uniform_mesh").unwrap();
         */
        let func = del_cudarc_sys::cache_func::get_function_cached(
            "del_msh::vtx2vtx",
            del_msh_cuda_kernels::get("vtx2vtx").unwrap(),
            "idx2vtx_from_vtx2buff_for_uniform_mesh",
        )
        .unwrap();
        let mut builder = del_cudarc_sys::Builder::new(stream);
        builder.arg_u32(num_vtx as u32);
        builder.arg_dptr(vtx2jdx.dptr);
        builder.arg_u32(num_node as u32);
        builder.arg_bool(is_self);
        builder.arg_dptr(vtx2idx.dptr);
        builder.arg_dptr(jdx2buff.dptr);
        builder.arg_dptr(idx2vtx.dptr);
        builder
            .launch_kernel(func, LaunchConfig::for_num_elems(num_vtx as u32))
            .unwrap();
    }
    (vtx2idx, idx2vtx)
}

pub fn multiply_graph_laplacian(
    stream: cu::CUstream,
    vtx2idx_offset: &CuVec<u32>,
    idx2vtx: &CuVec<u32>,
    num_vdim: usize,
    vtx2rhs: &CuVec<f32>,
    vtx2lhs: &CuVec<f32>,
) {
    let num_vtx = vtx2idx_offset.n - 1;
    assert_eq!(vtx2rhs.n, num_vtx * num_vdim);
    assert_eq!(vtx2lhs.n, num_vtx * num_vdim);
    /*
    let (func, _mdl) = del_cudarc_sys::load_function_in_module(
        del_msh_cuda_kernel::VTX2VTX,
        "multiply_graph_laplacian",
    )
    .unwrap();
     */
    // let func = crate::load_get_function("vtx2vtx", "multiply_graph_laplacian").unwrap();
    let func = del_cudarc_sys::cache_func::get_function_cached(
        "del_msh::vtx2vtx",
        del_msh_cuda_kernels::get("vtx2vtx").unwrap(),
        "multiply_graph_laplacian",
    )
    .unwrap();
    let mut builder = del_cudarc_sys::Builder::new(stream);
    builder.arg_u32(num_vtx as u32);
    builder.arg_dptr(vtx2idx_offset.dptr);
    builder.arg_dptr(idx2vtx.dptr);
    builder.arg_u32(num_vdim as u32);
    builder.arg_dptr(vtx2rhs.dptr);
    builder.arg_dptr(vtx2lhs.dptr);
    builder
        .launch_kernel(func, LaunchConfig::for_num_elems(num_vtx as u32))
        .unwrap();
}
