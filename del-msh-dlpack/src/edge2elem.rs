use del_dlpack::{
    check_1d_tensor as chk1, check_2d_tensor as chk2, dlpack,
    get_managed_tensor_from_pyany as get_tensor, get_shape_tensor as shape, slice, slice_mut,
};
use pyo3::{types::PyModule, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(
        edge2elem_from_edge2vtx_of_tri2vtx_with_vtx2vtx,
        m
    )?)?;
    Ok(())
}

#[pyo3::pyfunction]
pub fn edge2elem_from_edge2vtx_of_tri2vtx_with_vtx2vtx(
    _py: Python<'_>,
    edge2vtx: &Bound<'_, PyAny>,
    tri2vtx: &Bound<'_, PyAny>,
    vtx2idx_offset: &Bound<'_, PyAny>,
    idx2vtx: &Bound<'_, PyAny>,
    edge2tri: &Bound<'_, PyAny>,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let edge2vtx = get_tensor(edge2vtx)?;
    let tri2vtx = get_tensor(tri2vtx)?;
    let vtx2idx_offset = get_tensor(vtx2idx_offset)?;
    let idx2vtx = get_tensor(idx2vtx)?;
    let edge2tri = get_tensor(edge2tri)?;
    //
    let device = edge2vtx.ctx.device_type;
    let num_edge = shape(edge2vtx, 0).unwrap();
    let num_tri = shape(tri2vtx, 0).unwrap();
    let num_vtx = shape(vtx2idx_offset, 0).unwrap() - 1;
    let num_idx = shape(idx2vtx, 0).unwrap();
    //
    assert_eq!(num_edge, num_idx);
    chk2::<u32>(edge2vtx, num_edge, 2, device).unwrap();
    chk2::<u32>(tri2vtx, num_tri, 3, device).unwrap();
    chk1::<u32>(vtx2idx_offset, num_vtx + 1, device).unwrap();
    chk1::<u32>(idx2vtx, num_idx, device).unwrap();
    chk2::<u32>(edge2tri, num_edge, 2, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            del_msh_cpu::edge2elem::from_edge2vtx_of_tri2vtx_with_vtx2vtx(
                slice!(edge2vtx, u32).unwrap(),
                slice!(tri2vtx, u32).unwrap(),
                slice!(vtx2idx_offset, u32).unwrap(),
                slice!(idx2vtx, u32).unwrap(),
                slice_mut!(edge2tri, u32).unwrap(),
            );
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            use del_cudarc_sys::{cu, cuda_check};
            cuda_check!(cu::cuInit(0)).unwrap();
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            let func = del_cudarc_sys::cache_func::get_function_cached(
                "del_msh::edge2elem",
                del_msh_cuda_kernels::get("edge2elem").unwrap(),
                "edge2elem_from_edge2vtx_of_tri2vtx",
            )
            .unwrap();
            let mut builder = del_cudarc_sys::Builder::new(stream);
            builder.arg_u32(num_edge as u32);
            builder.arg_data(&edge2vtx.data);
            builder.arg_data(&tri2vtx.data);
            builder.arg_data(&vtx2idx_offset.data);
            builder.arg_data(&idx2vtx.data);
            builder.arg_data(&edge2tri.data);
            builder
                .launch_kernel(
                    func,
                    del_cudarc_sys::LaunchConfig::for_num_elems(num_edge as u32),
                )
                .unwrap();
        }
        _ => {}
    }
    Ok(())
}
