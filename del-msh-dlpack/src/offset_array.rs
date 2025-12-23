use del_dlpack::dlpack;
use pyo3::{pyfunction, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<pyo3::types::PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(offset_array_aggregate, m)?)?;
    Ok(())
}

#[pyfunction]
fn offset_array_aggregate(
    _py: Python<'_>,
    idx2jdx_offset: &Bound<'_, PyAny>,
    jdx2kdx: &Bound<'_, PyAny>,
    kdx2val: &Bound<'_, PyAny>,
    idx2aggval: &Bound<'_, PyAny>,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let idx2jdx_offset = del_dlpack::get_managed_tensor_from_pyany(idx2jdx_offset)?;
    let jdx2kdx = del_dlpack::get_managed_tensor_from_pyany(jdx2kdx)?;
    let kdx2val = del_dlpack::get_managed_tensor_from_pyany(kdx2val)?;
    let idx2aggval = del_dlpack::get_managed_tensor_from_pyany(idx2aggval)?;
    //
    let num_idx = del_dlpack::get_shape_tensor(idx2jdx_offset, 0).unwrap() - 1;
    let num_jdx = del_dlpack::get_shape_tensor(jdx2kdx, 0).unwrap();
    let num_dim = del_dlpack::get_shape_tensor(idx2aggval, 1).unwrap();
    let device = idx2jdx_offset.ctx.device_type;
    //
    del_dlpack::check_1d_tensor::<u32>(idx2jdx_offset, num_idx + 1, device).unwrap();
    del_dlpack::check_1d_tensor::<u32>(jdx2kdx, num_jdx, device).unwrap();
    del_dlpack::check_2d_tensor::<f32>(kdx2val, num_jdx, num_dim, device).unwrap();
    del_dlpack::check_2d_tensor::<f32>(idx2aggval, num_idx, num_dim, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            let idx2jdx_offset =
                unsafe { del_dlpack::slice_from_tensor::<u32>(idx2jdx_offset) }.unwrap();
            let jdx2kdx = unsafe { del_dlpack::slice_from_tensor::<u32>(jdx2kdx) }.unwrap();
            let kdx2val = unsafe { del_dlpack::slice_from_tensor::<f32>(kdx2val) }.unwrap();
            let idx2aggval =
                unsafe { del_dlpack::slice_from_tensor_mut::<f32>(idx2aggval) }.unwrap();
            let num_dim = num_dim as usize;
            for idx in 0..num_idx as usize {
                for jdx in idx2jdx_offset[idx]..idx2jdx_offset[idx + 1] {
                    let kdx = jdx2kdx[jdx as usize] as usize;
                    for i_dim in 0..num_dim {
                        idx2aggval[idx * num_dim + i_dim] += kdx2val[kdx * num_dim + i_dim];
                    }
                }
            }
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            use del_cudarc_sys::{cu, cuda_check};
            cuda_check!(cu::cuInit(0)).unwrap();
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            let fnc = del_cudarc_sys::cache_func::get_function_cached(
                "del_cudarc::offset_array",
                del_cudarc_kernels::get("offset_array").unwrap(),
                "aggregate",
            )
            .unwrap();
            let cfg = del_cudarc_sys::LaunchConfig::for_num_elems(num_idx as u32);
            let mut builder = del_cudarc_sys::Builder::new(stream);
            builder.arg_u32(num_idx as u32);
            builder.arg_data(&idx2jdx_offset.data);
            builder.arg_data(&jdx2kdx.data);
            builder.arg_u32(num_dim as u32);
            builder.arg_data(&kdx2val.data);
            builder.arg_data(&idx2aggval.data);
            builder.launch_kernel(fnc, cfg).unwrap();
        }
        _ => {
            todo!();
        }
    };
    Ok(())
}
