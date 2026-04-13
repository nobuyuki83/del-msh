use del_dlpack::{
    check_2d_tensor as chk2, dlpack, get_managed_tensor_from_pyany as get_tensor,
    get_shape_tensor as shape, slice, slice_mut,
};
use pyo3::{prelude::PyModule, pyfunction, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(trimesh3_tri2normal, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(trimesh3_bwd_tri2normal, m)?)?;
    Ok(())
}

#[pyfunction]
pub fn trimesh3_tri2normal(
    _py: Python<'_>,
    tri2vtx: &Bound<'_, PyAny>,
    vtx2xyz: &Bound<'_, PyAny>,
    tri2nrm: &Bound<'_, PyAny>,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let tri2vtx = get_tensor(tri2vtx)?;
    let vtx2xyz = get_tensor(vtx2xyz)?;
    let tri2nrm = get_tensor(tri2nrm)?;
    let device_type = tri2vtx.ctx.device_type;
    let num_tri = shape(tri2vtx, 0).unwrap();
    let num_vtx = shape(vtx2xyz, 0).unwrap();
    //
    chk2::<u32>(tri2vtx, num_tri, 3, device_type).unwrap();
    chk2::<f32>(vtx2xyz, num_vtx, 3, device_type).unwrap();
    chk2::<f32>(tri2nrm, num_tri, 3, device_type).unwrap();
    //
    match device_type {
        dlpack::device_type_codes::CPU => {
            del_msh_cpu::trimesh3::tri2normal::<f32, u32>(
                slice!(tri2vtx, u32).unwrap(),
                slice!(vtx2xyz, f32).unwrap(),
                slice_mut!(tri2nrm, f32).unwrap(),
            );
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            use del_cudarc_sys::{cu, cuda_check};
            cuda_check!(cu::cuInit(0)).unwrap();
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            let function = del_cudarc_sys::cache_func::get_function_cached(
                "del_msh::trimesh3",
                del_msh_cuda_kernels::get("trimesh3").unwrap(),
                "tri2normal",
            )
            .unwrap();
            let mut builder = del_cudarc_sys::Builder::new(stream);
            builder.arg_u32(num_tri as u32);
            builder.arg_data(&tri2vtx.data);
            builder.arg_data(&vtx2xyz.data);
            builder.arg_data(&tri2nrm.data);
            builder
                .launch_kernel(
                    function,
                    del_cudarc_sys::LaunchConfig::for_num_elems(num_tri as u32),
                )
                .unwrap();
        }
        _ => {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "GPU not supported (compile with --features cuda)",
            ));
        }
    }
    Ok(())
}

#[pyfunction]
pub fn trimesh3_bwd_tri2normal(
    _py: Python<'_>,
    tri2vtx: &Bound<'_, PyAny>,
    vtx2xyz: &Bound<'_, PyAny>,
    dw_tri2nrm: &Bound<'_, PyAny>,
    dw_vtx2xyz: &Bound<'_, PyAny>,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let tri2vtx = get_tensor(tri2vtx)?;
    let vtx2xyz = get_tensor(vtx2xyz)?;
    let dw_tri2nrm = get_tensor(dw_tri2nrm)?;
    let dw_vtx2xyz = get_tensor(dw_vtx2xyz)?;
    //
    let device_type = tri2vtx.ctx.device_type;
    let num_tri = shape(tri2vtx, 0).unwrap();
    let num_vtx = shape(vtx2xyz, 0).unwrap();
    //
    chk2::<u32>(tri2vtx, num_tri, 3, device_type).unwrap();
    chk2::<f32>(vtx2xyz, num_vtx, 3, device_type).unwrap();
    chk2::<f32>(dw_tri2nrm, num_tri, 3, device_type).unwrap();
    chk2::<f32>(dw_vtx2xyz, num_vtx, 3, device_type).unwrap();
    //
    match device_type {
        dlpack::device_type_codes::CPU => {
            del_msh_cpu::trimesh3::bwd_tri2normal::<u32>(
                slice!(tri2vtx, u32).unwrap(),
                slice!(vtx2xyz, f32).unwrap(),
                slice_mut!(dw_tri2nrm, f32).unwrap(),
                slice_mut!(dw_vtx2xyz, f32).unwrap(),
            );
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            use del_cudarc_sys::cu;
            use del_cudarc_sys::cuda_check;
            cuda_check!(cu::cuInit(0)).unwrap();
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            {
                let dptr: cu::CUdeviceptr = dw_vtx2xyz.data as cu::CUdeviceptr;
                cuda_check!(cu::cuMemsetD32Async(
                    dptr,
                    0,
                    (num_vtx * 3) as usize,
                    stream
                ))
                .unwrap();
                cuda_check!(cu::cuStreamSynchronize(stream)).unwrap();
            }
            {
                let function = del_cudarc_sys::cache_func::get_function_cached(
                    "del_msh::trimesh3",
                    del_msh_cuda_kernels::get("trimesh3").unwrap(),
                    "bwd_tri2normal",
                )
                .unwrap();
                let mut builder = del_cudarc_sys::Builder::new(stream);
                builder.arg_u32(num_tri as u32);
                builder.arg_data(&tri2vtx.data);
                builder.arg_data(&vtx2xyz.data);
                builder.arg_data(&dw_tri2nrm.data);
                builder.arg_data(&dw_vtx2xyz.data);
                builder
                    .launch_kernel(
                        function,
                        del_cudarc_sys::LaunchConfig::for_num_elems(num_tri as u32),
                    )
                    .unwrap();
            }
        }
        _ => {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "GPU not supported (compile with --features cuda)",
            ))
        }
    }
    Ok(())
}
