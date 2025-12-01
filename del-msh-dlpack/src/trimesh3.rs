use del_dlpack::dlpack;
use pyo3::prelude::PyModule;
use pyo3::{pyfunction, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(trimesh3_tri2normal, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(trimesh3_bwd_tri2normal, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(trimesh3_load_nastran, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(trimesh3_save_wavefront_obj, m)?)?;
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
    let tri2vtx = del_dlpack::get_managed_tensor_from_pyany(tri2vtx)?;
    let vtx2xyz = del_dlpack::get_managed_tensor_from_pyany(vtx2xyz)?;
    let tri2nrm = del_dlpack::get_managed_tensor_from_pyany(tri2nrm)?;
    let device_type = tri2vtx.ctx.device_type;
    let num_tri = del_dlpack::get_shape_tensor(tri2vtx, 0).unwrap();
    let num_vtx = del_dlpack::get_shape_tensor(vtx2xyz, 0).unwrap();
    //
    del_dlpack::check_2d_tensor::<u32>(tri2vtx, num_tri, 3, device_type).unwrap();
    del_dlpack::check_2d_tensor::<f32>(vtx2xyz, num_vtx, 3, device_type).unwrap();
    del_dlpack::check_2d_tensor::<f32>(tri2nrm, num_tri, 3, device_type).unwrap();
    //
    match device_type {
        dlpack::device_type_codes::CPU => {
            let tri2vtx = unsafe { del_dlpack::slice_from_tensor::<u32>(tri2vtx).unwrap() };
            let vtx2xyz = unsafe { del_dlpack::slice_from_tensor::<f32>(vtx2xyz).unwrap() };
            let tri2nrm = unsafe { del_dlpack::slice_from_tensor_mut::<f32>(tri2nrm).unwrap() };
            del_msh_cpu::trimesh3::tri2normal::<f32, u32>(tri2vtx, vtx2xyz, tri2nrm);
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            //println!("GPU_{}", tri2vtx.ctx.device_id);
            /*
            let (function, _module) = del_cudarc_sys::load_function_in_module(
                del_msh_cuda_kernel::TRIMESH3,
                "tri2normal",
            )
            .unwrap();
             */
            // let function = crate::load_get_function("trimesh3", "tri2normal").unwrap();
            let function = del_cudarc_sys::cache_func::get_function_cached(
                "del_msh::trimesh3",
                del_msh_cuda_kernels::get("trimesh3").unwrap(),
                "tri2normal",
            )
            .unwrap();
            use del_cudarc_sys::{cu, cuda_check};
            cuda_check!(cu::cuInit(0)).unwrap();
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
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
            println!("Unknown device type {}", device_type);
            todo!()
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
    let tri2vtx = del_dlpack::get_managed_tensor_from_pyany(tri2vtx)?;
    let vtx2xyz = del_dlpack::get_managed_tensor_from_pyany(vtx2xyz)?;
    let dw_tri2nrm = del_dlpack::get_managed_tensor_from_pyany(dw_tri2nrm)?;
    let dw_vtx2xyz = del_dlpack::get_managed_tensor_from_pyany(dw_vtx2xyz)?;
    //
    let device_type = tri2vtx.ctx.device_type;
    let num_tri = del_dlpack::get_shape_tensor(tri2vtx, 0).unwrap();
    let num_vtx = del_dlpack::get_shape_tensor(vtx2xyz, 0).unwrap();
    //
    del_dlpack::check_2d_tensor::<u32>(tri2vtx, num_tri, 3, device_type).unwrap();
    del_dlpack::check_2d_tensor::<f32>(vtx2xyz, num_vtx, 3, device_type).unwrap();
    del_dlpack::check_2d_tensor::<f32>(dw_tri2nrm, num_tri, 3, device_type).unwrap();
    del_dlpack::check_2d_tensor::<f32>(dw_vtx2xyz, num_vtx, 3, device_type).unwrap();
    //
    match device_type {
        dlpack::device_type_codes::CPU => {
            let tri2vtx = unsafe { del_dlpack::slice_from_tensor::<u32>(tri2vtx).unwrap() };
            let vtx2xyz = unsafe { del_dlpack::slice_from_tensor::<f32>(vtx2xyz).unwrap() };
            let dw_tri2nrm =
                unsafe { del_dlpack::slice_from_tensor_mut::<f32>(dw_tri2nrm).unwrap() };
            let dw_vtx2xyz =
                unsafe { del_dlpack::slice_from_tensor_mut::<f32>(dw_vtx2xyz).unwrap() };
            del_msh_cpu::trimesh3::bwd_tri2normal::<u32>(tri2vtx, vtx2xyz, dw_tri2nrm, dw_vtx2xyz);
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
                // println!("GPU_{}", tri2vtx.ctx.device_id);
                /*
                let (function, _module) = del_cudarc_sys::load_function_in_module(
                    del_msh_cuda_kernel::TRIMESH3,
                    "bwd_tri2normal",
                )
                .unwrap();
                 */
                //let function = crate::load_get_function("trimesh3", "bwd_tri2normal").unwrap();
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
            todo!()
        }
    }
    Ok(())
}

#[pyfunction]
fn trimesh3_load_nastran(
    py: Python<'_>,
    path: String,
) -> PyResult<(pyo3::Py<PyAny>, pyo3::Py<PyAny>)> {
    let (tri2vtx, vtx2xyz) = del_msh_cpu::io_nas::load_tri_mesh::<_, u32>(path);
    let tri2vtx_cap =
        del_dlpack::make_capsule_from_vec(py, vec![(tri2vtx.len() as i64) / 3, 3], tri2vtx);
    let vtx2xyz_cap =
        del_dlpack::make_capsule_from_vec(py, vec![(vtx2xyz.len() as i64) / 3, 3], vtx2xyz);
    Ok((tri2vtx_cap, vtx2xyz_cap))
}

#[del_dlpack::pyo3::pyfunction]
pub fn trimesh3_save_wavefront_obj(
    _py: Python<'_>,
    tri2vtx: &Bound<'_, PyAny>,
    vtx2xyz: &Bound<'_, PyAny>,
    path: String,
) -> PyResult<()> {
    let tri2vtx = del_dlpack::get_managed_tensor_from_pyany(tri2vtx)?;
    let vtx2xyz = del_dlpack::get_managed_tensor_from_pyany(vtx2xyz)?;
    let num_tri = del_dlpack::get_shape_tensor(tri2vtx, 0).unwrap();
    let num_vtx = del_dlpack::get_shape_tensor(vtx2xyz, 0).unwrap();
    //
    del_dlpack::check_2d_tensor::<u32>(tri2vtx, num_tri, 3, dlpack::device_type_codes::CPU)
        .unwrap();
    del_dlpack::check_2d_tensor::<f32>(vtx2xyz, num_vtx, 3, dlpack::device_type_codes::CPU)
        .unwrap();
    //
    let tri2vtx = unsafe { del_dlpack::slice_from_tensor::<u32>(tri2vtx) }.unwrap();
    let vtx2xyz = unsafe { del_dlpack::slice_from_tensor::<f32>(vtx2xyz) }.unwrap();
    del_msh_cpu::io_obj::save_tri2vtx_vtx2xyz(path, tri2vtx, vtx2xyz, 3).unwrap();
    Ok(())
}
