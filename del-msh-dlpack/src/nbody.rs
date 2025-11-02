use pyo3::{pyfunction, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<pyo3::types::PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(nbody_screened_poisson, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(nbody_elastic, m)?)?;
    Ok(())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn nbody_screened_poisson(
    _py: Python<'_>,
    vtx2co: &Bound<'_, PyAny>,
    vtx2rhs: &Bound<'_, PyAny>,
    wtx2co: &Bound<'_, PyAny>,
    wtx2lhs: &Bound<'_, PyAny>,
    lambda: f32,
    epsilon: f32,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let vtx2co = crate::get_managed_tensor_from_pyany(vtx2co)?;
    let vtx2rhs = crate::get_managed_tensor_from_pyany(vtx2rhs)?;
    let wtx2co = crate::get_managed_tensor_from_pyany(wtx2co)?;
    let wtx2lhs = crate::get_managed_tensor_from_pyany(wtx2lhs)?;
    let num_vtx = crate::get_shape_tensor(vtx2co, 0).unwrap();
    let num_wtx = crate::get_shape_tensor(wtx2co, 0).unwrap();
    let num_dim = crate::get_shape_tensor(vtx2co, 1).unwrap();
    let num_vdim = crate::get_shape_tensor(vtx2rhs, 1).unwrap();
    let device = vtx2co.ctx.device_type;
    //
    crate::check_2d_tensor::<f32>(vtx2co, num_vtx, num_dim, device).unwrap();
    crate::check_2d_tensor::<f32>(vtx2rhs, num_vtx, num_vdim, device).unwrap();
    crate::check_2d_tensor::<f32>(wtx2co, num_wtx, num_dim, device).unwrap();
    crate::check_2d_tensor::<f32>(wtx2lhs, num_wtx, num_vdim, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            let vtx2co = unsafe { crate::slice_from_tensor::<f32>(vtx2co) }.unwrap();
            let vtx2rhs = unsafe { crate::slice_from_tensor::<f32>(vtx2rhs) }.unwrap();
            let wtx2co = unsafe { crate::slice_from_tensor::<f32>(wtx2co) }.unwrap();
            let wtx2lhs = unsafe { crate::slice_from_tensor_mut::<f32>(wtx2lhs) }.unwrap();
            del_msh_cpu::nbody::screened_poisson3(
                wtx2co, wtx2lhs, lambda, epsilon, vtx2co, vtx2rhs,
            );
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            use del_cudarc_sys::{cu, cuda_check};
            cuda_check!(cu::cuInit(0)).unwrap();
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            let (fnc, _mdl) = del_cudarc_sys::load_function_in_module(
                del_msh_cuda_kernel::NBODY,
                "screened_poisson3",
            )
            .unwrap();
            let mut builder = del_cudarc_sys::Builder::new(stream);
            builder.arg_u32(num_wtx as u32);
            builder.arg_data(&wtx2co.data);
            builder.arg_data(&wtx2lhs.data);
            builder.arg_u32(num_vtx as u32);
            builder.arg_data(&vtx2co.data);
            builder.arg_data(&vtx2rhs.data);
            builder.arg_f32(lambda);
            builder.arg_f32(epsilon);
            builder
                .launch_kernel(
                    fnc,
                    del_cudarc_sys::LaunchConfig::for_num_elems(num_wtx as u32),
                )
                .unwrap();
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn nbody_elastic(
    _py: Python<'_>,
    vtx2co: &Bound<'_, PyAny>,
    vtx2rhs: &Bound<'_, PyAny>,
    wtx2co: &Bound<'_, PyAny>,
    wtx2lhs: &Bound<'_, PyAny>,
    nu: f32,
    epsilon: f32,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let vtx2co = crate::get_managed_tensor_from_pyany(vtx2co)?;
    let vtx2rhs = crate::get_managed_tensor_from_pyany(vtx2rhs)?;
    let wtx2co = crate::get_managed_tensor_from_pyany(wtx2co)?;
    let wtx2lhs = crate::get_managed_tensor_from_pyany(wtx2lhs)?;
    let num_vtx = crate::get_shape_tensor(vtx2co, 0).unwrap();
    let num_wtx = crate::get_shape_tensor(wtx2co, 0).unwrap();
    let num_dim = crate::get_shape_tensor(vtx2co, 1).unwrap();
    let num_vdim = crate::get_shape_tensor(vtx2rhs, 1).unwrap();
    let device = vtx2co.ctx.device_type;
    //
    crate::check_2d_tensor::<f32>(vtx2co, num_vtx, num_dim, device).unwrap();
    crate::check_2d_tensor::<f32>(vtx2rhs, num_vtx, num_vdim, device).unwrap();
    crate::check_2d_tensor::<f32>(wtx2co, num_wtx, num_dim, device).unwrap();
    crate::check_2d_tensor::<f32>(wtx2lhs, num_wtx, num_vdim, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            let vtx2co = unsafe { crate::slice_from_tensor::<f32>(vtx2co) }.unwrap();
            let vtx2rhs = unsafe { crate::slice_from_tensor::<f32>(vtx2rhs) }.unwrap();
            let wtx2co = unsafe { crate::slice_from_tensor::<f32>(wtx2co) }.unwrap();
            let wtx2lhs = unsafe { crate::slice_from_tensor_mut::<f32>(wtx2lhs) }.unwrap();
            del_msh_cpu::nbody::elastic3(wtx2co, wtx2lhs, nu, epsilon, vtx2co, vtx2rhs);
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            use del_cudarc_sys::{cu, cuda_check};
            cuda_check!(cu::cuInit(0)).unwrap();
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            let (fnc, _mdl) =
                del_cudarc_sys::load_function_in_module(del_msh_cuda_kernel::NBODY, "elastic")
                    .unwrap();
            let mut builder = del_cudarc_sys::Builder::new(stream);
            builder.arg_u32(num_wtx as u32);
            builder.arg_data(&wtx2co.data);
            builder.arg_data(&wtx2lhs.data);
            builder.arg_u32(num_vtx as u32);
            builder.arg_data(&vtx2co.data);
            builder.arg_data(&vtx2rhs.data);
            builder.arg_f32(nu);
            builder.arg_f32(epsilon);
            builder
                .launch_kernel(
                    fnc,
                    del_cudarc_sys::LaunchConfig::for_num_elems(num_wtx as u32),
                )
                .unwrap();
        }
        _ => {
            let _a = nu;
            let _b = epsilon;
            todo!();
        }
    }
    Ok(())
}
