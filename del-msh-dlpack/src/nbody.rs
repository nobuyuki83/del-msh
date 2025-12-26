use del_dlpack::dlpack;
use pyo3::{pyfunction, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<pyo3::types::PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(nbody_screened_poisson, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(nbody_elastic, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        nbody_screened_poisson_with_acceleration,
        m
    )?)?;
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
    let vtx2co = del_dlpack::get_managed_tensor_from_pyany(vtx2co)?;
    let vtx2rhs = del_dlpack::get_managed_tensor_from_pyany(vtx2rhs)?;
    let wtx2co = del_dlpack::get_managed_tensor_from_pyany(wtx2co)?;
    let wtx2lhs = del_dlpack::get_managed_tensor_from_pyany(wtx2lhs)?;
    let num_vtx = del_dlpack::get_shape_tensor(vtx2co, 0).unwrap();
    let num_wtx = del_dlpack::get_shape_tensor(wtx2co, 0).unwrap();
    let num_dim = del_dlpack::get_shape_tensor(vtx2co, 1).unwrap();
    let num_vdim = del_dlpack::get_shape_tensor(vtx2rhs, 1).unwrap();
    let device = vtx2co.ctx.device_type;
    //
    del_dlpack::check_2d_tensor::<f32>(vtx2co, num_vtx, num_dim, device).unwrap();
    del_dlpack::check_2d_tensor::<f32>(vtx2rhs, num_vtx, num_vdim, device).unwrap();
    del_dlpack::check_2d_tensor::<f32>(wtx2co, num_wtx, num_dim, device).unwrap();
    del_dlpack::check_2d_tensor::<f32>(wtx2lhs, num_wtx, num_vdim, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            let vtx2co = unsafe { del_dlpack::slice_from_tensor::<f32>(vtx2co) }.unwrap();
            let vtx2rhs = unsafe { del_dlpack::slice_from_tensor::<f32>(vtx2rhs) }.unwrap();
            let wtx2co = unsafe { del_dlpack::slice_from_tensor::<f32>(wtx2co) }.unwrap();
            let wtx2lhs = unsafe { del_dlpack::slice_from_tensor_mut::<f32>(wtx2lhs) }.unwrap();
            let spoisson = del_msh_cpu::nbody::ScreenedPoison::new(lambda, epsilon);
            del_msh_cpu::nbody::screened_poisson3(&spoisson, wtx2co, wtx2lhs, vtx2co, vtx2rhs);
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            use del_cudarc_sys::{cu, cuda_check};
            cuda_check!(cu::cuInit(0)).unwrap();
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            let fnc = del_cudarc_sys::cache_func::get_function_cached(
                "del_msh::nbody",
                del_msh_cuda_kernels::get("nbody").unwrap(),
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
    let vtx2co = del_dlpack::get_managed_tensor_from_pyany(vtx2co)?;
    let vtx2rhs = del_dlpack::get_managed_tensor_from_pyany(vtx2rhs)?;
    let wtx2co = del_dlpack::get_managed_tensor_from_pyany(wtx2co)?;
    let wtx2lhs = del_dlpack::get_managed_tensor_from_pyany(wtx2lhs)?;
    let num_vtx = del_dlpack::get_shape_tensor(vtx2co, 0).unwrap();
    let num_wtx = del_dlpack::get_shape_tensor(wtx2co, 0).unwrap();
    let num_dim = del_dlpack::get_shape_tensor(vtx2co, 1).unwrap();
    let num_vdim = del_dlpack::get_shape_tensor(vtx2rhs, 1).unwrap();
    let device = vtx2co.ctx.device_type;
    //
    del_dlpack::check_2d_tensor::<f32>(vtx2co, num_vtx, num_dim, device).unwrap();
    del_dlpack::check_2d_tensor::<f32>(vtx2rhs, num_vtx, num_vdim, device).unwrap();
    del_dlpack::check_2d_tensor::<f32>(wtx2co, num_wtx, num_dim, device).unwrap();
    del_dlpack::check_2d_tensor::<f32>(wtx2lhs, num_wtx, num_vdim, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            let vtx2co = unsafe { del_dlpack::slice_from_tensor::<f32>(vtx2co) }.unwrap();
            let vtx2rhs = unsafe { del_dlpack::slice_from_tensor::<f32>(vtx2rhs) }.unwrap();
            let wtx2co = unsafe { del_dlpack::slice_from_tensor::<f32>(wtx2co) }.unwrap();
            let wtx2lhs = unsafe { del_dlpack::slice_from_tensor_mut::<f32>(wtx2lhs) }.unwrap();
            del_msh_cpu::nbody::elastic3(wtx2co, wtx2lhs, nu, epsilon, vtx2co, vtx2rhs);
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            use del_cudarc_sys::{cu, cuda_check};
            cuda_check!(cu::cuInit(0)).unwrap();
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            let fnc = del_cudarc_sys::cache_func::get_function_cached(
                "del_msh::nbody",
                del_msh_cuda_kernels::get("nbody").unwrap(),
                "elastic",
            )
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

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn nbody_screened_poisson_with_acceleration(
    _py: Python<'_>,
    vtx2co: &Bound<'_, PyAny>,
    vtx2rhs: &Bound<'_, PyAny>,
    wtx2co: &Bound<'_, PyAny>,
    wtx2lhs: &Bound<'_, PyAny>,
    lambda: f32,
    epsilon: f32,
    transform_world2unit: &Bound<'_, PyAny>,
    idx2jdx_offset: &Bound<'_, PyAny>,
    jdx2vtx: &Bound<'_, PyAny>,
    onodes: &Bound<'_, PyAny>,
    onode2center: &Bound<'_, PyAny>,
    onode2depth: &Bound<'_, PyAny>,
    onode2gcunit: &Bound<'_, PyAny>,
    onode2rhs: &Bound<'_, PyAny>,
    theta: f32,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let vtx2co = del_dlpack::get_managed_tensor_from_pyany(vtx2co)?;
    let vtx2rhs = del_dlpack::get_managed_tensor_from_pyany(vtx2rhs)?;
    let wtx2co = del_dlpack::get_managed_tensor_from_pyany(wtx2co)?;
    let wtx2lhs = del_dlpack::get_managed_tensor_from_pyany(wtx2lhs)?;
    let transform_world2unit = del_dlpack::get_managed_tensor_from_pyany(transform_world2unit)?;
    let idx2jdx_offset = del_dlpack::get_managed_tensor_from_pyany(idx2jdx_offset)?;
    let jdx2vtx = del_dlpack::get_managed_tensor_from_pyany(jdx2vtx)?;
    let onodes = del_dlpack::get_managed_tensor_from_pyany(onodes)?;
    let onode2center = del_dlpack::get_managed_tensor_from_pyany(onode2center)?;
    let onode2depth = del_dlpack::get_managed_tensor_from_pyany(onode2depth)?;
    let onode2gcunit = del_dlpack::get_managed_tensor_from_pyany(onode2gcunit)?;
    let onode2rhs = del_dlpack::get_managed_tensor_from_pyany(onode2rhs)?;
    //
    let num_vtx = del_dlpack::get_shape_tensor(vtx2co, 0).unwrap();
    let num_wtx = del_dlpack::get_shape_tensor(wtx2co, 0).unwrap();
    let num_dim = del_dlpack::get_shape_tensor(vtx2co, 1).unwrap();
    let num_vdim = del_dlpack::get_shape_tensor(vtx2rhs, 1).unwrap();
    let num_onode = del_dlpack::get_shape_tensor(onodes, 0).unwrap();
    let device = vtx2co.ctx.device_type;
    //
    del_dlpack::check_2d_tensor::<f32>(vtx2co, num_vtx, num_dim, device).unwrap();
    del_dlpack::check_2d_tensor::<f32>(vtx2rhs, num_vtx, num_vdim, device).unwrap();
    del_dlpack::check_2d_tensor::<f32>(wtx2co, num_wtx, num_dim, device).unwrap();
    del_dlpack::check_2d_tensor::<f32>(wtx2lhs, num_wtx, num_vdim, device).unwrap();
    del_dlpack::check_1d_tensor::<f32>(transform_world2unit, 16, device).unwrap();
    del_dlpack::check_2d_tensor::<u32>(onodes, num_onode, 9, device).unwrap();
    del_dlpack::check_2d_tensor::<f32>(onode2center, num_onode, 3, device).unwrap();
    del_dlpack::check_1d_tensor::<u32>(onode2depth, num_onode, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            let vtx2co = unsafe { del_dlpack::slice_from_tensor::<f32>(vtx2co) }.unwrap();
            let vtx2rhs = unsafe { del_dlpack::slice_from_tensor::<f32>(vtx2rhs) }.unwrap();
            let wtx2co = unsafe { del_dlpack::slice_from_tensor::<f32>(wtx2co) }.unwrap();
            let wtx2lhs = unsafe { del_dlpack::slice_from_tensor_mut::<f32>(wtx2lhs) }.unwrap();
            let transform_world2unit =
                unsafe { del_dlpack::slice_from_tensor::<f32>(transform_world2unit) }.unwrap();
            let transform_world2unit = arrayref::array_ref![transform_world2unit, 0, 16];
            let idx2jdx_offset = unsafe { del_dlpack::slice_from_tensor(idx2jdx_offset) }.unwrap();
            let jdx2vtx = unsafe { del_dlpack::slice_from_tensor(jdx2vtx) }.unwrap();
            let onodes = unsafe { del_dlpack::slice_from_tensor::<u32>(onodes) }.unwrap();
            let onode2center =
                unsafe { del_dlpack::slice_from_tensor::<f32>(onode2center) }.unwrap();
            let onode2depth = unsafe { del_dlpack::slice_from_tensor::<u32>(onode2depth) }.unwrap();
            let onode2gcunit =
                unsafe { del_dlpack::slice_from_tensor::<f32>(onode2gcunit) }.unwrap();
            let onode2rhs = unsafe { del_dlpack::slice_from_tensor::<f32>(onode2rhs) }.unwrap();
            let spoisson = del_msh_cpu::nbody::ScreenedPoison::new(lambda, epsilon);
            use slice_of_array::SliceNestExt;
            del_msh_cpu::nbody::barnes_hut(
                &spoisson,
                vtx2co,
                vtx2rhs,
                wtx2co,
                wtx2lhs,
                transform_world2unit,
                del_msh_cpu::nbody::Octree {
                    onodes,
                    onode2center,
                    onode2depth,
                },
                idx2jdx_offset,
                jdx2vtx,
                onode2gcunit.nest(),
                onode2rhs,
                theta,
            );
            //    &spoisson, wtx2co, wtx2lhs, vtx2co, vtx2rhs);
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}
