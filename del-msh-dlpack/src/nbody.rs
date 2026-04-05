#[cfg(feature = "cuda")]
use del_cudarc_sys::{cu::CUdeviceptr, CuVec};
use del_dlpack::{
    dlpack, get_managed_tensor_from_pyany as get_tensor, get_shape_tensor as shape,
    check_1d_tensor as chk1, check_2d_tensor as chk2, slice, slice_mut,
};
use pyo3::{pyfunction, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<pyo3::types::PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(nbody_screened_poisson, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(nbody_elastic, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        nbody_screened_poisson_with_acceleration,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(nbody_elastic_with_acceleration, m)?)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn filter_brute_force(
    model: del_msh_cpu::nbody::NBodyModel,
    _py: Python<'_>,
    vtx2co: &Bound<'_, PyAny>,
    vtx2rhs: &Bound<'_, PyAny>,
    wtx2co: &Bound<'_, PyAny>,
    wtx2lhs: &Bound<'_, PyAny>,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let vtx2co = get_tensor(vtx2co)?;
    let vtx2rhs = get_tensor(vtx2rhs)?;
    let wtx2co = get_tensor(wtx2co)?;
    let wtx2lhs = get_tensor(wtx2lhs)?;
    //
    let num_vtx = shape(vtx2co, 0).unwrap();
    let num_wtx = shape(wtx2co, 0).unwrap();
    let num_dim = shape(vtx2co, 1).unwrap();
    let num_vdim = shape(vtx2rhs, 1).unwrap();
    let device = vtx2co.ctx.device_type;
    //
    chk2::<f32>(vtx2co, num_vtx, num_dim, device).unwrap();
    chk2::<f32>(vtx2rhs, num_vtx, num_vdim, device).unwrap();
    chk2::<f32>(wtx2co, num_wtx, num_dim, device).unwrap();
    chk2::<f32>(wtx2lhs, num_wtx, num_vdim, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            del_msh_cpu::nbody::filter_brute_force(
                &model,
                slice!(wtx2co, f32).unwrap(),
                slice_mut!(wtx2lhs, f32).unwrap(),
                slice!(vtx2co, f32).unwrap(),
                slice!(vtx2rhs, f32).unwrap(),
            );
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            use del_cudarc_sys::{cu, cuda_check};
            cuda_check!(cu::cuInit(0)).unwrap();
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            match model {
                del_msh_cpu::nbody::NBodyModel::ScreenedPoison {
                    eps: epsilon,
                    lambda,
                    ..
                } => {
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
                del_msh_cpu::nbody::NBodyModel::Elastic {
                    eps: epsilon, nu, ..
                } => {
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
            }
        }
        _ => {
            todo!()
        }
    }
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
    let spoisson = del_msh_cpu::nbody::NBodyModel::screened_poisson(lambda, epsilon);
    filter_brute_force(spoisson, _py, vtx2co, vtx2rhs, wtx2co, wtx2lhs, stream_ptr)
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
    let elastic_model = del_msh_cpu::nbody::NBodyModel::elastic(nu, epsilon);
    filter_brute_force(
        elastic_model,
        _py,
        vtx2co,
        vtx2rhs,
        wtx2co,
        wtx2lhs,
        stream_ptr,
    )
}

// ----------------------------------

#[allow(clippy::too_many_arguments)]
fn filter_with_acceleration(
    _py: Python<'_>,
    model: del_msh_cpu::nbody::NBodyModel,
    vtx2co: &Bound<'_, PyAny>,
    vtx2rhs: &Bound<'_, PyAny>,
    wtx2co: &Bound<'_, PyAny>,
    wtx2lhs: &Bound<'_, PyAny>,
    transform_world2unit: &Bound<'_, PyAny>,
    idx2jdx_offset: &Bound<'_, PyAny>,
    jdx2vtx: &Bound<'_, PyAny>,
    onode2idx_tree: &Bound<'_, PyAny>,
    onode2center: &Bound<'_, PyAny>,
    onode2depth: &Bound<'_, PyAny>,
    onode2gcunit: &Bound<'_, PyAny>,
    onode2rhs: &Bound<'_, PyAny>,
    theta: f32,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let vtx2co = get_tensor(vtx2co)?;
    let vtx2rhs = get_tensor(vtx2rhs)?;
    let wtx2co = get_tensor(wtx2co)?;
    let wtx2lhs = get_tensor(wtx2lhs)?;
    let transform_world2unit = get_tensor(transform_world2unit)?;
    let idx2jdx_offset = get_tensor(idx2jdx_offset)?;
    let jdx2vtx = get_tensor(jdx2vtx)?;
    let onode2idx_tree = get_tensor(onode2idx_tree)?;
    let onode2center = get_tensor(onode2center)?;
    let onode2depth = get_tensor(onode2depth)?;
    let onode2gcunit = get_tensor(onode2gcunit)?;
    let onode2rhs = get_tensor(onode2rhs)?;
    //
    let num_vtx = shape(vtx2co, 0).unwrap();
    let num_wtx = shape(wtx2co, 0).unwrap();
    let num_dim = shape(vtx2co, 1).unwrap();
    let num_vdim = shape(vtx2rhs, 1).unwrap();
    let num_onode = shape(onode2idx_tree, 0).unwrap();
    let num_idx = shape(idx2jdx_offset, 0).unwrap() - 1;
    let device = vtx2co.ctx.device_type;
    //
    chk2::<f32>(vtx2co, num_vtx, num_dim, device).unwrap();
    chk2::<f32>(vtx2rhs, num_vtx, num_vdim, device).unwrap();
    chk2::<f32>(wtx2co, num_wtx, num_dim, device).unwrap();
    chk2::<f32>(wtx2lhs, num_wtx, num_vdim, device).unwrap();
    chk1::<f32>(transform_world2unit, 16, device).unwrap();
    chk1::<u32>(idx2jdx_offset, num_idx + 1, device).unwrap();
    chk1::<u32>(jdx2vtx, num_vtx, device).unwrap();
    chk2::<u32>(onode2idx_tree, num_onode, 9, device).unwrap();
    chk2::<f32>(onode2center, num_onode, 3, device).unwrap();
    chk1::<u32>(onode2depth, num_onode, device).unwrap();
    chk2::<f32>(onode2gcunit, num_onode, 3, device).unwrap();
    chk2::<f32>(onode2rhs, num_onode, 3, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            use slice_of_array::SliceNestExt;
            del_msh_cpu::nbody::barnes_hut(
                &model,
                slice!(vtx2co, f32).unwrap(),
                slice!(vtx2rhs, f32).unwrap(),
                slice!(wtx2co, f32).unwrap(),
                slice_mut!(wtx2lhs, f32).unwrap(),
                arrayref::array_ref![slice!(transform_world2unit, f32).unwrap(), 0, 16],
                del_msh_cpu::nbody::Octree {
                    onodes: slice!(onode2idx_tree, u32).unwrap(),
                    onode2center: slice!(onode2center, f32).unwrap(),
                    onode2depth: slice!(onode2depth, u32).unwrap(),
                },
                slice!(idx2jdx_offset, u32).unwrap(),
                slice!(jdx2vtx, u32).unwrap(),
                slice!(onode2gcunit, f32).unwrap().nest(),
                slice!(onode2rhs, f32).unwrap(),
                theta,
            );
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            use del_cudarc_sys::{cu, cuda_check};
            cuda_check!(cu::cuInit(0)).unwrap();
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            let transform_unit2world = {
                let a = CuVec::<f32>::new(transform_world2unit.data as CUdeviceptr, 16, false);
                let a = a.copy_to_host().unwrap();
                let a = arrayref::array_ref![&a, 0, 16];
                let b = del_geo_core::mat4_col_major::try_inverse_with_pivot(a).unwrap();
                CuVec::from_slice(b.as_slice()).unwrap()
            };
            match model {
                del_msh_cpu::nbody::NBodyModel::ScreenedPoison {
                    lambda,
                    eps: epsilon,
                    ..
                } => {
                    let func = del_cudarc_sys::cache_func::get_function_cached(
                        "del_msh::nbody",
                        del_msh_cuda_kernels::get("nbody").unwrap(),
                        "screened_poisson3_with_acceleration",
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
                    builder.arg_data(&transform_world2unit.data);
                    builder.arg_dptr(transform_unit2world.dptr);
                    builder.arg_u32(num_onode as u32);
                    builder.arg_data(&onode2idx_tree.data);
                    builder.arg_data(&onode2center.data);
                    builder.arg_data(&onode2depth.data);
                    builder.arg_data(&idx2jdx_offset.data);
                    builder.arg_data(&jdx2vtx.data);
                    builder.arg_data(&onode2gcunit.data);
                    builder.arg_data(&onode2rhs.data);
                    builder.arg_f32(theta);
                    builder
                        .launch_kernel(
                            func,
                            del_cudarc_sys::LaunchConfig::for_num_elems(num_wtx as u32),
                        )
                        .unwrap();
                }
                del_msh_cpu::nbody::NBodyModel::Elastic {
                    nu, eps: epsilon, ..
                } => {
                    //let cfg = del_cudarc_sys::LaunchConfig::for_num_elems(num_wtx as u32);
                    let cfg = {
                        const NUM_THREADS: u32 = 128;
                        let num_blocks = (num_wtx as u32).div_ceil(NUM_THREADS);
                        del_cudarc_sys::LaunchConfig {
                            grid_dim: (num_blocks, 1, 1),
                            block_dim: (NUM_THREADS, 1, 1),
                            shared_mem_bytes: 0,
                        }
                    };
                    let func = del_cudarc_sys::cache_func::get_function_cached(
                        "del_msh::nbody",
                        del_msh_cuda_kernels::get("nbody").unwrap(),
                        "elastic3_with_acceleration",
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
                    builder.arg_data(&transform_world2unit.data);
                    builder.arg_dptr(transform_unit2world.dptr);
                    builder.arg_u32(num_onode as u32);
                    builder.arg_data(&onode2idx_tree.data);
                    builder.arg_data(&onode2center.data);
                    builder.arg_data(&onode2depth.data);
                    builder.arg_data(&idx2jdx_offset.data);
                    builder.arg_data(&jdx2vtx.data);
                    builder.arg_data(&onode2gcunit.data);
                    builder.arg_data(&onode2rhs.data);
                    builder.arg_f32(theta);
                    builder.launch_kernel(func, cfg).unwrap();
                }
            }
        }
        _ => {
            todo!()
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
    onode2idx_tree: &Bound<'_, PyAny>,
    onode2center: &Bound<'_, PyAny>,
    onode2depth: &Bound<'_, PyAny>,
    onode2gcunit: &Bound<'_, PyAny>,
    onode2rhs: &Bound<'_, PyAny>,
    theta: f32,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let model = del_msh_cpu::nbody::NBodyModel::screened_poisson(lambda, epsilon);
    filter_with_acceleration(
        _py,
        model,
        vtx2co,
        vtx2rhs,
        wtx2co,
        wtx2lhs,
        transform_world2unit,
        idx2jdx_offset,
        jdx2vtx,
        onode2idx_tree,
        onode2center,
        onode2depth,
        onode2gcunit,
        onode2rhs,
        theta,
        stream_ptr,
    )
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn nbody_elastic_with_acceleration(
    _py: Python<'_>,
    vtx2co: &Bound<'_, PyAny>,
    vtx2rhs: &Bound<'_, PyAny>,
    wtx2co: &Bound<'_, PyAny>,
    wtx2lhs: &Bound<'_, PyAny>,
    nu: f32,
    epsilon: f32,
    transform_world2unit: &Bound<'_, PyAny>,
    idx2jdx_offset: &Bound<'_, PyAny>,
    jdx2vtx: &Bound<'_, PyAny>,
    onode2idx_tree: &Bound<'_, PyAny>,
    onode2center: &Bound<'_, PyAny>,
    onode2depth: &Bound<'_, PyAny>,
    onode2gcunit: &Bound<'_, PyAny>,
    onode2rhs: &Bound<'_, PyAny>,
    theta: f32,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let model = del_msh_cpu::nbody::NBodyModel::elastic(nu, epsilon);
    filter_with_acceleration(
        _py,
        model,
        vtx2co,
        vtx2rhs,
        wtx2co,
        wtx2lhs,
        transform_world2unit,
        idx2jdx_offset,
        jdx2vtx,
        onode2idx_tree,
        onode2center,
        onode2depth,
        onode2gcunit,
        onode2rhs,
        theta,
        stream_ptr,
    )
}
