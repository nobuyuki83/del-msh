use del_dlpack::{
    check_1d_tensor as chk1, check_2d_tensor as chk2, dlpack,
    get_managed_tensor_from_pyany as get_tensor, get_shape_tensor as shape, slice, slice_mut,
};
use pyo3::{pyfunction, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<pyo3::types::PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(mortons_vtx2morton_from_vtx2co, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        mortons_make_bvhnodes_from_sorted_mortons,
        m
    )?)?;
    Ok(())
}

#[pyfunction]
fn mortons_vtx2morton_from_vtx2co(
    _py: Python<'_>,
    vtx2co: &Bound<'_, PyAny>,
    transform_co2unit: &Bound<'_, PyAny>,
    vtx2morton: &Bound<'_, PyAny>,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let vtx2co = get_tensor(vtx2co)?;
    let transform_co2unit = get_tensor(transform_co2unit)?;
    let vtx2morton = get_tensor(vtx2morton)?;
    //
    let num_vtx = shape(vtx2co, 0).unwrap();
    let num_dim = shape(vtx2co, 1).unwrap();
    assert!(num_dim == 2 || num_dim == 3);
    let device = vtx2co.ctx.device_type;
    chk2::<f32>(vtx2co, num_vtx, num_dim, device).unwrap();
    chk2::<f32>(transform_co2unit, num_dim + 1, num_dim + 1, device).unwrap();
    chk1::<u32>(vtx2morton, num_vtx, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            del_msh_cpu::mortons::vtx2morton_from_vtx2co(
                num_dim as usize,
                slice!(vtx2co, f32).unwrap(),
                slice!(transform_co2unit, f32).unwrap(),
                slice_mut!(vtx2morton, u32).unwrap(),
            );
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            use del_cudarc_sys::{cu, cuda_check};
            cuda_check!(cu::cuInit(0)).unwrap();
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            let func = del_cudarc_sys::cache_func::get_function_cached(
                "del_msh::mortons",
                del_msh_cuda_kernels::get("mortons").unwrap(),
                "vtx2morton",
            )
            .unwrap();
            let mut builder = del_cudarc_sys::Builder::new(stream);
            builder.arg_u32(num_vtx as u32);
            builder.arg_data(&vtx2co.data);
            builder.arg_u32(num_dim as u32);
            builder.arg_data(&transform_co2unit.data);
            builder.arg_data(&vtx2morton.data);
            builder
                .launch_kernel(
                    func,
                    del_cudarc_sys::LaunchConfig::for_num_elems(num_vtx as u32),
                )
                .unwrap();
        }
        _ => {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err("GPU not supported (compile with --features cuda)"))
        }
    }
    Ok(())
}

#[pyfunction]
fn mortons_make_bvhnodes_from_sorted_mortons(
    _py: Python<'_>,
    idx2obj: &Bound<'_, PyAny>,
    idx2morton: &Bound<'_, PyAny>,
    bhvnodes: &Bound<'_, PyAny>,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let idx2obj = get_tensor(idx2obj)?;
    let idx2morton = get_tensor(idx2morton)?;
    let bvhnodes = get_tensor(bhvnodes)?;
    //
    let n = shape(idx2obj, 0).unwrap();
    let device = idx2obj.ctx.device_type;
    //
    chk1::<u32>(idx2obj, n, device).unwrap();
    chk1::<u32>(idx2morton, n, device).unwrap();
    chk2::<u32>(bvhnodes, 2 * n - 1, 3, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            del_msh_cpu::bvhnodes_morton::update_bvhnodes(
                slice_mut!(bvhnodes, u32).unwrap(),
                slice!(idx2obj, u32).unwrap(),
                slice!(idx2morton, u32).unwrap(),
            );
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            use del_cudarc_sys::{cu, cuda_check};
            cuda_check!(cu::cuInit(0)).unwrap();
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            let func = del_cudarc_sys::cache_func::get_function_cached(
                "del_msh::bvhnodes_morton",
                del_msh_cuda_kernels::get("bvhnodes_morton").unwrap(),
                "kernel_MortonCode_BVHTopology",
            )
            .unwrap();
            {
                let mut builder = del_cudarc_sys::Builder::new(stream);
                builder.arg_u32(n as u32);
                builder.arg_dptr(bvhnodes.data as cu::CUdeviceptr);
                builder.arg_dptr(idx2morton.data as cu::CUdeviceptr);
                builder.arg_dptr(idx2obj.data as cu::CUdeviceptr);
                builder
                    .launch_kernel(func, del_cudarc_sys::LaunchConfig::for_num_elems(n as u32))
                    .unwrap();
            }
        }
        _ => {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err("GPU not supported (compile with --features cuda)"))
        }
    }
    Ok(())
}
