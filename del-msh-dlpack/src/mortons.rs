use crate::get_shape_tensor;
use pyo3::{pyfunction, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<pyo3::types::PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(mortons_vtx2morton_from_vtx2co, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(mortons_make_bvh, m)?)?;
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
    let vtx2co = crate::get_managed_tensor_from_pyany(vtx2co)?;
    let transform_co2unit = crate::get_managed_tensor_from_pyany(transform_co2unit)?;
    let vtx2morton = crate::get_managed_tensor_from_pyany(vtx2morton)?;
    let num_vtx = get_shape_tensor(vtx2co, 0);
    let num_dim = get_shape_tensor(vtx2co, 1);
    assert!(num_dim == 2 || num_dim == 3);
    let device = vtx2co.ctx.device_type;
    crate::check_2d_tensor::<f32>(vtx2co, num_vtx, num_dim, device);
    crate::check_2d_tensor::<f32>(transform_co2unit, num_dim + 1, num_dim + 1, device);
    crate::check_1d_tensor::<u32>(vtx2morton, num_vtx, device);
    //
    match device {
        dlpack::device_type_codes::CPU => {
            let vtx2co = unsafe { crate::slice_from_tensor::<f32>(vtx2co) }.unwrap();
            let transform_co2unit =
                unsafe { crate::slice_from_tensor::<f32>(transform_co2unit) }.unwrap();
            let vtx2morton = unsafe { crate::slice_from_tensor_mut::<u32>(vtx2morton) }.unwrap();
            del_msh_cpu::mortons::vtx2morton_from_vtx2co(
                num_dim as usize,
                vtx2co,
                transform_co2unit,
                vtx2morton,
            );
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            use del_cudarc_sys::cu;
            use del_cudarc_sys::cuda_check;
            cuda_check!(cu::cuInit(0));
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            {
                let (func, _mdl) = del_cudarc_sys::load_function_in_module(
                    del_msh_cuda_kernel::MORTONS,
                    "vtx2morton",
                );
                let mut builder = del_cudarc_sys::Builder::new(stream);
                builder.arg_i32(num_vtx as i32);
                builder.arg_data(&vtx2co.data);
                builder.arg_i32(num_dim as i32);
                builder.arg_data(&transform_co2unit.data);
                builder.arg_data(&vtx2morton.data);
                builder.launch_kernel(
                    func,
                    del_cudarc_sys::LaunchConfig::for_num_elems(num_vtx as u32),
                );
            }
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}

#[pyfunction]
fn mortons_make_bvh(
    _py: Python<'_>,
    idx2obj: &Bound<'_, PyAny>,
    idx2morton: &Bound<'_, PyAny>,
    bhvnodes: &Bound<'_, PyAny>,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let idx2obj = crate::get_managed_tensor_from_pyany(idx2obj)?;
    let idx2morton = crate::get_managed_tensor_from_pyany(idx2morton)?;
    let bvhnodes = crate::get_managed_tensor_from_pyany(bhvnodes)?;
    let n = crate::get_shape_tensor(idx2obj, 0);
    let device = idx2obj.ctx.device_type;
    crate::check_1d_tensor::<u32>(idx2obj, n, device);
    crate::check_1d_tensor::<u32>(idx2morton, n, device);
    crate::check_2d_tensor::<u32>(bvhnodes, 2 * n - 1, 3, device);
    match device {
        dlpack::device_type_codes::CPU => {
            let idx2obj = unsafe { crate::slice_from_tensor::<u32>(idx2obj) }.unwrap();
            let idx2morton = unsafe { crate::slice_from_tensor::<u32>(idx2morton) }.unwrap();
            let bvhnodes = unsafe { crate::slice_from_tensor_mut::<u32>(bvhnodes) }.unwrap();
            del_msh_cpu::bvhnodes_morton::update_bvhnodes(bvhnodes, idx2obj, idx2morton);
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            use del_cudarc_sys::cu;
            use del_cudarc_sys::cuda_check;
            cuda_check!(cu::cuInit(0));
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            let (func, _mdl) = del_cudarc_sys::load_function_in_module(
                del_msh_cuda_kernel::BVHNODES_MORTON,
                "kernel_MortonCode_BVHTopology",
            );
            {
                let mut builder = del_cudarc_sys::Builder::new(stream);
                builder.arg_i32(n as i32);
                builder.arg_dptr(bvhnodes.data as cu::CUdeviceptr);
                builder.arg_dptr(idx2morton.data as cu::CUdeviceptr);
                builder.arg_dptr(idx2obj.data as cu::CUdeviceptr);
                /*
                builder.arg_i32(num_vtx as i32);
                builder.arg_data(&vtx2co.data);
                builder.arg_i32(num_dim as i32);
                builder.arg_data(&transform_co2unit.data);
                builder.arg_data(&vtx2morton.data);
                 */
                builder.launch_kernel(func, del_cudarc_sys::LaunchConfig::for_num_elems(n as u32));
            }
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}
