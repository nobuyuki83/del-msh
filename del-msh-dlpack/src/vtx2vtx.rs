use pyo3::{types::PyModule, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(
        solve,
        m
    )?)?;
    Ok(())
}

/// Pythonから渡された PyCapsule を Rust 側で読み取る
#[pyo3::pyfunction]
fn solve(
    _py: Python,
    vtx2idx: &Bound<'_, PyAny>,
    idx2vtx: &Bound<'_, PyAny>,
    vtx2xyz: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
    lambda: f32,
    vtx2xyz_tmp: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let vtx2idx = crate::get_managed_tensor_from_pyany(vtx2idx)?;
    let idx2vtx = crate::get_managed_tensor_from_pyany(idx2vtx)?;
    let vtx2xyz = crate::get_managed_tensor_from_pyany(vtx2xyz)?;
    let rhs = crate::get_managed_tensor_from_pyany(rhs)?;
    let vtx2xyz_tmp = crate::get_managed_tensor_from_pyany(vtx2xyz_tmp)?;

    let vtx2idx_shape = unsafe { std::slice::from_raw_parts(vtx2idx.shape, vtx2idx.ndim as usize) };
    let vtx2xyz_shape = unsafe { std::slice::from_raw_parts(vtx2xyz.shape, vtx2xyz.ndim as usize) };
    let num_vtx = vtx2idx_shape[0];
    let num_dim = vtx2xyz_shape[1];
    // dbg!(num_vtx, num_dim);

    // DLPack を unsafe にアンラップ
    match vtx2idx.ctx.device_type {
        dlpack::device_type_codes::CPU => {
            /*
            let data_ptr = tensor.data as *mut u32;
            let data = std::slice::from_raw_parts_mut(data_ptr, total_elements);
            data.iter_mut().enumerate().for_each(|(i, v)| *v = i as u32);
             */
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            println!("GPU_{}", vtx2idx.ctx.device_id);
            let (function, _module) = del_cudarc_sys::load_function_in_module(
                del_msh_cuda_kernel::LAPLACIAN_SMOOTHING_JACOBI,
                "laplacian_smoothing_jacobi",
            );
            let stream = del_cudarc_sys::create_stream_in_current_context();
            for _itr in 0..20 {
                {
                    let mut builder = del_cudarc_sys::Builder::new(stream);
                    builder.arg_i32(num_vtx as i32);
                    builder.arg_data(&vtx2idx.data);
                    builder.arg_data(&idx2vtx.data);
                    builder.arg_f32(lambda);
                    builder.arg_data(&vtx2xyz_tmp.data);
                    builder.arg_data(&vtx2xyz.data);
                    builder.arg_data(&rhs.data);
                    unsafe {
                        builder.launch_kernel(function, num_vtx as u32);
                    }
                }
                {
                    let mut builder = del_cudarc_sys::Builder::new(stream);
                    builder.arg_i32(num_vtx as i32);
                    builder.arg_data(&vtx2idx.data);
                    builder.arg_data(&idx2vtx.data);
                    builder.arg_f32(lambda);
                    builder.arg_data(&vtx2xyz.data);
                    builder.arg_data(&vtx2xyz_tmp.data);
                    builder.arg_data(&rhs.data);
                    unsafe {
                        builder.launch_kernel(function, num_vtx as u32);
                    }
                }
            }
            unsafe {
                del_cudarc_sys::cuStreamSynchronize(stream);
                del_cudarc_sys::cuStreamDestroy_v2(stream);
            }
        }
        _ => println!("Unknown device type {}", vtx2idx.ctx.device_type),
    }
    Ok(())
}