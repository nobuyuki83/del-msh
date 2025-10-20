use pyo3::{pyfunction, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<pyo3::types::PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(
        quad_oct_tree_binary_radix_tree_and_depth,
        m
    )?)?;
    // m.add_function(pyo3::wrap_pyfunction!(mortons_make_bvh, m)?)?;
    Ok(())
}

// binary_radix_tree_and_depth(&idx2morton, NDIM, max_depth, &mut bnodes, &mut bnode2depth);
#[pyfunction]
fn quad_oct_tree_binary_radix_tree_and_depth(
    _py: Python<'_>,
    idx2morton: &Bound<'_, PyAny>,
    num_dim: usize,
    max_depth: usize,
    bnodes: &Bound<'_, PyAny>,
    bnode2depth: &Bound<'_, PyAny>,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let idx2morton = crate::get_managed_tensor_from_pyany(idx2morton)?;
    let bnodes = crate::get_managed_tensor_from_pyany(bnodes)?;
    let bnode2depth = crate::get_managed_tensor_from_pyany(bnode2depth)?;
    let num_idx = crate::get_shape_tensor(idx2morton, 0);
    let device = idx2morton.ctx.device_type;
    crate::check_1d_tensor::<u32>(idx2morton, num_idx, device);
    crate::check_2d_tensor::<u32>(bnodes, num_idx - 1, 3, device);
    crate::check_1d_tensor::<u32>(bnode2depth, num_idx - 1, device);
    match device {
        dlpack::device_type_codes::CPU => {
            let idx2morton = unsafe { crate::slice_from_tensor::<u32>(idx2morton) }.unwrap();
            let bnodes = unsafe { crate::slice_from_tensor_mut::<u32>(bnodes) }.unwrap();
            let bnode2depth = unsafe { crate::slice_from_tensor_mut::<u32>(bnode2depth) }.unwrap();
            del_msh_cpu::quad_oct_tree::binary_radix_tree_and_depth(
                idx2morton,
                num_dim,
                max_depth,
                bnodes,
                bnode2depth,
            );
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            use del_cudarc_sys::{cu, cuda_check};
            cuda_check!(cu::cuInit(0));
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            //
            let (function, _module) = del_cudarc_sys::load_function_in_module(
                del_msh_cuda_kernel::QUAD_OCT_TREE,
                "binary_radix_tree_and_depth",
            );
            let num_bnode = num_idx as u32 - 1;
            let cfg = del_cudarc_sys::LaunchConfig::for_num_elems(num_bnode);
            let mut builder = del_cudarc_sys::Builder::new(stream);
            builder.arg_i32(num_bnode as i32);
            builder.arg_data(&idx2morton.data);
            builder.arg_i32(num_dim as i32);
            builder.arg_i32(max_depth as i32);
            builder.arg_data(&bnodes.data);
            builder.arg_data(&bnode2depth.data);
            builder.launch_kernel(function, cfg);
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}
