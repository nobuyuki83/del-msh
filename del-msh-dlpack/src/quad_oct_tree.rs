use pyo3::{pyfunction, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<pyo3::types::PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(
        quad_oct_tree_bnodes_and_bnode2depth_and_bnode2onode,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        quad_oct_tree_make_tree_from_binary_radix_tree,
        m
    )?)?;
    Ok(())
}

// binary_radix_tree_and_depth(&idx2morton, NDIM, max_depth, &mut bnodes, &mut bnode2depth);
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn quad_oct_tree_bnodes_and_bnode2depth_and_bnode2onode(
    _py: Python<'_>,
    idx2morton: &Bound<'_, PyAny>,
    num_dim: usize,
    max_depth: usize,
    bnodes: &Bound<'_, PyAny>,
    bnode2depth: &Bound<'_, PyAny>,
    bnode2onode: &Bound<'_, PyAny>,
    idx2bnode: &Bound<'_, PyAny>,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let idx2morton = crate::get_managed_tensor_from_pyany(idx2morton)?;
    let bnodes = crate::get_managed_tensor_from_pyany(bnodes)?;
    let bnode2depth = crate::get_managed_tensor_from_pyany(bnode2depth)?;
    let bnode2onode = crate::get_managed_tensor_from_pyany(bnode2onode)?;
    let idx2bnode = crate::get_managed_tensor_from_pyany(idx2bnode)?;
    //
    let num_idx = crate::get_shape_tensor(idx2morton, 0);
    let num_bnode = num_idx - 1;
    let device = idx2morton.ctx.device_type;
    //
    crate::check_1d_tensor::<u32>(idx2morton, num_idx, device);
    crate::check_2d_tensor::<u32>(bnodes, num_bnode, 3, device);
    crate::check_1d_tensor::<u32>(bnode2depth, num_bnode, device);
    crate::check_1d_tensor::<u32>(bnode2onode, num_bnode, device);
    crate::check_1d_tensor::<u32>(idx2bnode, num_idx, device);
    //
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
            let bnode2onode = unsafe { crate::slice_from_tensor_mut::<u32>(bnode2onode) }.unwrap();
            let idx2bnode = unsafe { crate::slice_from_tensor_mut::<u32>(idx2bnode) }.unwrap();
            del_msh_cpu::quad_oct_tree::bnode2onode_and_idx2bnode(
                bnodes,
                bnode2depth,
                bnode2onode,
                idx2bnode,
            );
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            use del_cudarc_sys::{cu, cuda_check};
            cuda_check!(cu::cuInit(0));
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            //
            {
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
            let bnode2isonode = del_cudarc_sys::CuVec::<u32>::with_capacity(num_bnode as usize);
            {
                let (function, _module) = del_cudarc_sys::load_function_in_module(
                    del_msh_cuda_kernel::QUAD_OCT_TREE,
                    "bnode2isonode_and_idx2bnode",
                );
                let cfg = del_cudarc_sys::LaunchConfig::for_num_elems(num_bnode as u32);
                let mut builder = del_cudarc_sys::Builder::new(stream);
                builder.arg_i32(num_bnode as i32);
                builder.arg_data(&bnodes.data);
                builder.arg_data(&bnode2depth.data);
                builder.arg_dptr(bnode2isonode.dptr);
                builder.arg_data(&idx2bnode.data);
                builder.launch_kernel(function, cfg);
            }
            let bnode2onode = del_cudarc_sys::CuVec::<u32> {
                n: num_bnode as usize,
                dptr: bnode2onode.data as cu::CUdeviceptr,
                is_free_at_drop: false,
                phantom: std::marker::PhantomData,
            };
            del_cudarc_sys::cumsum::exclusive_scan(stream, &bnode2isonode, &bnode2onode);
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}

#[pyfunction]
pub fn quad_oct_tree_make_tree_from_binary_radix_tree(
    _py: Python<'_>,
    bnodes: &Bound<'_, PyAny>,
    bnode2onode: &Bound<'_, PyAny>,
    bnode2depth: &Bound<'_, PyAny>,
    idx2bnode: &Bound<'_, PyAny>,
    idx2morton: &Bound<'_, PyAny>,
    num_dim: usize,
    max_depth: usize,
    onodes: &Bound<'_, PyAny>,
    onode2depth: &Bound<'_, PyAny>,
    onode2center: &Bound<'_, PyAny>,
    idx2onode: &Bound<'_, PyAny>,
    idx2center: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let bnodes = crate::get_managed_tensor_from_pyany(bnodes)?;
    let bnode2onode = crate::get_managed_tensor_from_pyany(bnode2onode)?;
    let bnode2depth = crate::get_managed_tensor_from_pyany(bnode2depth)?;
    let idx2bnode = crate::get_managed_tensor_from_pyany(idx2bnode)?;
    let idx2morton = crate::get_managed_tensor_from_pyany(idx2morton)?;
    //
    let onodes = crate::get_managed_tensor_from_pyany(onodes)?;
    let onode2depth = crate::get_managed_tensor_from_pyany(onode2depth)?;
    let onode2center = crate::get_managed_tensor_from_pyany(onode2center)?;
    let idx2onode = crate::get_managed_tensor_from_pyany(idx2onode)?;
    let idx2center = crate::get_managed_tensor_from_pyany(idx2center)?;
    //
    let num_bnode = crate::get_shape_tensor(bnodes, 0);
    let num_idx = num_bnode + 1;
    let device = bnodes.ctx.device_type;
    //
    crate::check_2d_tensor::<u32>(bnodes, num_bnode, 3, device);
    crate::check_1d_tensor::<u32>(bnode2onode, num_bnode, device);
    crate::check_1d_tensor::<u32>(bnode2depth, num_bnode, device);
    let num_onode = crate::get_shape_tensor(onodes, 0);
    let num_link = crate::get_shape_tensor(onodes, 1);
    assert!(num_dim == 2 || num_dim == 3);
    assert_eq!(num_link as usize, 1 + 2i32.pow(num_dim as u32) as usize);
    crate::check_2d_tensor::<u32>(onodes, num_onode, num_link, device);
    crate::check_1d_tensor::<u32>(onode2depth, num_onode, device);
    crate::check_2d_tensor::<f32>(onode2center, num_onode, 3, device);
    crate::check_1d_tensor::<u32>(idx2onode, num_idx, device);
    crate::check_2d_tensor::<f32>(idx2center, num_idx, 3, device);
    //
    match device {
        dlpack::device_type_codes::CPU => {
            let bnodes = unsafe { crate::slice_from_tensor(bnodes) }.unwrap();
            let bnode2onode = unsafe { crate::slice_from_tensor(bnode2onode) }.unwrap();
            let bnode2depth = unsafe { crate::slice_from_tensor(bnode2depth) }.unwrap();
            let idx2bnode = unsafe { crate::slice_from_tensor(idx2bnode) }.unwrap();
            let idx2morton = unsafe { crate::slice_from_tensor(idx2morton) }.unwrap();
            let onodes = unsafe { crate::slice_from_tensor_mut(onodes) }.unwrap();
            let onode2depth = unsafe { crate::slice_from_tensor_mut(onode2depth) }.unwrap();
            let onode2center = unsafe { crate::slice_from_tensor_mut(onode2center) }.unwrap();
            let idx2onode = unsafe { crate::slice_from_tensor_mut(idx2onode) }.unwrap();
            let idx2center = unsafe { crate::slice_from_tensor_mut(idx2center) }.unwrap();
            if num_dim == 2 {
                del_msh_cpu::quad_oct_tree::make_tree_from_binary_radix_tree::<2>(
                    bnodes,
                    bnode2onode,
                    bnode2depth,
                    idx2bnode,
                    idx2morton,
                    num_onode as usize,
                    max_depth,
                    onodes,
                    onode2depth,
                    onode2center,
                    idx2onode,
                    idx2center,
                );
            } else if num_dim == 3 {
                del_msh_cpu::quad_oct_tree::make_tree_from_binary_radix_tree::<3>(
                    bnodes,
                    bnode2onode,
                    bnode2depth,
                    idx2bnode,
                    idx2morton,
                    num_onode as usize,
                    max_depth,
                    onodes,
                    onode2depth,
                    onode2center,
                    idx2onode,
                    idx2center,
                );
            } else {
                unreachable!();
            }
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            todo!()
        }
        _ => {}
    }

    Ok(())
}
