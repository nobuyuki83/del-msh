use del_dlpack::dlpack;
use pyo3::{pyfunction, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<pyo3::types::PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(
        quad_oct_tree_bnodes_and_bnode2depth_and_bnode2onode_and_idx2bnode,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        quad_oct_tree_make_tree_from_binary_radix_tree,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(quad_oct_tree_aggregate, m)?)?;
    Ok(())
}

// binary_radix_tree_and_depth(&idx2morton, NDIM, max_depth, &mut bnodes, &mut bnode2depth);
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn quad_oct_tree_bnodes_and_bnode2depth_and_bnode2onode_and_idx2bnode(
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
    let idx2morton = del_dlpack::get_managed_tensor_from_pyany(idx2morton)?;
    let bnodes = del_dlpack::get_managed_tensor_from_pyany(bnodes)?;
    let bnode2depth = del_dlpack::get_managed_tensor_from_pyany(bnode2depth)?;
    let bnode2onode = del_dlpack::get_managed_tensor_from_pyany(bnode2onode)?;
    let idx2bnode = del_dlpack::get_managed_tensor_from_pyany(idx2bnode)?;
    //
    let num_idx = del_dlpack::get_shape_tensor(idx2morton, 0).unwrap();
    let num_bnode = num_idx - 1;
    let device = idx2morton.ctx.device_type;
    //
    del_dlpack::check_1d_tensor::<u32>(idx2morton, num_idx, device).unwrap();
    del_dlpack::check_2d_tensor::<u32>(bnodes, num_bnode, 3, device).unwrap();
    del_dlpack::check_1d_tensor::<u32>(bnode2depth, num_bnode, device).unwrap();
    del_dlpack::check_1d_tensor::<u32>(bnode2onode, num_bnode, device).unwrap();
    del_dlpack::check_1d_tensor::<u32>(idx2bnode, num_idx, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            let idx2morton = unsafe { del_dlpack::slice_from_tensor::<u32>(idx2morton) }.unwrap();
            let bnodes = unsafe { del_dlpack::slice_from_tensor_mut::<u32>(bnodes) }.unwrap();
            let bnode2depth =
                unsafe { del_dlpack::slice_from_tensor_mut::<u32>(bnode2depth) }.unwrap();
            del_msh_cpu::quad_oct_tree::binary_radix_tree_and_depth(
                idx2morton,
                num_dim,
                max_depth,
                bnodes,
                bnode2depth,
            );
            let bnode2onode =
                unsafe { del_dlpack::slice_from_tensor_mut::<u32>(bnode2onode) }.unwrap();
            let idx2bnode = unsafe { del_dlpack::slice_from_tensor_mut::<u32>(idx2bnode) }.unwrap();
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
            cuda_check!(cu::cuInit(0)).unwrap();
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            //
            {
                /*
                let (function, _module) = del_cudarc_sys::load_function_in_module(
                    del_msh_cuda_kernel::QUAD_OCT_TREE,
                    "binary_radix_tree_and_depth",
                )
                .unwrap();
                 */
                /*
                let function =
                    crate::load_get_function("quad_oct_tree", "binary_radix_tree_and_depth")
                        .unwrap();
                 */
                let function = del_cudarc_sys::cache_func::get_function_cached(
                    "del_msh::quad_oct_tree",
                    del_msh_cuda_kernels::get("quad_oct_tree").unwrap(),
                    "binary_radix_tree_and_depth",
                )
                .unwrap();
                let num_bnode = num_idx as u32 - 1;
                let cfg = del_cudarc_sys::LaunchConfig::for_num_elems(num_bnode);
                let mut builder = del_cudarc_sys::Builder::new(stream);
                builder.arg_u32(num_bnode);
                builder.arg_data(&idx2morton.data);
                builder.arg_u32(num_dim as u32);
                builder.arg_u32(max_depth as u32);
                builder.arg_data(&bnodes.data);
                builder.arg_data(&bnode2depth.data);
                builder.launch_kernel(function, cfg).unwrap();
            }
            let bnode2isonode =
                del_cudarc_sys::CuVec::<u32>::with_capacity(num_bnode as usize).unwrap();
            bnode2isonode.set_zeros(stream).unwrap();
            {
                /*
                let (function, _module) = del_cudarc_sys::load_function_in_module(
                    del_msh_cuda_kernel::QUAD_OCT_TREE,
                    "bnode2isonode_and_idx2bnode",
                )
                .unwrap();
                 */
                /*
                let function =
                    crate::load_get_function("quad_oct_tree", "bnode2isonode_and_idx2bnode")
                        .unwrap();
                 */
                let function = del_cudarc_sys::cache_func::get_function_cached(
                    "del_msh::quad_oct_tree",
                    del_msh_cuda_kernels::get("quad_oct_tree").unwrap(),
                    "bnode2isonode_and_idx2bnode",
                )
                .unwrap();
                let cfg = del_cudarc_sys::LaunchConfig::for_num_elems(num_bnode as u32);
                let mut builder = del_cudarc_sys::Builder::new(stream);
                builder.arg_u32(num_bnode as u32);
                builder.arg_data(&bnodes.data);
                builder.arg_data(&bnode2depth.data);
                builder.arg_dptr(bnode2isonode.dptr);
                builder.arg_data(&idx2bnode.data);
                builder.launch_kernel(function, cfg).unwrap();
            }
            let bnode2onode = del_cudarc_sys::CuVec::<u32>::from_dptr(
                bnode2onode.data as cu::CUdeviceptr,
                num_bnode as usize,
            );
            del_cudarc_sys::cumsum::exclusive_scan(stream, &bnode2isonode, &bnode2onode);
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
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
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let bnodes = del_dlpack::get_managed_tensor_from_pyany(bnodes)?;
    let bnode2onode = del_dlpack::get_managed_tensor_from_pyany(bnode2onode)?;
    let bnode2depth = del_dlpack::get_managed_tensor_from_pyany(bnode2depth)?;
    let idx2bnode = del_dlpack::get_managed_tensor_from_pyany(idx2bnode)?;
    let idx2morton = del_dlpack::get_managed_tensor_from_pyany(idx2morton)?;
    //
    let onodes = del_dlpack::get_managed_tensor_from_pyany(onodes)?;
    let onode2depth = del_dlpack::get_managed_tensor_from_pyany(onode2depth)?;
    let onode2center = del_dlpack::get_managed_tensor_from_pyany(onode2center)?;
    let idx2onode = del_dlpack::get_managed_tensor_from_pyany(idx2onode)?;
    let idx2center = del_dlpack::get_managed_tensor_from_pyany(idx2center)?;
    //
    let num_bnode = del_dlpack::get_shape_tensor(bnodes, 0).unwrap();
    let num_idx = num_bnode + 1;
    let device = bnodes.ctx.device_type;
    //
    del_dlpack::check_2d_tensor::<u32>(bnodes, num_bnode, 3, device).unwrap();
    del_dlpack::check_1d_tensor::<u32>(bnode2onode, num_bnode, device).unwrap();
    del_dlpack::check_1d_tensor::<u32>(bnode2depth, num_bnode, device).unwrap();
    let num_onode = del_dlpack::get_shape_tensor(onodes, 0).unwrap();
    let num_link = del_dlpack::get_shape_tensor(onodes, 1).unwrap();
    assert!(num_dim == 2 || num_dim == 3);
    assert_eq!(num_link as usize, 1 + 2i32.pow(num_dim as u32) as usize);
    del_dlpack::check_2d_tensor::<u32>(onodes, num_onode, num_link, device).unwrap();
    del_dlpack::check_1d_tensor::<u32>(onode2depth, num_onode, device).unwrap();
    del_dlpack::check_2d_tensor::<f32>(onode2center, num_onode, num_dim as i64, device).unwrap();
    del_dlpack::check_1d_tensor::<u32>(idx2onode, num_idx, device).unwrap();
    del_dlpack::check_2d_tensor::<f32>(idx2center, num_idx, num_dim as i64, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            let bnodes = unsafe { del_dlpack::slice_from_tensor(bnodes) }.unwrap();
            let bnode2onode = unsafe { del_dlpack::slice_from_tensor(bnode2onode) }.unwrap();
            let bnode2depth = unsafe { del_dlpack::slice_from_tensor(bnode2depth) }.unwrap();
            let idx2bnode = unsafe { del_dlpack::slice_from_tensor(idx2bnode) }.unwrap();
            let idx2morton = unsafe { del_dlpack::slice_from_tensor(idx2morton) }.unwrap();
            let onodes = unsafe { del_dlpack::slice_from_tensor_mut(onodes) }.unwrap();
            let onode2depth = unsafe { del_dlpack::slice_from_tensor_mut(onode2depth) }.unwrap();
            let onode2center = unsafe { del_dlpack::slice_from_tensor_mut(onode2center) }.unwrap();
            let idx2onode = unsafe { del_dlpack::slice_from_tensor_mut(idx2onode) }.unwrap();
            let idx2center = unsafe { del_dlpack::slice_from_tensor_mut(idx2center) }.unwrap();
            del_msh_cpu::quad_oct_tree::make_tree_from_binary_radix_tree(
                bnodes,
                bnode2onode,
                bnode2depth,
                idx2bnode,
                idx2morton,
                num_onode as usize,
                max_depth,
                num_dim,
                onodes,
                onode2depth,
                onode2center,
                idx2onode,
                idx2center,
            );
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            use del_cudarc_sys::{cu, cuda_check};
            cuda_check!(cu::cuInit(0)).unwrap();
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            let fnc = del_cudarc_sys::cache_func::get_function_cached(
                "del_msh::quad_oct_tree",
                del_msh_cuda_kernels::get("quad_oct_tree").unwrap(),
                "make_tree_from_binary_radix_tree",
            )
            .unwrap();
            let cfg = del_cudarc_sys::LaunchConfig::for_num_elems(num_idx as u32);
            let mut builder = del_cudarc_sys::Builder::new(stream);
            builder.arg_u32(num_idx as u32);
            builder.arg_data(&bnodes.data);
            builder.arg_data(&bnode2onode.data);
            builder.arg_data(&bnode2depth.data);
            builder.arg_data(&idx2bnode.data);
            builder.arg_data(&idx2morton.data);
            builder.arg_u32(num_onode as u32);
            builder.arg_u32(max_depth as u32);
            builder.arg_u32(num_dim as u32);
            builder.arg_data(&onodes.data);
            builder.arg_data(&onode2depth.data);
            builder.arg_data(&onode2center.data);
            builder.arg_data(&idx2onode.data);
            builder.arg_data(&idx2center.data);
            builder.launch_kernel(fnc, cfg).unwrap();
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn quad_oct_tree_aggregate(
    _py: Python<'_>,
    idx2val: &Bound<'_, PyAny>,
    idx2onode: &Bound<'_, PyAny>,
    onodes: &Bound<'_, PyAny>,
    onode2aggval: &Bound<'_, PyAny>,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let idx2val = del_dlpack::get_managed_tensor_from_pyany(idx2val)?;
    let idx2onode = del_dlpack::get_managed_tensor_from_pyany(idx2onode)?;
    let onodes = del_dlpack::get_managed_tensor_from_pyany(onodes)?;
    let onode2aggval = del_dlpack::get_managed_tensor_from_pyany(onode2aggval)?;
    //
    let num_idx = del_dlpack::get_shape_tensor(idx2val, 0).unwrap();
    let num_vdim = del_dlpack::get_shape_tensor(idx2val, 1).unwrap();
    let num_onode = del_dlpack::get_shape_tensor(onodes, 0).unwrap();
    let num_link = del_dlpack::get_shape_tensor(onodes, 1).unwrap();
    let device = idx2val.ctx.device_type;
    //
    del_dlpack::check_2d_tensor::<f32>(idx2val, num_idx, num_vdim, device).unwrap();
    del_dlpack::check_1d_tensor::<u32>(idx2onode, num_idx, device).unwrap();
    del_dlpack::check_2d_tensor::<u32>(onodes, num_onode, num_link, device).unwrap();
    del_dlpack::check_2d_tensor::<f32>(onode2aggval, num_onode, num_vdim, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            let idx2val = unsafe { del_dlpack::slice_from_tensor::<f32>(idx2val).unwrap() };
            let idx2onode = unsafe { del_dlpack::slice_from_tensor::<u32>(idx2onode).unwrap() };
            let onodes = unsafe { del_dlpack::slice_from_tensor::<u32>(onodes).unwrap() };
            let onode2aggval =
                unsafe { del_dlpack::slice_from_tensor_mut::<f32>(onode2aggval).unwrap() };
            del_msh_cpu::quad_oct_tree::aggregate(
                num_vdim as usize,
                idx2val,
                idx2onode,
                num_link as usize,
                onodes,
                onode2aggval,
            );
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            use del_cudarc_sys::{cu, cuda_check};
            cuda_check!(cu::cuInit(0)).unwrap();
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            /*
            let (fnc, _mdl) = del_cudarc_sys::load_function_in_module(
                del_msh_cuda_kernel::QUAD_OCT_TREE,
                "aggregate",
            )
            .unwrap();
             */
            //let fnc = crate::load_get_function("quad_oct_tree", "aggregate").unwrap();
            let fnc = del_cudarc_sys::cache_func::get_function_cached(
                "del_msh::quad_oct_tree",
                del_msh_cuda_kernels::get("quad_oct_tree").unwrap(),
                "aggregate",
            )
            .unwrap();
            let cfg = del_cudarc_sys::LaunchConfig::for_num_elems(num_idx as u32);
            let mut builder = del_cudarc_sys::Builder::new(stream);
            builder.arg_u32(num_idx as u32);
            builder.arg_u32(num_vdim as u32);
            builder.arg_data(&idx2val.data);
            builder.arg_data(&idx2onode.data);
            builder.arg_u32(num_link as u32);
            builder.arg_data(&onodes.data);
            builder.arg_data(&onode2aggval.data);
            builder.launch_kernel(fnc, cfg).unwrap();
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}
