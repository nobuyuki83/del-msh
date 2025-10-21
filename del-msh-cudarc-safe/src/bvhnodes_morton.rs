use cudarc::driver::{CudaSlice, CudaStream, PushKernelArg};
use del_cudarc_safe::cudarc;

pub fn vtx2morton(
    stream: &std::sync::Arc<CudaStream>,
    vtx2xyz: &CudaSlice<f32>,
    transform_xyz2uni: &CudaSlice<f32>,
    vtx2morton: &mut cudarc::driver::CudaViewMut<u32>,
) -> Result<(), cudarc::driver::DriverError> {
    let num_vtx = vtx2xyz.len() / 3;
    let cfg = cudarc::driver::LaunchConfig::for_num_elems(num_vtx as u32);
    let func = del_cudarc_safe::get_or_load_func(
        stream.context(),
        "vtx2morton",
        del_msh_cuda_kernel::MORTONS,
    )?;
    let num_dim = 3;
    let mut builder = stream.launch_builder(&func);
    builder.arg(&num_vtx);
    builder.arg(vtx2xyz);
    builder.arg(&num_dim);
    builder.arg(transform_xyz2uni);
    builder.arg(vtx2morton);
    unsafe { builder.launch(cfg) }?;
    Ok(())
}

pub fn from_sorted_morton_codes(
    stream: &std::sync::Arc<CudaStream>,
    bvnodes: &mut cudarc::driver::CudaViewMut<u32>,
    idx2morton: &cudarc::driver::CudaView<u32>,
    idx2tri: &cudarc::driver::CudaView<u32>,
) -> Result<(), cudarc::driver::DriverError> {
    let num_leaf = idx2morton.len();
    let cfg = cudarc::driver::LaunchConfig::for_num_elems(num_leaf as u32);
    let func = del_cudarc_safe::get_or_load_func(
        stream.context(),
        "kernel_MortonCode_BVHTopology",
        del_msh_cuda_kernel::BVHNODES_MORTON,
    )?;
    {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&num_leaf);
        builder.arg(bvnodes);
        builder.arg(idx2morton);
        builder.arg(idx2tri);
        unsafe { builder.launch(cfg) }?;
    }
    Ok(())
}
