use del_cudarc::cudarc as cudarc;
use cudarc::driver::{CudaDevice, CudaSlice};

pub fn vtx2morton(
    dev: &std::sync::Arc<CudaDevice>,
    vtx2xyz: &CudaSlice<f32>,
    transform_xyz2uni: &CudaSlice<f32>,
    vtx2morton: &mut cudarc::driver::CudaViewMut<u32>,
) -> std::result::Result<(), cudarc::driver::DriverError> {
    use cudarc::driver::DeviceSlice;
    let num_vtx = vtx2xyz.len() / 3;
    let cfg = cudarc::driver::LaunchConfig::for_num_elems(num_vtx as u32);
    let param = (num_vtx, vtx2xyz, transform_xyz2uni, vtx2morton);
    let func =
        del_cudarc::get_or_load_func(dev, "vtx2morton", del_msh_cudarc_kernel::BVHNODES_MORTON)?;
    use cudarc::driver::LaunchAsync;
    unsafe { func.launch(cfg, param) }?;
    Ok(())
}

pub fn from_sorted_morton_codes(
    dev: &std::sync::Arc<CudaDevice>,
    bvnodes: &mut cudarc::driver::CudaViewMut<u32>,
    idx2morton: &cudarc::driver::CudaView<u32>,
    idx2tri: &cudarc::driver::CudaView<u32>,
) -> std::result::Result<(), cudarc::driver::DriverError> {
    use cudarc::driver::DeviceSlice;
    let num_leaf = idx2morton.len();
    let cfg = cudarc::driver::LaunchConfig::for_num_elems(num_leaf as u32);
    let param = (num_leaf, bvnodes, idx2morton, idx2tri);
    let func = del_cudarc::get_or_load_func(
        dev,
        "kernel_MortonCode_BVHTopology",
        del_msh_cudarc_kernel::BVHNODES_MORTON,
    )?;
    use cudarc::driver::LaunchAsync;
    unsafe { func.launch(cfg, param) }?;
    Ok(())
}
