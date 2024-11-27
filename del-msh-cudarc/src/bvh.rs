use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};

pub fn tri2cntr_from_trimesh3(
    dev: &std::sync::Arc<CudaDevice>,
    tri2vtx: &CudaSlice<u32>,
    vtx2xyz: &CudaSlice<f32>,
    tri2cntr: &mut CudaSlice<f32>,
) -> anyhow::Result<()> {
    let num_tri = tri2vtx.len() / 3;
    let cfg = cudarc::driver::LaunchConfig::for_num_elems(num_tri as u32);
    let param = (tri2cntr, num_tri as u32, tri2vtx, vtx2xyz);
    let from_trimsh =
        del_cudarc_util::get_or_load_func(dev, "tri2cntr", del_msh_cudarc_kernel::BVHNODES_MORTON)?;
    use cudarc::driver::LaunchAsync;
    unsafe { from_trimsh.launch(cfg, param) }?;
    Ok(())
}
