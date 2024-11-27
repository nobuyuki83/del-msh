pub fn from_trimesh3_with_bvhnodes(
    dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
    tri2vtx: &cudarc::driver::CudaSlice<u32>,
    vtx2xyz: &cudarc::driver::CudaSlice<f32>,
    bvhnodes: &cudarc::driver::CudaSlice<u32>,
    bvhnode2aabb: &mut cudarc::driver::CudaSlice<f32>,
) -> anyhow::Result<()> {
    use cudarc::driver::DeviceSlice;
    let num_tri = tri2vtx.len() / 3;
    let num_bvhnode = bvhnodes.len() / 3;
    assert_eq!(num_bvhnode, 2 * num_tri - 1);
    let num_branch = num_tri - 1;
    let mut bvhbranch2flag = dev.alloc_zeros::<u32>(num_branch)?;
    let cfg = cudarc::driver::LaunchConfig::for_num_elems(num_tri as u32);
    let param = (
        bvhnode2aabb,
        &mut bvhbranch2flag,
        num_bvhnode,
        bvhnodes,
        num_tri as u32,
        tri2vtx,
        vtx2xyz,
        0.,
    );
    let from_trimsh =
        del_cudarc::get_or_load_func(dev, "from_trimesh3", del_msh_cudarc_kernel::BVHNODE2AABB)?;
    use cudarc::driver::LaunchAsync;
    unsafe { from_trimsh.launch(cfg, param) }?;
    Ok(())
}
