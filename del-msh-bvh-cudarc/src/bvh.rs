use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};

pub fn bvhnode2aabb_from_trimesh_with_bvhnodes(
    dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
    tri2vtx: &cudarc::driver::CudaSlice<u32>,
    vtx2xyz: &cudarc::driver::CudaSlice<f32>,
    bvhnodes: &cudarc::driver::CudaSlice<u32>,
    bvhnode2aabb: &mut cudarc::driver::CudaSlice<f32>,
) -> anyhow::Result<()> {
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
    let from_trimsh = del_cudarc_util::get_or_load_func(
        dev,
        "from_trimesh3",
        del_msh_bvh_cudarc_kernel::BVHNODE2AABB,
    )?;
    use cudarc::driver::LaunchAsync;
    unsafe { from_trimsh.launch(cfg, param) }?;
    Ok(())
}

pub fn tri2cntr_from_trimesh3(
    dev: &std::sync::Arc<CudaDevice>,
    tri2vtx: &CudaSlice<u32>,
    vtx2xyz: &CudaSlice<f32>,
    tri2cntr: &mut CudaSlice<f32>,
) -> anyhow::Result<()> {
    let num_tri = tri2vtx.len() / 3;
    let cfg = cudarc::driver::LaunchConfig::for_num_elems(num_tri as u32);
    let param = (tri2cntr, num_tri as u32, tri2vtx, vtx2xyz);
    let from_trimsh = del_cudarc_util::get_or_load_func(
        dev,
        "tri2cntr",
        del_msh_bvh_cudarc_kernel::BVHNODES_MORTON,
    )?;
    use cudarc::driver::LaunchAsync;
    unsafe { from_trimsh.launch(cfg, param) }?;
    Ok(())
}

pub fn vtx2morton(
    dev: &std::sync::Arc<CudaDevice>,
    vtx2xyz: &CudaSlice<f32>,
    transform_xyz2uni: &CudaSlice<f32>,
    vtx2morton: &mut CudaSlice<u32>,
) -> anyhow::Result<()> {
    let num_vtx = vtx2xyz.len() / 3;
    let cfg = cudarc::driver::LaunchConfig::for_num_elems(num_vtx as u32);
    let param = (num_vtx, vtx2xyz, transform_xyz2uni, vtx2morton);
    let func = del_cudarc_util::get_or_load_func(
        dev,
        "vtx2morton",
        del_msh_bvh_cudarc_kernel::BVHNODES_MORTON,
    )?;
    use cudarc::driver::LaunchAsync;
    unsafe { func.launch(cfg, param) }?;
    Ok(())
}

#[cfg(feature = "cuda")]
pub fn bvhnodes_from_sorted_morton_codes(
    dev: &std::sync::Arc<CudaDevice>,
    bvnodes: &mut CudaSlice<u32>,
    idx2morton: &CudaSlice<u32>,
    idx2tri: &CudaSlice<u32>,
) -> anyhow::Result<()> {
    let num_leaf = idx2morton.len();
    let cfg = cudarc::driver::LaunchConfig::for_num_elems(num_leaf as u32);
    let param = (num_leaf, bvnodes, idx2morton, idx2tri);
    let func = del_cudarc_util::get_or_load_func(
        dev,
        "kernel_MortonCode_BVHTopology",
        del_msh_bvh_cudarc_kernel::BVHNODES_MORTON,
    )?;
    use cudarc::driver::LaunchAsync;
    unsafe { func.launch(cfg, param) }?;
    Ok(())
}

pub fn aabb3_from_vtx2xyz(
    dev: &std::sync::Arc<CudaDevice>,
    vtx2xyz: &CudaSlice<f32>,
) -> anyhow::Result<CudaSlice<f32>> {
    let num_vtx = vtx2xyz.len() / 3;
    let cfg = {
        let ngrid = (num_vtx - 1) / 256 + 1;
        cudarc::driver::LaunchConfig {
            grid_dim: (ngrid as u32, 1, 1),
            block_dim: (256, 3, 1),
            shared_mem_bytes: 0,
        }
    };

    let aabb = dev.htod_copy(vec![
        f32::MAX,
        f32::MAX,
        f32::MAX,
        f32::MIN,
        f32::MIN,
        f32::MIN,
    ])?;
    let param = (&aabb, vtx2xyz, num_vtx);
    let func = del_cudarc_util::get_or_load_func(
        dev,
        "kernel_MinMax_TPB256",
        del_msh_bvh_cudarc_kernel::AABB3_FROM_VTX2XYZ,
    )?;
    use cudarc::driver::LaunchAsync;
    unsafe { func.launch(cfg, param) }?;
    Ok(aabb)
}
