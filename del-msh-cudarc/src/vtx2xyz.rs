use del_cudarc::cudarc as cudarc;
use cudarc::driver::{CudaDevice, CudaSlice};

pub fn to_aabb3(
    dev: &std::sync::Arc<CudaDevice>,
    vtx2xyz: &CudaSlice<f32>,
) -> core::result::Result<CudaSlice<f32>, cudarc::driver::DriverError> {
    use cudarc::driver::DeviceSlice;
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
    let func = del_cudarc::get_or_load_func(
        dev,
        "kernel_MinMax_TPB256",
        del_msh_cudarc_kernel::AABB3_FROM_VTX2XYZ,
    )?;
    use cudarc::driver::LaunchAsync;
    unsafe { func.launch(cfg, param) }?;
    Ok(aabb)
}

#[test]
fn test_to_aabb3() -> Result<(), cudarc::driver::DriverError> {
    let (_tri2vtx, vtx2xyz) =
        del_msh_core::trimesh3_primitive::torus_zup::<u32, f32>(2.0, 1.0, 32, 32);
    let dev = cudarc::driver::CudaDevice::new(0)?;
    let vtx2xyz = dev.htod_copy(vtx2xyz)?;
    let aabb3 = to_aabb3(&dev, &vtx2xyz)?;
    let aabb3 = dev.dtoh_sync_copy(&aabb3)?;
    let aabb3 = arrayref::array_ref![&aabb3, 0, 6];
    use del_geo_core::vecn::Arr;
    assert!(aabb3.sub(&[-3., -3., -1., 3., 3., 1.]).norm() < 1.0e-8);
    Ok(())
}
