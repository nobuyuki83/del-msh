use cudarc::driver::{CudaSlice, CudaStream, PushKernelArg};
use del_cudarc_safe::cudarc;

pub fn to_aabb3(
    stream: &std::sync::Arc<CudaStream>,
    vtx2xyz: &CudaSlice<f32>,
) -> Result<CudaSlice<f32>, cudarc::driver::DriverError> {
    let num_vtx = vtx2xyz.len() / 3;
    let cfg = {
        let ngrid = (num_vtx - 1) / 256 + 1;
        cudarc::driver::LaunchConfig {
            grid_dim: (ngrid as u32, 1, 1),
            block_dim: (256, 3, 1),
            shared_mem_bytes: 0,
        }
    };
    let aabb = stream.memcpy_stod(&[f32::MAX, f32::MAX, f32::MAX, f32::MIN, f32::MIN, f32::MIN])?;
    let func = del_cudarc_safe::get_or_load_func(
        stream.context(),
        "kernel_MinMax_TPB256",
        del_msh_cudarc_kernel::AABB3_FROM_VTX2XYZ,
    )?;
    let mut builder = stream.launch_builder(&func);
    let num_vtx = num_vtx as u32;
    builder.arg(&aabb);
    builder.arg(vtx2xyz);
    builder.arg(&num_vtx);
    unsafe { builder.launch(cfg) }?;
    // let param = (&aabb, vtx2xyz, num_vtx);
    // unsafe { func.launch(cfg, param) }?;
    Ok(aabb)
}

#[test]
fn test_to_aabb3() -> Result<(), cudarc::driver::DriverError> {
    let (_tri2vtx, vtx2xyz) =
        del_msh_core::trimesh3_primitive::torus_zup::<u32, f32>(2.0, 1.0, 32, 32);
    let ctx = cudarc::driver::CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let vtx2xyz = stream.memcpy_stod(&vtx2xyz)?;
    let aabb3 = to_aabb3(&stream, &vtx2xyz)?;
    let aabb3 = stream.memcpy_dtov(&aabb3)?;
    let aabb3 = arrayref::array_ref![&aabb3, 0, 6];
    use del_geo_core::vecn::VecN;
    assert!(aabb3.sub(&[-3., -3., -1., 3., 3., 1.]).norm() < 1.0e-8);
    Ok(())
}
