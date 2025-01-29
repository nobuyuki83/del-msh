use del_cudarc::cudarc as cudarc;

pub fn fwd(
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
    tri2vtx: &cudarc::driver::CudaSlice<u32>,
    vtx2xyz: &cudarc::driver::CudaSlice<f32>,
    edge2vtx: &cudarc::driver::CudaSlice<u32>,
    edge2tri: &cudarc::driver::CudaSlice<u32>,
    transform_world2ndc: &cudarc::driver::CudaSlice<f32>,
) -> std::result::Result<cudarc::driver::CudaSlice<u32>, cudarc::driver::DriverError> {
    use cudarc::driver::DeviceSlice;
    let num_edge = edge2vtx.len() / 2;
    let transform_ndc2world = {
        let transform_world2ndc_cpu = device.dtoh_sync_copy(transform_world2ndc)?;
        let transform_world2ndc_cpu = arrayref::array_ref![&transform_world2ndc_cpu, 0, 16];
        let transform_ndc2world_cpu =
            del_geo_core::mat4_col_major::try_inverse(transform_world2ndc_cpu).unwrap();
        device.htod_sync_copy(&transform_ndc2world_cpu)?
    };
    let edge2flg = {
        let mut edge2flg = device.alloc_zeros::<u32>(num_edge + 1)?;
        let cfg = cudarc::driver::LaunchConfig::for_num_elems(num_edge as u32);
        let param = (
            num_edge,
            &mut edge2flg,
            edge2vtx,
            edge2tri,
            tri2vtx,
            vtx2xyz,
            transform_world2ndc,
            &transform_ndc2world,
        );
        let func = del_cudarc::get_or_load_func(
            device,
            "edge2vtx_contour_set_flag",
            del_msh_cudarc_kernel::EDGE2VTX,
        )?;
        use cudarc::driver::LaunchAsync;
        unsafe { func.launch(cfg, param) }?;
        edge2flg
    };
    let edge2vtx_contour =
        del_cudarc::get_flagged_element::get_flagged_element(device, edge2vtx, &edge2flg)?;
    Ok(edge2vtx_contour)
}
