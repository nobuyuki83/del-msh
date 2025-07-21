#[cfg(feature = "cuda")]
use del_cudarc_safe::cudarc;

#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    let ctx = cudarc::driver::CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let (tri2vtx, vtx2xyz, _vtx2uv) = {
        let mut obj = del_msh_cpu::io_obj::WavefrontObj::<u32, f32>::new();
        obj.load("asset/spot/spot_triangulated.obj")?;
        obj.unified_xyz_uv_as_trimesh()
    };
    let bvhnodes = del_msh_cpu::bvhnodes_morton::from_triangle_mesh(&tri2vtx, &vtx2xyz, 3);
    let bvhnode2aabb = del_msh_cpu::bvhnode2aabb3::from_uniform_mesh_with_bvh(
        0,
        &bvhnodes,
        Some((&tri2vtx, 3)),
        &vtx2xyz,
        None,
    );
    let tri2vtx_dev = stream.memcpy_stod(&tri2vtx)?;
    let vtx2xyz_dev = stream.memcpy_stod(&vtx2xyz)?;
    let bvhnodes_dev = stream.memcpy_stod(&bvhnodes)?;
    let bvhnode2aabb_dev = stream.memcpy_stod(&bvhnode2aabb)?;
    // --------------
    let img_size = {
        const TILE_SIZE: usize = 16;
        (TILE_SIZE * 28, TILE_SIZE * 28)
    };
    let cam_projection = del_geo_core::mat4_col_major::camera_perspective_blender(
        img_size.0 as f32 / img_size.1 as f32,
        24f32,
        0.5,
        3.0,
        true,
    );
    let cam_modelview =
        del_geo_core::mat4_col_major::camera_external_blender(&[0., 0., 2.], 0., 0., 0.);

    let transform_world2ndc =
        del_geo_core::mat4_col_major::mult_mat_col_major(&cam_projection, &cam_modelview);
    let transform_ndc2world =
        del_geo_core::mat4_col_major::try_inverse(&transform_world2ndc).unwrap();
    //
    let func = del_cudarc_safe::get_or_load_func(
        stream.context(),
        "pix_to_tri",
        del_msh_cuda_kernel::PIX2TRI.into(),
    )?;
    // let pix_to_tri = dev.get_func("my_module", "pix_to_tri").unwrap();
    //
    let mut pix2tri_dev = stream.alloc_zeros::<u32>(img_size.1 * img_size.0)?;
    let transform_ndc2world_dev = stream.memcpy_stod(&transform_ndc2world)?;
    let now = std::time::Instant::now();
    let cfg = {
        let num_threads = 256;
        let num_blocks = (img_size.0 * img_size.1) / num_threads + 1;
        cudarc::driver::LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (num_threads as u32, 1, 1),
            shared_mem_bytes: 0,
        }
    };
    {
        use del_cudarc_safe::cudarc::driver::PushKernelArg;
        let mut builder = stream.launch_builder(&func);
        let num_tri = (tri2vtx.len() / 3) as u32;
        let img_width = img_size.0 as u32;
        let img_height = img_size.1 as u32;
        builder.arg(&mut pix2tri_dev);
        builder.arg(&num_tri);
        builder.arg(&tri2vtx_dev);
        builder.arg(&vtx2xyz_dev);
        builder.arg(&img_width);
        builder.arg(&img_height);
        builder.arg(&transform_ndc2world_dev);
        builder.arg(&bvhnodes_dev);
        builder.arg(&bvhnode2aabb_dev);
        unsafe { builder.launch(cfg) }?;
    }
    let pix2tri = stream.memcpy_dtov(&pix2tri_dev)?;
    println!("   Elapsed pix2tri: {:.2?}", now.elapsed());
    let pix2flag: Vec<f32> = pix2tri
        .iter()
        .map(|v| if *v == u32::MAX { 0f32 } else { 1f32 })
        .collect();
    del_canvas::write_png_from_float_image_grayscale(
        "target/raycast_trimesh3_silhouette.png",
        img_size,
        &pix2flag,
    )?;
    dbg!(tri2vtx.len());
    Ok(())
}
#[cfg(not(feature = "cuda"))]
fn main() {
    panic!("This example requires the cuda feature. \"--features cuda\"");
}
