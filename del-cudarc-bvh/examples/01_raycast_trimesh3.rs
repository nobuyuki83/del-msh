#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, LaunchAsync};

#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    let dev = CudaDevice::new(0)?;
    let (tri2vtx, vtx2xyz, _vtx2uv) = {
        let mut obj = del_msh_core::io_obj::WavefrontObj::<u32, f32>::new();
        obj.load("asset/spot/spot_triangulated.obj")?;
        obj.unified_xyz_uv_as_trimesh()
    };
    del_cudarc_bvh::assert_equal_cpu_gpu(&dev, &tri2vtx, &vtx2xyz)?;
    let bvhnodes = del_msh_core::bvhnodes_morton::from_triangle_mesh(&tri2vtx, &vtx2xyz, 3);
    let bvhnode2aabb = del_msh_core::bvhnode2aabb3::from_uniform_mesh_with_bvh(
        0,
        &bvhnodes,
        Some((&tri2vtx, 3)),
        &vtx2xyz,
        None,
    );
    let tri2vtx_dev = dev.htod_copy(tri2vtx.clone())?;
    let vtx2xyz_dev = dev.htod_copy(vtx2xyz.clone())?;
    let bvhnodes_dev = dev.htod_copy(bvhnodes.clone())?;
    let bvhnode2aabb_dev = dev.htod_copy(bvhnode2aabb.clone())?;
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
        del_geo_core::mat4_col_major::mult_mat(&cam_projection, &cam_modelview);
    let transform_ndc2world =
        del_geo_core::mat4_col_major::try_inverse(&transform_world2ndc).unwrap();
    //
    dev.load_ptx(kernel_bvh::PIX2TRI.into(), "my_module", &["pix_to_tri"])?;
    let pix_to_tri = dev.get_func("my_module", "pix_to_tri").unwrap();
    //
    let mut pix2tri_dev = dev.alloc_zeros::<u32>(img_size.1 * img_size.0)?;
    let transform_ndc2world_dev = dev.htod_copy(transform_ndc2world.to_vec())?;
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
    //for_num_elems((img_size.0 * img_size.1).try_into()?);
    let param = (
        &mut pix2tri_dev,
        tri2vtx.len() / 3,
        &tri2vtx_dev,
        &vtx2xyz_dev,
        img_size.0,
        img_size.1,
        &transform_ndc2world_dev,
        &bvhnodes_dev,
        &bvhnode2aabb_dev,
    );
    unsafe { pix_to_tri.launch(cfg, param) }?;
    let pix2tri = dev.dtoh_sync_copy(&pix2tri_dev)?;
    println!("   Elapsed pix2tri: {:.2?}", now.elapsed());
    let pix2flag: Vec<f32> = pix2tri
        .iter()
        .map(|v| if *v == u32::MAX { 0f32 } else { 1f32 })
        .collect();
    del_canvas_image::write_png_from_float_image_grayscale(
        "target/raycast_trimesh3_silhouette.png",
        &img_size,
        &pix2flag,
    )?;
    dbg!(tri2vtx.len());
    Ok(())
}
#[cfg(not(feature = "cuda"))]
fn main() {}
