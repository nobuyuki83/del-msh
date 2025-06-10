fn main() -> anyhow::Result<()> {
    let (tri2vtx, vtx2xyz, vtx2uv) = {
        let mut obj = del_msh_cpu::io_obj::WavefrontObj::<usize, f32>::new();
        obj.load("asset/spot/spot_triangulated.obj")?;
        obj.unified_xyz_uv_as_trimesh()
    };
    let img_shape = {
        const TILE_SIZE: usize = 16;
        (TILE_SIZE * 28, TILE_SIZE * 28)
    };
    let cam_projection = del_geo_core::mat4_col_major::camera_perspective_blender(
        img_shape.0 as f32 / img_shape.1 as f32,
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
    /*
    {
        let vtx2xyz2 = del_msh_cpu::vtx2xyz::transform(&vtx2xyz, &transform_world2ndc);
        del_msh_cpu::io_obj::save_tri2vtx_vtx2xyz("target/hoge.obj", &tri2vtx, &vtx2xyz2, 3);
    }
     */
    let bvhnodes = del_msh_cpu::bvhnodes_morton::from_triangle_mesh(&tri2vtx, &vtx2xyz, 3);
    let aabbs = del_msh_cpu::bvhnode2aabb3::from_uniform_mesh_with_bvh(
        0,
        &bvhnodes,
        Some((&tri2vtx, 3)),
        &vtx2xyz,
        None,
    );
    let mut pix2tri = vec![0usize; img_shape.0 * img_shape.1];
    del_msh_cpu::trimesh3_raycast::update_pix2tri(
        &mut pix2tri,
        &tri2vtx,
        &vtx2xyz,
        &bvhnodes,
        &aabbs,
        img_shape,
        &transform_ndc2world,
    );

    {
        // render normalmap
        let pix2rgb = del_msh_cpu::trimesh3_raycast::render_normalmap_from_pix2tri(
            img_shape,
            &cam_modelview,
            &tri2vtx,
            &vtx2xyz,
            &pix2tri,
        );
        del_canvas::write_png_from_float_image_rgb(
            "target/05_trimesh3_normalmap.png",
            &img_shape,
            &pix2rgb,
        )?;
    }

    // render depth
    {
        let mut img_data = vec![0f32; img_shape.0 * img_shape.1];
        del_msh_cpu::trimesh3_raycast::render_depth_bvh(
            img_shape,
            &mut img_data,
            &transform_ndc2world,
            &tri2vtx,
            &vtx2xyz,
            &bvhnodes,
            &aabbs,
        );
        del_canvas::write_png_from_float_image_grayscale(
            "target/05_trimesh3_depth.png",
            img_shape,
            &img_data,
        )?;
    }

    {
        // render texture
        let (tex_data, tex_shape, bitdepth) =
            del_canvas::load_image_as_float_array("asset/spot/spot_texture.png")?;
        assert_eq!(bitdepth, 3);
        let img_data = del_msh_cpu::trimesh3_raycast::render_texture_from_pix2tri(
            img_shape,
            &transform_ndc2world,
            &tri2vtx,
            &vtx2xyz,
            &vtx2uv,
            &pix2tri,
            tex_shape,
            &tex_data,
            &del_msh_cpu::grid2::Interpolation::Nearest,
        );
        del_canvas::write_png_from_float_image_rgb(
            "target/05_trimesh3_texture_nearest.png",
            &img_shape,
            &img_data,
        )?;
        //
        let img_data = del_msh_cpu::trimesh3_raycast::render_texture_from_pix2tri(
            img_shape,
            &transform_ndc2world,
            &tri2vtx,
            &vtx2xyz,
            &vtx2uv,
            &pix2tri,
            tex_shape,
            &tex_data,
            &del_msh_cpu::grid2::Interpolation::Bilinear,
        );
        del_canvas::write_png_from_float_image_rgb(
            "target/05_trimesh3_texture_bilinear.png",
            &img_shape,
            &img_data,
        )?;
    }

    Ok(())
}
