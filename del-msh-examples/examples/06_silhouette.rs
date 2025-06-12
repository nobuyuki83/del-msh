fn main() -> anyhow::Result<()> {
    let (tri2vtx, vtx2xyz) = {
        let mut obj = del_msh_cpu::io_obj::WavefrontObj::<usize, f32>::new();
        obj.load("asset/spot/spot_triangulated.obj")?;
        (obj.idx2vtx_xyz, obj.vtx2xyz)
    };
    // let (tri2vtx, vtx2xyz) = del_msh_cpu::trimesh3_primitive::sphere_yup(0.8, 64, 64);
    let bvhnodes = del_msh_cpu::bvhnodes_morton::from_triangle_mesh(&tri2vtx, &vtx2xyz, 3);
    let bvhnode2aabb = del_msh_cpu::bvhnode2aabb3::from_uniform_mesh_with_bvh(
        0,
        &bvhnodes,
        Some((&tri2vtx, 3)),
        &vtx2xyz,
        None,
    );
    //
    let img_shape = {
        const TILE_SIZE: usize = 16;
        (TILE_SIZE * 28, TILE_SIZE * 28)
    };
    let cam_projection = del_geo_core::mat4_col_major::camera_perspective_blender(
        img_shape.0 as f32 / img_shape.1 as f32,
        24f32,
        0.3,
        10.0,
        true,
    );
    let cam_modelview =
        del_geo_core::mat4_col_major::camera_external_blender(&[0., 0., 2.0], 0., 0., 0.);
    let transform_world2ndc =
        del_geo_core::mat4_col_major::mult_mat_col_major(&cam_projection, &cam_modelview);
    dbg!(&transform_world2ndc);
    let edge2vtx_silhouette = {
        let edge2vtx =
            del_msh_cpu::edge2vtx::from_triangle_mesh(tri2vtx.as_slice(), vtx2xyz.len() / 3);
        let edge2tri = del_msh_cpu::edge2elem::from_edge2vtx_of_tri2vtx(
            &edge2vtx,
            &tri2vtx,
            vtx2xyz.len() / 3,
        );
        del_msh_cpu::edge2vtx::silhouette_for_triangle_mesh(
            //del_msh_cpu::edge2vtx::occluding_contour_for_triangle_mesh(
            &tri2vtx,
            &vtx2xyz,
            &transform_world2ndc,
            &edge2vtx,
            &edge2tri,
            &bvhnodes,
            &bvhnode2aabb,
        )
    };
    println!("# of edge in silhouette: {}", edge2vtx_silhouette.len() / 3);
    let transform_ndc2pix = del_geo_core::mat3_col_major::from_transform_ndc2pix(img_shape);
    let mut img_data = vec![[0f32; 3]; img_shape.0 * img_shape.1];
    {
        let transform_ndc2world =
            del_geo_core::mat4_col_major::try_inverse(&transform_world2ndc).unwrap();
        let mut pix2tri = vec![0usize; img_shape.0 * img_shape.1];
        del_msh_cpu::trimesh3_raycast::update_pix2tri(
            &mut pix2tri,
            &tri2vtx,
            &vtx2xyz,
            &bvhnodes,
            &bvhnode2aabb,
            img_shape,
            &transform_ndc2world,
        );
        let img_data_nrm = del_msh_cpu::trimesh3_raycast::render_normalmap_from_pix2tri(
            img_shape,
            &cam_modelview,
            &tri2vtx,
            &vtx2xyz,
            &pix2tri,
        );
        for i_pix in 0..img_shape.0 * img_shape.1 {
            img_data[i_pix][0] = img_data_nrm[i_pix * 3];
            img_data[i_pix][1] = img_data_nrm[i_pix * 3 + 1];
            img_data[i_pix][2] = img_data_nrm[i_pix * 3 + 2];
        }
    }
    for node2edge in edge2vtx_silhouette.chunks(2) {
        let (i0_vtx, i1_vtx) = (node2edge[0], node2edge[1]);
        let p0 = del_msh_cpu::vtx2xyz::to_xyz(&vtx2xyz, i0_vtx);
        let p1 = del_msh_cpu::vtx2xyz::to_xyz(&vtx2xyz, i1_vtx);
        let r0 = del_geo_core::mat4_col_major::transform_homogeneous(&transform_world2ndc, p0.p)
            .unwrap();
        let r1 = del_geo_core::mat4_col_major::transform_homogeneous(&transform_world2ndc, p1.p)
            .unwrap();
        del_canvas::rasterize::line2::draw_dda(
            &mut img_data,
            img_shape.0,
            &[r0[0], r0[1]],
            &[r1[0], r1[1]],
            &transform_ndc2pix,
            [0f32, 0f32, 1f32],
        );
    }
    use slice_of_array::SliceFlatExt;
    let img_data = img_data.flat();
    del_canvas::write_png_from_float_image_rgb("target/06_silhouette.png", &img_shape, img_data)?;
    Ok(())
}
