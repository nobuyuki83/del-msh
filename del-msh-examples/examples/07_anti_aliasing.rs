fn write_silhouette_on_magnified_image(
    img_shape_lowres: (usize, usize),
    img_grayscale_lowres: &[f32],
    transform_world2ndc: &[f32; 16],
    edge2vtx_contour: &[u32],
    vtx2xyz: &[f32],
) -> anyhow::Result<()> {
    assert_eq!(
        img_grayscale_lowres.len(),
        img_shape_lowres.0 * img_shape_lowres.1
    );
    let img_rgb_lowres: Vec<f32> = img_grayscale_lowres.iter().flat_map(|&v| [v; 3]).collect();
    {
        let expansion_ratio = 9;
        let (img_shape_hires, mut img_hires) =
            del_canvas::expand_image(img_shape_lowres, &img_rgb_lowres, 3, expansion_ratio);
        let transform_world2pix_hires = {
            let transform_ndc2pix =
                del_geo_core::mat3_col_major::from_transform_ndc2pix(img_shape_hires);
            let transform_ndc2pix =
                del_geo_core::mat4_col_major::from_mat3_col_major_adding_z(&transform_ndc2pix);
            del_geo_core::mat4_col_major::mult_mat_col_major(
                &transform_ndc2pix,
                &transform_world2ndc,
            )
        };
        for node2vtx in edge2vtx_contour.chunks(2) {
            let (i0_vtx, i1_vtx) = (node2vtx[0], node2vtx[1]);
            let p0 = del_msh_cpu::vtx2xyz::to_xyz(&vtx2xyz, i0_vtx as usize).p;
            let p1 = del_msh_cpu::vtx2xyz::to_xyz(&vtx2xyz, i1_vtx as usize).p;
            use del_geo_core::vec3::Vec3;
            let q0 = p0
                .transform_homogeneous(&transform_world2pix_hires)
                .unwrap()
                .xy();
            let q1 = p1
                .transform_homogeneous(&transform_world2pix_hires)
                .unwrap()
                .xy();
            use slice_of_array::SliceNestExt;
            del_canvas::rasterize::line2::draw_dda_pixel_coordinate(
                img_hires.nest_mut(),
                img_shape_hires.0,
                &q0,
                &q1,
                [0., 1., 1.],
            );
        }
        del_canvas::write_png_from_float_image_rgb(
            "target/07_anti_aliasing_hires.png",
            &img_shape_hires,
            &img_hires,
        )?;
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let (tri2vtx, vtx2xyz) = del_msh_cpu::trimesh3_primitive::sphere_yup::<u32, f32>(0.8, 64, 64);

    /*
    let (tri2vtx, vtx2xyz) = {
        let mut obj = del_msh_cpu::io_obj::WavefrontObj::<u32, f32>::new();
        obj.load("asset/spot/spot_triangulated.obj")?;
        (obj.idx2vtx_xyz, obj.vtx2xyz)
    };
     */

    let bvhnodes = del_msh_cpu::bvhnodes_morton::from_triangle_mesh(&tri2vtx, &vtx2xyz, 3);
    let bvhnode2aabb = del_msh_cpu::bvhnode2aabb3::from_uniform_mesh_with_bvh(
        0,
        &bvhnodes,
        Some((&tri2vtx, 3)),
        &vtx2xyz,
        None,
    );
    //
    let img_asp = 1.0;
    let img_shape = (((16 * 6) as f32 * img_asp) as usize, 16 * 6);
    let cam_projection =
        del_geo_core::mat4_col_major::camera_perspective_blender(img_asp, 24f32, 0.3, 10.0, true);
    let cam_modelview =
        del_geo_core::mat4_col_major::camera_external_blender(&[0., 0., 2.0], 0., 0., 0.);

    // ----------------------

    let transform_world2ndc =
        del_geo_core::mat4_col_major::mult_mat_col_major(&cam_projection, &cam_modelview);
    let transform_ndc2world =
        del_geo_core::mat4_col_major::try_inverse(&transform_world2ndc).unwrap();
    let transform_world2pix = {
        let transform_ndc2pix = del_geo_core::mat3_col_major::from_transform_ndc2pix(img_shape);
        let transform_ndc2pix =
            del_geo_core::mat4_col_major::from_mat3_col_major_adding_z(&transform_ndc2pix);
        del_geo_core::mat4_col_major::mult_mat_col_major(&transform_ndc2pix, &transform_world2ndc)
    };

    let mut pix2tri = vec![0u32; img_shape.0 * img_shape.1];
    del_msh_cpu::trimesh3_raycast::update_pix2tri(
        &mut pix2tri,
        &tri2vtx,
        &vtx2xyz,
        &bvhnodes,
        &bvhnode2aabb,
        img_shape,
        &transform_ndc2world,
    );

    {
        let img_out = del_msh_cpu::trimesh3_raycast::render_normalmap_from_pix2tri(
            img_shape,
            &cam_modelview,
            &tri2vtx,
            &vtx2xyz,
            &pix2tri,
        );
        del_canvas::write_png_from_float_image_rgb(
            "target/07_anti_aliasing_normalmap.png",
            &img_shape,
            &img_out,
        )?;
    }

    let edge2vtx_contour = {
        let edge2vtx = del_msh_cpu::edge2vtx::from_triangle_mesh(&tri2vtx, vtx2xyz.len() / 3);
        let edge2tri = del_msh_cpu::edge2elem::from_edge2vtx_of_tri2vtx(
            &edge2vtx,
            &tri2vtx,
            vtx2xyz.len() / 3,
        );
        del_msh_cpu::edge2vtx::contour_for_triangle_mesh(
            &tri2vtx,
            &vtx2xyz,
            &transform_world2ndc,
            &edge2vtx,
            &edge2tri,
        )
    };

    let img_data = {
        let mut img_data = vec![0f32; img_shape.0 * img_shape.1];
        use rayon::prelude::*;
        img_data
            .par_iter_mut()
            .zip(pix2tri.par_iter())
            .for_each(|(a, &b)| {
                if b != u32::MAX {
                    *a = 1f32;
                }
            });
        del_msh_cpu::silhouette::update_image(
            &edge2vtx_contour,
            &vtx2xyz,
            &transform_world2pix,
            img_shape,
            &mut img_data,
            &pix2tri,
        );
        del_canvas::write_png_from_float_image_grayscale(
            "target/07_anti_aliasing_lowres.png",
            img_shape,
            &img_data,
        )?;
        img_data
    };
    write_silhouette_on_magnified_image(
        img_shape,
        &img_data,
        &transform_world2ndc,
        &edge2vtx_contour,
        &vtx2xyz,
    )?;
    {
        // compute loss for random target image
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaChaRng::seed_from_u64(0u64);
        let img_data_trg: Vec<f32> = (0..pix2tri.len())
            .into_iter()
            .map(|_i| rng.random::<f32>())
            .collect();
        let loss = img_data
            .iter()
            .zip(img_data_trg.iter())
            .map(|(&a, &b)| a * b)
            .sum::<f32>();
        let dldw_vtx2xyz = {
            let mut dldw_vtx2xyz = vec![0f32; vtx2xyz.len()];
            del_msh_cpu::silhouette::backward_wrt_vtx2xyz(
                &edge2vtx_contour,
                &vtx2xyz,
                &mut dldw_vtx2xyz,
                &transform_world2pix,
                img_shape,
                &img_data_trg,
                &pix2tri,
            );
            dldw_vtx2xyz
        };
        // the vertex to move
        let list_vtx_on_silhouette = {
            let unique: std::collections::HashSet<u32> =
                edge2vtx_contour.clone().into_iter().collect();
            Vec::from_iter(unique)
        };
        // check gradient
        let eps = 8.0e-4;
        let mut i_cnt_success = 0;
        for i_vtx in list_vtx_on_silhouette.iter().map(|&v| v as usize) {
            for i_dim in 0..3 {
                let vtx2xyz1 = {
                    let mut vtx2xyz1 = vtx2xyz.clone();
                    vtx2xyz1[i_vtx * 3 + i_dim] += eps;
                    vtx2xyz1
                };
                let mut img_data1 = vec![0f32; img_data.len()];
                use rayon::prelude::*;
                img_data1
                    .par_iter_mut()
                    .zip(pix2tri.par_iter())
                    .for_each(|(a, &b)| {
                        if b != u32::MAX {
                            *a = 1f32;
                        }
                    });
                del_msh_cpu::silhouette::update_image(
                    &edge2vtx_contour,
                    &vtx2xyz1,
                    &transform_world2pix,
                    img_shape,
                    &mut img_data1,
                    &pix2tri,
                );
                let loss1 = img_data1
                    .iter()
                    .zip(img_data_trg.iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<f32>();
                let diff_num = (loss1 - loss) / eps;
                let diff_ana = dldw_vtx2xyz[i_vtx * 3 + i_dim];
                let err = (diff_num - diff_ana).abs();
                // println!("{} {}", diff_num, diff_ana);
                let ratio = err / (diff_ana.abs() + 0.1);
                if ratio < 0.08 {
                    i_cnt_success += 1;
                }
            }
        }
        println!("{} {}", i_cnt_success, list_vtx_on_silhouette.len() * 3);
        assert!(i_cnt_success > ((list_vtx_on_silhouette.len() * 3) as f32 * 0.85f32) as usize);
    }
    Ok(())
}
