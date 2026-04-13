#[cfg(test)]
mod tests {

    const IMG_RES: usize = 128;

    fn geometry(eps: f32) -> (Vec<u32>, Vec<f32>, [f32; 16], Vec<f32>) {
        let (tri2vtx, vtx2xyz) = crate::trimesh3_primitive::torus_zup::<u32, f32>(1.3, 0.4, 64, 32);
        let vtx2xyz = {
            let transform0 = del_geo_core::mat4_col_major::from_rot_x(1.15);
            //let transform0 = del_geo_core::mat4_col_major::from_rot_x(std::f32::consts::PI*0.25);
            let transform1 = del_geo_core::mat4_col_major::from_translate(&[eps, 0.6, 0.0]);
            //let transform1 = del_geo_core::mat4_col_major::from_translate(&[0.0, 0.6+eps, 0.0]);
            //let transform1 = del_geo_core::mat4_col_major::from_translate(&[0.0, 0.6, eps]);
            let transform =
                del_geo_core::mat4_col_major::mult_mat_col_major(&transform1, &transform0);
            crate::vtx2xyz::transform_homogeneous(&vtx2xyz, &transform)
        };
        let transform_world2ndc = del_geo_core::mat4_col_major::from_diagonal(0.5, 0.5, 0.5, 1.0);
        let dxyz: Vec<f32> = (0..vtx2xyz.len() / 3).flat_map(|_| [1., 0., 0.]).collect();
        //let dxyz: Vec<f32> = (0..vtx2xyz.len()/3).flat_map(|_| [0., 1., 0.]).collect();
        //let dxyz: Vec<f32> = (0..vtx2xyz.len()/3).flat_map(|_| [0., 0., 1.]).collect();
        (tri2vtx, vtx2xyz, transform_world2ndc, dxyz)
    }

    fn save_diff_image(path: &str, img_shape: (usize, usize), pix2grad: &[f32]) {
        let pix2rgb_diff: Vec<_> = pix2grad
            .iter()
            .flat_map(|&grad| {
                let c = del_canvas::colormap::apply_colormap(
                    grad,
                    //-0.5,
                    -(IMG_RES as f32) * 0.5,
                    //0.5,
                    (IMG_RES as f32) * 0.5,
                    del_canvas::colormap::COLORMAP_BWR,
                );
                c
            })
            .collect();
        del_canvas::write_png_from_float_image(path, img_shape, 3, &pix2rgb_diff).unwrap()
    }

    fn test_sample_for_model<T: crate::trimesh3_raycast::RenderTri + Sync>(
        mode: T,
        str_mode: &str,
    ) {
        use rand::SeedableRng;
        let num_sample = 2056;
        let img_shape = (IMG_RES, IMG_RES);
        let eps = 1.0e-1 / IMG_RES as f32;
        let pix2val0 = {
            let (tri2vtx, vtx2xyz, transform_world2ndc, _dxyz) = geometry(-eps);
            crate::trimesh3_raycast::multi_sample(
                &tri2vtx,
                &vtx2xyz,
                &transform_world2ndc,
                img_shape,
                num_sample,
                &mode,
                |i_pix| rand_chacha::ChaChaRng::seed_from_u64(i_pix as u64),
            )
        };
        {
            let (tri2vtx, vtx2xyz, transform_world2ndc, _dxyz) = geometry(0.);
            let pix2val1 = crate::trimesh3_raycast::multi_sample(
                &tri2vtx,
                &vtx2xyz,
                &transform_world2ndc,
                img_shape,
                num_sample,
                &mode,
                |i_pix| rand_chacha::ChaChaRng::seed_from_u64(i_pix as u64),
            );
            del_canvas::write_png_from_float_image(
                format!(
                    "../target/differentiable_rasterization_ren_finitesample_{}.png",
                    str_mode
                ),
                img_shape,
                1,
                &pix2val1,
            )
            .unwrap()
        }
        let pix2val2 = {
            let (tri2vtx, vtx2xyz, transform_world2ndc, _dxyz) = geometry(eps);
            crate::trimesh3_raycast::multi_sample(
                &tri2vtx,
                &vtx2xyz,
                &transform_world2ndc,
                img_shape,
                num_sample,
                &mode,
                |i_pix| rand_chacha::ChaChaRng::seed_from_u64(i_pix as u64),
            )
        };
        let grad: Vec<f32> = pix2val0
            .iter()
            .zip(pix2val2.iter())
            .map(|(&v0, &v2)| (v2 - v0) / (2.0 * eps))
            .collect();
        save_diff_image(
            &format!(
                "../target/differentiable_rasterization_diff_finitesample_{}.png",
                str_mode
            ),
            img_shape,
            &grad,
        );
    }

    #[test]
    fn test_sample() {
        test_sample_for_model(crate::trimesh3_raycast::Depth, "depth");
        test_sample_for_model(crate::trimesh3_raycast::Occlusion, "occ");
    }

    fn nvdiffrast<T: crate::trimesh3_raycast::RenderTri>(
        eps: f32,
        mode: &T,
    ) -> (Vec<f32>, Vec<f32>, Vec<u32>, Vec<u32>) {
        let img_shape = (IMG_RES, IMG_RES);
        let (tri2vtx, vtx2xyz, transform_world2ndc, _dxyz) = geometry(eps);
        let transform_ndc2world =
            del_geo_core::mat4_col_major::try_inverse_with_pivot(&transform_world2ndc).unwrap();
        let transform_ndc2pix = del_geo_core::mat4_col_major::from_transform_ndc2pix(img_shape);
        let transform_world2pix = del_geo_core::mat4_col_major::mult_mat_col_major(
            &transform_ndc2pix,
            &transform_world2ndc,
        );
        let num_vtx = vtx2xyz.len() / 3;
        let edge2vtx = crate::edge2vtx::from_triangle_mesh(&tri2vtx, num_vtx);
        let edge2tri = crate::edge2elem::from_edge2vtx_of_tri2vtx(&edge2vtx, &tri2vtx, num_vtx);
        let cedge2vtx = crate::edge2vtx::contour_for_triangle_mesh::<u32>(
            &tri2vtx,
            &vtx2xyz,
            &transform_world2ndc,
            &edge2vtx,
            &edge2tri,
        );
        let pix2tri = {
            let bvhnodes = crate::bvhnodes_morton::from_triangle_mesh(&tri2vtx, &vtx2xyz, 3);
            let bvhnode2aabb = crate::bvhnode2aabb3::from_uniform_mesh_with_bvh(
                0, &bvhnodes, &tri2vtx, 3, &vtx2xyz, None,
            );
            let mut pix2tri = vec![u32::MAX; IMG_RES * IMG_RES];
            crate::trimesh3_raycast::update_pix2tri(
                &mut pix2tri,
                &tri2vtx,
                &vtx2xyz,
                &bvhnodes,
                &bvhnode2aabb,
                img_shape,
                &transform_ndc2world,
            );
            pix2tri
        };
        let pix2vin = crate::trimesh3_raycast::fwd_continuous(
            &pix2tri,
            img_shape,
            &tri2vtx,
            &vtx2xyz,
            &transform_ndc2world,
            mode,
        );
        let mut pix2vout = pix2vin.clone();
        crate::antialias::antialias(
            &cedge2vtx,
            &vtx2xyz,
            &transform_world2pix,
            img_shape,
            &pix2tri,
            &pix2vin,
            &mut pix2vout,
        );
        (pix2vout, pix2vin, pix2tri, cedge2vtx)
    }

    #[test]
    fn test_contour() {
        let img_shape = (IMG_RES, IMG_RES);
        let (_tri2vtx, vtx2xyz, transform_world2ndc, _dxyz) = geometry(0.);
        let (_, _, _, cedge2vtx) = nvdiffrast(0., &crate::trimesh3_raycast::Occlusion);
        // draw edge image
        let mut pix2isedge = vec![1f32; img_shape.0 * img_shape.1];
        for chunk in cedge2vtx.chunks(2) {
            let (iv0, iv1) = (chunk[0] as usize, chunk[1] as usize);
            let xyz0_world = crate::vtx2xyz::to_vec3(&vtx2xyz, iv0);
            let xyz1_world = crate::vtx2xyz::to_vec3(&vtx2xyz, iv1);
            let xyz0_ndc = del_geo_core::mat4_col_major::transform_homogeneous(
                &transform_world2ndc,
                xyz0_world,
            )
            .unwrap();
            let xyz1_ndc = del_geo_core::mat4_col_major::transform_homogeneous(
                &transform_world2ndc,
                xyz1_world,
            )
            .unwrap();
            let xy0_img = del_geo_core::ndc::to_image_coordinate(&xyz0_ndc, img_shape);
            let xy1_img = del_geo_core::ndc::to_image_coordinate(&xyz1_ndc, img_shape);
            del_canvas::rasterize::line2::draw_dda_pixel_coordinate(
                &mut pix2isedge,
                img_shape.0,
                &xy0_img,
                &xy1_img,
                0.,
            );
        }
        del_canvas::write_png_from_float_image(
            "../target/differentiable_rasterizer_edge.png",
            img_shape,
            1,
            &pix2isedge,
        )
        .unwrap();
    }

    fn test_nvdiffrast_for_model<T: crate::trimesh3_raycast::RenderTri>(mode: &T, suffix: &str) {
        let img_shape = (IMG_RES, IMG_RES);
        let (tri2vtx, vtx2xyz, transform_world2ndc, dxyz) = geometry(0.);
        let transform_ndc2world =
            del_geo_core::mat4_col_major::try_inverse_with_pivot(&transform_world2ndc).unwrap();
        let transform_ndc2pix = del_geo_core::mat4_col_major::from_transform_ndc2pix(img_shape);
        let transform_world2pix = del_geo_core::mat4_col_major::mult_mat_col_major(
            &transform_ndc2pix,
            &transform_world2ndc,
        );
        // -------------------
        let (pix2vout, pix2vin, pix2tri, cedge2vtx) = nvdiffrast(0., mode);
        del_canvas::write_png_from_float_image(
            &format!(
                "../target/differentiable_rasterizer_ren_nvdiffrast_{}.png",
                suffix
            ),
            img_shape,
            1,
            &pix2vout,
        )
        .unwrap();
        {
            let mut pix2grad: Vec<_> = vec![0f32; img_shape.0 * img_shape.1];
            for i_pix in 0..IMG_RES * IMG_RES {
                let dldw_pix2val = {
                    let mut dldw_pix2val = vec![0f32; IMG_RES * IMG_RES];
                    dldw_pix2val[i_pix] = 1.0;
                    dldw_pix2val
                };
                let mut dldw_vtx2xyz = vec![0f32; vtx2xyz.len()];
                crate::trimesh3_raycast::bwd_continuous(
                    &pix2tri,
                    &tri2vtx,
                    &vtx2xyz,
                    &dldw_pix2val,
                    &transform_ndc2world,
                    img_shape,
                    &mut dldw_vtx2xyz,
                    mode,
                );
                crate::antialias::bwd_antialias(
                    &cedge2vtx,
                    &vtx2xyz,
                    &mut dldw_vtx2xyz,
                    &transform_world2pix,
                    img_shape,
                    &pix2vin,
                    &dldw_pix2val,
                    &pix2tri,
                );
                let grad: f32 = dxyz
                    .iter()
                    .zip(dldw_vtx2xyz.iter())
                    .map(|(&v0, &v1)| v0 * v1)
                    .sum();
                pix2grad[i_pix] = grad;
            }
            save_diff_image(
                &format!(
                    "../target/differentiable_rasterizer_diff_bwdnvdiffrast_{}.png",
                    suffix
                ),
                img_shape,
                &pix2grad,
            );
        }
        {
            let eps = 3.0e-3;
            let (pix2val0, _, _, _) = nvdiffrast(-eps, mode);
            let (pix2val2, _, _, _) = nvdiffrast(eps, mode);
            let grad: Vec<f32> = pix2val0
                .iter()
                .zip(pix2val2.iter())
                .map(|(&v0, &v2)| (v2 - v0) / (2.0 * eps))
                .collect();
            save_diff_image(
                &format!(
                    "../target/differentiable_rasterizer_diff_finitenvdiffrast_{}.png",
                    suffix
                ),
                img_shape,
                &grad,
            );
        }
    }

    #[test]
    fn test_nvdiffrast() {
        test_nvdiffrast_for_model(&crate::trimesh3_raycast::Occlusion, "occ");
        test_nvdiffrast_for_model(&crate::trimesh3_raycast::Depth, "depth");
    }

    #[test]
    fn test_microedge() {
        let img_shape = (IMG_RES, IMG_RES);
        let (tri2vtx, vtx2xyz, transform_world2ndc, dxyz) = geometry(0.);
        let transform_ndc2world =
            del_geo_core::mat4_col_major::try_inverse_with_pivot(&transform_world2ndc).unwrap();
        let transform_ndc2pix = del_geo_core::mat4_col_major::from_transform_ndc2pix(img_shape);
        let transform_world2pix = del_geo_core::mat4_col_major::mult_mat_col_major(
            &transform_ndc2pix,
            &transform_world2ndc,
        );
        let pix2tri = {
            let bvhnodes = crate::bvhnodes_morton::from_triangle_mesh(&tri2vtx, &vtx2xyz, 3);
            let bvhnode2aabb = crate::bvhnode2aabb3::from_uniform_mesh_with_bvh(
                0, &bvhnodes, &tri2vtx, 3, &vtx2xyz, None,
            );
            let mut pix2tri = vec![u32::MAX; IMG_RES * IMG_RES];
            crate::trimesh3_raycast::update_pix2tri(
                &mut pix2tri,
                &tri2vtx,
                &vtx2xyz,
                &bvhnodes,
                &bvhnode2aabb,
                img_shape,
                &transform_ndc2world,
            );
            pix2tri
        };
        let mut pix2rgb_diff: Vec<_> = vec![0f32; img_shape.0 * img_shape.1 * 3];
        for i_pix in 0..IMG_RES * IMG_RES {
            let dldw_pixval = {
                let mut dldw_pix2val = vec![0f32; IMG_RES * IMG_RES];
                dldw_pix2val[i_pix] = 1.0;
                dldw_pix2val
            };
            let mut dldw_vtx2xyz = vec![0f32; vtx2xyz.len()];
            //
            crate::microedge::bwd_microedge(
                &tri2vtx,
                &vtx2xyz,
                &mut dldw_vtx2xyz,
                &transform_world2pix,
                img_shape,
                &dldw_pixval,
                &pix2tri,
            );
            //
            let dpix: f32 = dxyz
                .iter()
                .zip(dldw_vtx2xyz.iter())
                .map(|(&v0, &v1)| v0 * v1)
                .sum();
            let c = del_canvas::colormap::apply_colormap(
                dpix,
                -(IMG_RES as f32),
                IMG_RES as f32,
                del_canvas::colormap::COLORMAP_BWR,
            );
            pix2rgb_diff[i_pix * 3..(i_pix + 1) * 3].copy_from_slice(&c);
        }
        del_canvas::write_png_from_float_image(
            "../target/silhouette_diff_microedge.png",
            img_shape,
            3,
            &pix2rgb_diff,
        )
        .unwrap();
    }
}
