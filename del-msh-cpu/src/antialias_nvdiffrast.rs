struct Neighbour {
    i_cnt: usize,
    is_horizontal: bool,
    list_i_pix: [usize; 5],
    list_pos_c: [[f32; 2]; 5],
}

impl Neighbour {
    fn new(i_pix: usize, img_width: usize, is_horizontal: bool) -> Self {
        let (iw1, ih1) = (i_pix % img_width, i_pix / img_width);
        let list_i_pix = [
            ih1 * img_width + iw1,       // c
            ih1 * img_width + iw1 - 1,   // w
            ih1 * img_width + iw1 + 1,   // e
            (ih1 - 1) * img_width + iw1, // s
            (ih1 + 1) * img_width + iw1, // n
        ];
        let list_pos_c = [
            [iw1 as f32 + 0.5, ih1 as f32 + 0.5], // c
            [iw1 as f32 - 0.5, ih1 as f32 + 0.5], // w
            [iw1 as f32 + 1.5, ih1 as f32 + 0.5], // e
            [iw1 as f32 + 0.5, ih1 as f32 - 0.5], // s
            [iw1 as f32 + 0.5, ih1 as f32 + 1.5], // n
        ];
        Neighbour {
            i_cnt: 0,
            list_i_pix,
            list_pos_c,
            is_horizontal,
        }
    }
}

impl Iterator for Neighbour {
    type Item = (usize, [f32; 2], usize, [f32; 2]);
    fn next(&mut self) -> Option<Self::Item> {
        if self.i_cnt >= 4 {
            return None;
        }
        let list_index = if self.is_horizontal {
            [(0, 1), (1, 0), (0, 2), (2, 0)]
        } else {
            [(0, 3), (3, 0), (0, 4), (4, 0)]
        };
        let (idx0, idx1) = list_index[self.i_cnt];
        let i_pix0 = self.list_i_pix[idx0];
        let i_pix1 = self.list_i_pix[idx1];
        let c0 = self.list_pos_c[idx0];
        let c1 = self.list_pos_c[idx1];
        self.i_cnt += 1;
        Some((i_pix0, c0, i_pix1, c1))
    }
}
pub fn fwd(
    edge2vtx_contour: &[u32],
    vtx2xyz: &[f32],
    transform_world2pix: &[f32; 16],
    img_shape: (usize, usize),
    img_data: &mut [f32],
    pix2tri: &[u32],
) {
    for node2vtx in edge2vtx_contour.chunks(2) {
        use del_geo_core::vec3::Vec3;
        let (i0_vtx, i1_vtx) = (node2vtx[0], node2vtx[1]);
        let q0 = crate::vtx2xyz::to_vec3(vtx2xyz, i0_vtx as usize)
            .transform_homogeneous(transform_world2pix)
            .unwrap()
            .xy();
        let q1 = crate::vtx2xyz::to_vec3(vtx2xyz, i1_vtx as usize)
            .transform_homogeneous(transform_world2pix)
            .unwrap()
            .xy();
        let v01 = del_geo_core::vec2::sub(&q1, &q0);
        let is_horizontal = v01[0].abs() < v01[1].abs();
        let list_pix = del_geo_core::edge2::overlapping_pixels_dda(img_shape, &q0, &q1);
        for &i_pix in list_pix.iter() {
            let hoge = Neighbour::new(i_pix, img_shape.0, is_horizontal);
            for (i_pix0, c0, i_pix1, c1) in hoge {
                // dbg!(i_pix0, c0, i_pix1, c1);
                if pix2tri[i_pix0] == u32::MAX || pix2tri[i_pix1] != u32::MAX {
                    continue;
                }
                let Some((rc, _re)) = del_geo_core::edge2::intersection_edge2(&c0, &c1, &q1, &q0)
                else {
                    continue;
                };
                assert!((0. ..1.0).contains(&rc));
                if rc < 0.5 {
                    img_data[i_pix0] = 0.5 + rc;
                } else {
                    img_data[i_pix1] = rc - 0.5;
                }
            }
        }
    }
}

pub fn bwd_wrt_vtx2xyz(
    edge2vtx_contour: &[u32],
    vtx2xyz: &[f32],
    dldw_vtx2xyz: &mut [f32],
    transform_world2pix: &[f32; 16],
    img_shape: (usize, usize),
    dldw_img_data: &[f32],
    pix2tri: &[u32],
) {
    use del_geo_core::mat2x3_col_major;
    use del_geo_core::mat3_col_major;
    use del_geo_core::mat4_col_major;
    for node2vtx in edge2vtx_contour.chunks(2) {
        use del_geo_core::vec3::Vec3;
        let (i0_vtx, i1_vtx) = (node2vtx[0], node2vtx[1]);
        let p0 = crate::vtx2xyz::to_xyz(vtx2xyz, i0_vtx as usize).p;
        let p1 = crate::vtx2xyz::to_xyz(vtx2xyz, i1_vtx as usize).p;
        let q0 = p0.transform_homogeneous(transform_world2pix).unwrap().xy();
        let q1 = p1.transform_homogeneous(transform_world2pix).unwrap().xy();
        let v01 = del_geo_core::vec2::sub(&q1, &q0);
        let is_horizontal = v01[0].abs() < v01[1].abs();
        let list_pix = del_geo_core::edge2::overlapping_pixels_dda(img_shape, &q0, &q1);
        for &i_pix in list_pix.iter() {
            let hoge = Neighbour::new(i_pix, img_shape.0, is_horizontal);
            for (i_pix0, c0, i_pix1, c1) in hoge {
                if pix2tri[i_pix0] == u32::MAX || pix2tri[i_pix1] != u32::MAX {
                    continue;
                } // zero in && one out
                let Some((rc, _re)) = del_geo_core::edge2::intersection_edge2(&c0, &c1, &q1, &q0)
                else {
                    continue;
                };
                // ---------------------------------
                assert!((0. ..=1.0).contains(&rc));
                let dldr0 = if rc < 0.5 {
                    dldw_img_data[i_pix0]
                } else {
                    dldw_img_data[i_pix1]
                };
                let (_dlc0, _dlc1, dldq1, dldq0) =
                    del_geo_core::edge2::dldw_intersection_edge2(&c0, &c1, &q1, &q0, dldr0, 0.0);
                let dqdp0 = mat4_col_major::jacobian_transform(transform_world2pix, p0);
                let dqdp1 = mat4_col_major::jacobian_transform(transform_world2pix, p1);
                let dqdp0 = mat3_col_major::to_mat2x3_col_major_xy(&dqdp0);
                let dqdp1 = mat3_col_major::to_mat2x3_col_major_xy(&dqdp1);
                let dldp0 = mat2x3_col_major::vec3_from_mult_transpose_vec2(&dqdp0, &dldq0);
                let dldp1 = mat2x3_col_major::vec3_from_mult_transpose_vec2(&dqdp1, &dldq1);
                use del_geo_core::vec3::Vec3;
                arrayref::array_mut_ref![dldw_vtx2xyz, (i0_vtx as usize) * 3, 3]
                    .add_in_place(&dldp0);
                arrayref::array_mut_ref![dldw_vtx2xyz, (i1_vtx as usize) * 3, 3]
                    .add_in_place(&dldp1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    const IMG_RES: usize = 256;

    fn geometry(eps: f32) -> (Vec<u32>, Vec<f32>, [f32; 16]) {
        let (tri2vtx, vtx2xyz) = crate::trimesh3_primitive::torus_zup::<u32, f32>(1.3, 0.4, 64, 32);
        let vtx2xyz = {
            let transform0 = del_geo_core::mat4_col_major::from_rot_x(1.15);
            let transform1 = del_geo_core::mat4_col_major::from_translate(&[eps, 0.6, 0.0]);
            let transform =
                del_geo_core::mat4_col_major::mult_mat_col_major(&transform1, &transform0);
            crate::vtx2xyz::transform_homogeneous(&vtx2xyz, &transform)
        };
        let transform_world2ndc = del_geo_core::mat4_col_major::from_diagonal(0.5, 0.5, 0.5, 1.0);
        (tri2vtx, vtx2xyz, transform_world2ndc)
    }

    fn sample(eps: f32, img_shape: (usize, usize), num_sample: usize) -> Vec<f32> {
        let (tri2vtx, vtx2xyz, transform_world2ndc) = geometry(eps);
        // let transform_ndc2world = del_geo_core::mat4_col_major::from_identity();
        let transform_ndc2world =
            del_geo_core::mat4_col_major::try_inverse_with_pivot(&transform_world2ndc).unwrap();
        let bvhnodes = crate::bvhnodes_morton::from_triangle_mesh(&tri2vtx, &vtx2xyz, 3);
        let bvhnode2aabb = crate::bvhnode2aabb3::from_uniform_mesh_with_bvh(
            0,
            &bvhnodes,
            Some((&tri2vtx, 3)),
            &vtx2xyz,
            None,
        );
        let fn_pix2val = |i_pix: usize| -> f32 {
            use rand::Rng;
            use rand::SeedableRng;
            let mut rng = rand_chacha::ChaChaRng::seed_from_u64(0);
            let i_h = i_pix / img_shape.0;
            let i_w = i_pix - i_h * img_shape.0;
            //
            let mut sum = 0.0f32;
            for _itr in 0..num_sample {
                let x_offset = rng.random_range(-0.5..0.5);
                let y_offset = rng.random_range(-0.5..0.5);
                let (ray_org, ray_dir) =
                    del_geo_core::mat4_col_major::ray_from_transform_ndc2world_and_pixel_coordinate(
                        (i_w as f32 + x_offset, i_h as f32 + y_offset),
                        &(img_shape.0 as f32, img_shape.1 as f32),
                        &transform_ndc2world,
                    );
                if let Some((_t, _i_tri)) = crate::search_bvh3::first_intersection_ray(
                    &ray_org,
                    &ray_dir,
                    &crate::search_bvh3::TriMeshWithBvh {
                        tri2vtx: &tri2vtx,
                        vtx2xyz: &vtx2xyz,
                        bvhnodes: &bvhnodes,
                        bvhnode2aabb: &bvhnode2aabb,
                    },
                    0,
                    f32::INFINITY,
                ) {
                    sum += 1.0;
                }
            }
            sum / num_sample as f32
        };
        let mut pix2val = vec![0f32; img_shape.0 * img_shape.1];
        use rayon::prelude::*;
        pix2val
            .par_iter_mut()
            .enumerate()
            .for_each(|(i_pix, i_tri)| *i_tri = fn_pix2val(i_pix));
        pix2val
    }

    #[test]
    fn test_sample() {
        let num_sample = 2056;
        let img_shape = (IMG_RES, IMG_RES);
        let eps = 1.0e-1 / IMG_RES as f32;
        let pix2val0 = sample(-eps, img_shape, num_sample);
        {
            let pix2val1 = sample(0., img_shape, num_sample);
            del_canvas::write_png_from_float_image_grayscale(
                "../target/silhouette_ren_finitesample.png",
                img_shape,
                &pix2val1,
            )
            .unwrap()
        }
        let pix2val2 = sample(eps, img_shape, num_sample);
        let pix2rgb_diff: Vec<_> = pix2val0
            .iter()
            .zip(pix2val2.iter())
            .flat_map(|(&v0, &v1)| {
                let grad = (v1 - v0) / (2.0 * eps);
                let c = del_canvas::colormap::apply_colormap(
                    grad,
                    -(IMG_RES as f32),
                    IMG_RES as f32,
                    del_canvas::colormap::COLORMAP_BWR,
                );
                c
            })
            .collect();
        del_canvas::write_png_from_float_image_rgb(
            "../target/silhouette_diff_finitesample.png",
            &img_shape,
            &pix2rgb_diff,
        )
        .unwrap()
    }

    fn nvdiffrast(eps: f32) -> Vec<f32> {
        let img_shape = (IMG_RES, IMG_RES);
        let (tri2vtx, vtx2xyz, transform_world2ndc) = geometry(eps);
        let transform_ndc2world =
            del_geo_core::mat4_col_major::try_inverse_with_pivot(&transform_world2ndc).unwrap();
        let transform_ndc2pix = del_geo_core::mat4_col_major::from_transform_ndc2pix(img_shape);
        let transform_world2pix = del_geo_core::mat4_col_major::mult_mat_col_major(
            &transform_ndc2pix,
            &transform_world2ndc,
        );
        //
        let num_vtx = vtx2xyz.len() / 3;
        let (_vtx2idx, _idx2vtx) = crate::vtx2vtx::from_uniform_mesh(&tri2vtx, 3, num_vtx, false);
        let edge2vtx = crate::edge2vtx::from_triangle_mesh(&tri2vtx, num_vtx);
        let _num_tri = tri2vtx.len() / 3;
        let _num_edge = edge2vtx.len() / 2;
        let edge2tri = crate::edge2elem::from_edge2vtx_of_tri2vtx(&edge2vtx, &tri2vtx, num_vtx);
        //
        let edge2vtx_contour = crate::edge2vtx::contour_for_triangle_mesh::<u32>(
            &tri2vtx,
            &vtx2xyz,
            &transform_world2ndc,
            &edge2vtx,
            &edge2tri,
        );
        let bvhnodes = crate::bvhnodes_morton::from_triangle_mesh(&tri2vtx, &vtx2xyz, 3);
        let bvhnode2aabb = crate::bvhnode2aabb3::from_uniform_mesh_with_bvh(
            0,
            &bvhnodes,
            Some((&tri2vtx, 3)),
            &vtx2xyz,
            None,
        );
        let pix2tri = {
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
        let mut img_data: Vec<f32> = pix2tri
            .iter()
            .map(|&v| if v == u32::MAX { 0. } else { 1. })
            .collect();
        crate::antialias_nvdiffrast::fwd(
            &edge2vtx_contour,
            &vtx2xyz,
            &transform_world2pix,
            img_shape,
            &mut img_data,
            &pix2tri,
        );
        img_data
    }

    #[test]
    fn test_nvdiffrast() {
        let img_shape = (IMG_RES, IMG_RES);
        let (tri2vtx, vtx2xyz, transform_world2ndc) = geometry(0.);
        let transform_ndc2world =
            del_geo_core::mat4_col_major::try_inverse_with_pivot(&transform_world2ndc).unwrap();
        let transform_ndc2pix = del_geo_core::mat4_col_major::from_transform_ndc2pix(img_shape);
        let transform_world2pix = del_geo_core::mat4_col_major::mult_mat_col_major(
            &transform_ndc2pix,
            &transform_world2ndc,
        );
        //
        let num_vtx = vtx2xyz.len() / 3;
        // let (_vtx2idx, _idx2vtx) = crate::vtx2vtx::from_uniform_mesh(&tri2vtx, 3, num_vtx, false);
        let edge2vtx = crate::edge2vtx::from_triangle_mesh(&tri2vtx, num_vtx);
        let _num_tri = tri2vtx.len() / 3;
        let _num_edge = edge2vtx.len() / 2;
        let edge2tri = crate::edge2elem::from_edge2vtx_of_tri2vtx(&edge2vtx, &tri2vtx, num_vtx);
        //
        let edge2vtx_contour = crate::edge2vtx::contour_for_triangle_mesh::<u32>(
            &tri2vtx,
            &vtx2xyz,
            &transform_world2ndc,
            &edge2vtx,
            &edge2tri,
        );
        //
        {
            let mut pix2val_edge = vec![1f32; img_shape.0 * img_shape.1];
            for chunk in edge2vtx_contour.chunks(2) {
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
                    &mut pix2val_edge,
                    img_shape.0,
                    &xy0_img,
                    &xy1_img,
                    0.,
                );
            }
            del_canvas::write_png_from_float_image_grayscale(
                "../target/silhouette_edge.png",
                img_shape,
                &pix2val_edge,
            )
            .unwrap();
        }
        let bvhnodes = crate::bvhnodes_morton::from_triangle_mesh(&tri2vtx, &vtx2xyz, 3);
        let bvhnode2aabb = crate::bvhnode2aabb3::from_uniform_mesh_with_bvh(
            0,
            &bvhnodes,
            Some((&tri2vtx, 3)),
            &vtx2xyz,
            None,
        );
        let pix2tri = {
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
        {
            let mut img_data: Vec<f32> = pix2tri
                .iter()
                .map(|&v| if v == u32::MAX { 0. } else { 1. })
                .collect();
            crate::antialias_nvdiffrast::fwd(
                &edge2vtx_contour,
                &vtx2xyz,
                &transform_world2pix,
                img_shape,
                &mut img_data,
                &pix2tri,
            );
            del_canvas::write_png_from_float_image_grayscale(
                "../target/silhouette_ren_nvdiffrast.png",
                img_shape,
                &img_data,
            )
            .unwrap()
        }
        let dxyz: Vec<f32> = (0..num_vtx).flat_map(|_| [1., 0., 0.]).collect();
        let mut pix2rgb_diff: Vec<_> = vec![0f32; img_shape.0 * img_shape.1 * 3];
        for i_pix in 0..IMG_RES * IMG_RES {
            let dldw_img = {
                let mut dldw_pix2val = vec![0f32; IMG_RES * IMG_RES];
                dldw_pix2val[i_pix] = 1.0;
                dldw_pix2val
            };
            let mut dldw_vtx2xyz = vec![0f32; vtx2xyz.len()];
            crate::antialias_nvdiffrast::bwd_wrt_vtx2xyz(
                &edge2vtx_contour,
                &vtx2xyz,
                &mut dldw_vtx2xyz,
                &transform_world2pix,
                img_shape,
                &dldw_img,
                &pix2tri,
            );
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
        del_canvas::write_png_from_float_image_rgb(
            "../target/silhouette_diff_nvdiffrast.png",
            &img_shape,
            &pix2rgb_diff,
        )
        .unwrap();
        //
        let eps = 1.0e-3;
        let pix2val0 = nvdiffrast(-eps);
        let pix2val2 = nvdiffrast(eps);
        let pix2rgb_diff: Vec<_> = pix2val0
            .iter()
            .zip(pix2val2.iter())
            .flat_map(|(&v0, &v1)| {
                let grad = (v1 - v0) / (2.0 * eps);
                let c = del_canvas::colormap::apply_colormap(
                    grad,
                    -(IMG_RES as f32),
                    IMG_RES as f32,
                    del_canvas::colormap::COLORMAP_BWR,
                );
                c
            })
            .collect();
        del_canvas::write_png_from_float_image_rgb(
            "../target/silhouette_diff_finitenvdiffrast.png",
            &img_shape,
            &pix2rgb_diff,
        )
        .unwrap()
    }
}
