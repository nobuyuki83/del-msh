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
pub fn antialias(
    edge2vtx_contour: &[u32],
    vtx2xyz: &[f32],
    transform_world2pix: &[f32; 16],
    img_shape: (usize, usize),
    pix2val: &mut [f32],
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
                if pix2tri[i_pix0] == u32::MAX || pix2tri[i_pix1] != u32::MAX { // pix0 is background
                    continue;
                }
                let Some((rc, _re)) = del_geo_core::edge2::intersection_edge2(&c0, &c1, &q1, &q0)
                else {
                    continue;
                };
                assert!((0. ..1.0).contains(&rc));
                if rc < 0.5 {
                    pix2val[i_pix0] = 0.5 + rc;
                } else {
                    pix2val[i_pix1] = rc - 0.5;
                }
            }
        }
    }
}

pub fn bwd_antialias(
    edge2vtx_contour: &[u32],
    vtx2xyz: &[f32],
    dldw_vtx2xyz: &mut [f32],
    transform_world2pix: &[f32; 16],
    img_shape: (usize, usize),
    dldw_pixval: &[f32],
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
                    dldw_pixval[i_pix0]
                } else {
                    dldw_pixval[i_pix1]
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

pub fn bwd_microedge(
    tri2vtx: &[u32],
    vtx2xyz: &[f32],
    dldw_vtx2xyz: &mut [f32],
    transform_world2pix: &[f32; 16],
    (img_w, img_h): (usize, usize),
    dldw_pixval: &[f32],
    pix2tri: &[u32],
) {
    use del_geo_core::mat4_col_major::Mat4ColMajor;
    let fn_barycentric = |pixcntr0: &[f32; 2], itri1: u32| {
        if itri1 == u32::MAX {
            None
        } else {
            use del_geo_core::mat4_col_major::Mat4ColMajor;
            use del_geo_core::vec3::Vec3;
            let itri1 = itri1 as usize;
            let i0 = tri2vtx[itri1 * 3] as usize;
            let i1 = tri2vtx[itri1 * 3 + 1] as usize;
            let i2 = tri2vtx[itri1 * 3 + 2] as usize;
            let xyz0 = arrayref::array_ref![vtx2xyz, i0 * 3, 3];
            let xyz1 = arrayref::array_ref![vtx2xyz, i1 * 3, 3];
            let xyz2 = arrayref::array_ref![vtx2xyz, i2 * 3, 3];
            let p0 = transform_world2pix
                .transform_homogeneous(xyz0)
                .unwrap()
                .xy();
            let p1 = transform_world2pix
                .transform_homogeneous(xyz1)
                .unwrap()
                .xy();
            let p2 = transform_world2pix
                .transform_homogeneous(xyz2)
                .unwrap()
                .xy();
            del_geo_core::tri2::barycentric_coords(&p0, &p1, &p2, pixcntr0)
        }
    };
    let fn_inside = |b: Option<(f32, f32, f32)>| {
        if let Some(b0) = b {
            if (b0.0 >= 0. && b0.1 >= 0. && b0.2 >= 0.) ||
                (b0.0 <= 0. && b0.1 <= 0. && b0.2 <= 0.) {
                return true;
            }
            false
        } else {
            true
        }
    };
    let transform_pix2world = transform_world2pix.transpose();
    // horizontal edge
    for iw in 0..img_w {
        for ih0 in 0..img_h - 1 {
            let ih1 = ih0 + 1;
            let ipix0 = ih0 * img_w + iw;
            let ipix1 = ih1 * img_w + iw;
            let itri0 = pix2tri[ipix0];
            let itri1 = pix2tri[ipix1];
            if itri0 == itri1 {
                continue;
            } // no edge
            let pixcntr0 = [iw as f32 + 0.5, ih0 as f32 + 0.5];
            let pixcntr1 = [iw as f32 + 0.5, ih1 as f32 + 0.5];
            let is_pixcentr0_inside_tri1 = fn_inside(fn_barycentric(&pixcntr0, itri1));
            let is_pixcentr1_inside_tri0 = fn_inside(fn_barycentric(&pixcntr1, itri0));
            if !is_pixcentr0_inside_tri1 && !is_pixcentr1_inside_tri0 {
                continue;
            }
            let val0 = if pix2tri[ipix0] == u32::MAX { 0. } else { 1. };
            let val1 = if pix2tri[ipix1] == u32::MAX { 0. } else { 1. };
            let dldpa = (dldw_pixval[ipix0] + dldw_pixval[ipix1]) * 0.5 * (val0 - val1);
            if is_pixcentr0_inside_tri1 && is_pixcentr1_inside_tri0 {
                dbg!("todo");
                continue;
            } else {
                if is_pixcentr1_inside_tri0 {
                    // only tri1 recieve gradient
                    let b = fn_barycentric(&pixcntr1, itri1).unwrap();
                    let b = [b.0, b.1, b.2];
                    let itri1 = itri1 as usize;
                    use del_geo_core::mat4_col_major::Mat4ColMajor;
                    let dxyz = transform_pix2world.transform_direction(&[0., 1., 0.]);
                    for inode in 0..3 {
                        let ivtx = tri2vtx[itri1 * 3 + inode] as usize;
                        dldw_vtx2xyz[ivtx * 3] += b[inode] * dxyz[0] * dldpa;
                        dldw_vtx2xyz[ivtx * 3 + 1] += b[inode] * dxyz[1] * dldpa;
                        dldw_vtx2xyz[ivtx * 3 + 2] += b[inode] * dxyz[2] * dldpa;
                    }
                } else {
                    // only tri0 recieve gradient
                    let b = fn_barycentric(&pixcntr0, itri0).unwrap();
                    let b = [b.0, b.1, b.2];
                    let itri0 = itri0 as usize;
                    use del_geo_core::mat4_col_major::Mat4ColMajor;
                    let dxyz = transform_pix2world.transform_direction(&[0., 1., 0.]);
                    for inode in 0..3 {
                        let ivtx = tri2vtx[itri0 * 3 + inode] as usize;
                        dldw_vtx2xyz[ivtx * 3] += b[inode] * dxyz[0] * dldpa;
                        dldw_vtx2xyz[ivtx * 3 + 1] += b[inode] * dxyz[1] * dldpa;
                        dldw_vtx2xyz[ivtx * 3 + 2] += b[inode] * dxyz[2] * dldpa;
                    }
                }
            }
        }
    }

    // horizontal edge
    for iw0 in 0..img_w - 1 {
        for ih in 0..img_h {
            let iw1 = iw0 + 1;
            let ipix0 = ih * img_w + iw0;
            let ipix1 = ih * img_w + iw1;
            let itri0 = pix2tri[ipix0];
            let itri1 = pix2tri[ipix1];
            if itri0 == itri1 {
                continue;
            } // no edge
            let pixcntr0 = [iw0 as f32 + 0.5, ih as f32 + 0.5];
            let pixcntr1 = [iw1 as f32 + 0.5, ih as f32 + 0.5];
            let is_pixcentr0_inside_tri1 = fn_inside(fn_barycentric(&pixcntr0, itri1));
            let is_pixcentr1_inside_tri0 = fn_inside(fn_barycentric(&pixcntr1, itri0));
            if !is_pixcentr0_inside_tri1 && !is_pixcentr1_inside_tri0 {
                continue;
            }
            let val0 = if pix2tri[ipix0] == u32::MAX { 0. } else { 1. };
            let val1 = if pix2tri[ipix1] == u32::MAX { 0. } else { 1. };
            let dldpa = (dldw_pixval[ipix0] + dldw_pixval[ipix1]) * 0.5 * (val0 - val1);
            if is_pixcentr0_inside_tri1 && is_pixcentr1_inside_tri0 {
                dbg!("todo");
                continue;
            } else {
                if is_pixcentr1_inside_tri0 {
                    // only tri1 recieve gradient
                    let b = fn_barycentric(&pixcntr1, itri1).unwrap();
                    let b = [b.0, b.1, b.2];
                    let itri1 = itri1 as usize;
                    use del_geo_core::mat4_col_major::Mat4ColMajor;
                    let dxyz = transform_pix2world.transform_direction(&[1., 0., 0.]);
                    for inode in 0..3 {
                        let ivtx = tri2vtx[itri1 * 3 + inode] as usize;
                        dldw_vtx2xyz[ivtx * 3] += b[inode] * dxyz[0] * dldpa;
                        dldw_vtx2xyz[ivtx * 3 + 1] += b[inode] * dxyz[1] * dldpa;
                        dldw_vtx2xyz[ivtx * 3 + 2] += b[inode] * dxyz[2] * dldpa;
                    }
                } else {
                    // only tri0 recieve gradient
                    let b = fn_barycentric(&pixcntr0, itri0).unwrap();
                    let b = [b.0, b.1, b.2];
                    let itri0 = itri0 as usize;
                    use del_geo_core::mat4_col_major::Mat4ColMajor;
                    let dxyz = transform_pix2world.transform_direction(&[1., 0., 0.]);
                    for inode in 0..3 {
                        let ivtx = tri2vtx[itri0 * 3 + inode] as usize;
                        dldw_vtx2xyz[ivtx * 3] += b[inode] * dxyz[0] * dldpa;
                        dldw_vtx2xyz[ivtx * 3 + 1] += b[inode] * dxyz[1] * dldpa;
                        dldw_vtx2xyz[ivtx * 3 + 2] += b[inode] * dxyz[2] * dldpa;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {

    const IMG_RES: usize = 128;

    enum RenderMode {
        Occlusion,
        Depth,
    }


    fn geometry(eps: f32) -> (Vec<u32>, Vec<f32>, [f32; 16], Vec<f32>) {
        let (tri2vtx, vtx2xyz) = crate::trimesh3_primitive::torus_zup::<u32, f32>(1.3, 0.4, 64, 32);
        let vtx2xyz = {
            let transform0 = del_geo_core::mat4_col_major::from_rot_x(1.15);
            //let transform0 = del_geo_core::mat4_col_major::from_rot_x(std::f32::consts::PI*0.25);
            //let transform1 = del_geo_core::mat4_col_major::from_translate(&[0.0, 0.6+eps, 0.0]);
            let transform1 = del_geo_core::mat4_col_major::from_translate(&[eps, 0.6, 0.0]);
            let transform =
                del_geo_core::mat4_col_major::mult_mat_col_major(&transform1, &transform0);
            crate::vtx2xyz::transform_homogeneous(&vtx2xyz, &transform)
        };
        let transform_world2ndc = del_geo_core::mat4_col_major::from_diagonal(0.5, 0.5, 0.5, 1.0);
        //let dxyz: Vec<f32> = (0..vtx2xyz.len()/3).flat_map(|_| [0., 1., 0.]).collect();
        let dxyz: Vec<f32> = (0..vtx2xyz.len() / 3).flat_map(|_| [1., 0., 0.]).collect();
        (tri2vtx, vtx2xyz, transform_world2ndc, dxyz)
    }

    fn sample(eps: f32, img_shape: (usize, usize), num_sample: usize, mode: &RenderMode) -> Vec<f32> {
        let (tri2vtx, vtx2xyz, transform_world2ndc, _dxyz) = geometry(eps);
        // let transform_ndc2world = del_geo_core::mat4_col_major::from_identity();
        let transform_ndc2world =
            del_geo_core::mat4_col_major::try_inverse_with_pivot(&transform_world2ndc).unwrap();
        let bvhnodes = crate::bvhnodes_morton::from_triangle_mesh(&tri2vtx, &vtx2xyz, 3);
        let bvhnode2aabb = crate::bvhnode2aabb3::from_uniform_mesh_with_bvh(
            0,
            &bvhnodes,
            &tri2vtx,
            3,
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
                if let Some((t, _i_tri)) = crate::search_bvh3::first_intersection_ray(
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
                    match mode {
                        RenderMode::Depth => {
                            sum += 1.0 - t;
                        }
                        RenderMode::Occlusion => { sum += 1.0; }
                    }
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

    fn test_sample_mode(mode: &RenderMode, str_mode: &str) {
        let num_sample = 2056;
        let img_shape = (IMG_RES, IMG_RES);
        let eps = 1.0e-1 / IMG_RES as f32;
        let pix2val0 = sample(-eps, img_shape, num_sample, mode);
        {
            let pix2val1 = sample(0., img_shape, num_sample, mode);
            del_canvas::write_png_from_float_image_grayscale(
                format!("../target/differentiable_rasterization_ren_finitesample_{}.png", str_mode),
                img_shape,
                &pix2val1,
            )
            .unwrap()
        }
        let pix2val2 = sample(eps, img_shape, num_sample, mode);
        let pix2rgb_diff: Vec<_> = pix2val0
            .iter()
            .zip(pix2val2.iter())
            .flat_map(|(&v0, &v1)| {
                let grad = (v1 - v0) / (2.0 * eps);
                let c = del_canvas::colormap::apply_colormap(
                    grad,
                    -(IMG_RES as f32)*0.5,
                    (IMG_RES as f32)*0.5,
                    del_canvas::colormap::COLORMAP_BWR,
                );
                c
            })
            .collect();
        del_canvas::write_png_from_float_image_rgb(
            format!("../target/differentiable_rasterization_diff_finitesample_{}.png", str_mode),
            &img_shape,
            &pix2rgb_diff,
        )
        .unwrap()
    }

    #[test]
    fn test_sample() {
        test_sample_mode(&RenderMode::Depth, "depth");
        test_sample_mode(&RenderMode::Occlusion, "occ");
    }

    fn nvdiffrast_occ(eps: f32) -> Vec<f32> {
        let img_shape = (IMG_RES, IMG_RES);
        let (tri2vtx, vtx2xyz, transform_world2ndc, dxyz) = geometry(eps);
        let transform_ndc2world =
            del_geo_core::mat4_col_major::try_inverse_with_pivot(&transform_world2ndc).unwrap();
        let transform_ndc2pix = del_geo_core::mat4_col_major::from_transform_ndc2pix(img_shape);
        let transform_world2pix = del_geo_core::mat4_col_major::mult_mat_col_major(
            &transform_ndc2pix,
            &transform_world2ndc,
        );
        //
        let num_vtx = vtx2xyz.len() / 3;
        let edge2vtx = crate::edge2vtx::from_triangle_mesh(&tri2vtx, num_vtx);
        let edge2tri = crate::edge2elem::from_edge2vtx_of_tri2vtx(&edge2vtx, &tri2vtx, num_vtx);
        //
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
                0,
                &bvhnodes,
                &tri2vtx,
                3,
                &vtx2xyz,
                None,
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
        let mut pic2val: Vec<f32> = pix2tri
            .iter()
            .map(|&v| if v == u32::MAX { 0. } else { 1. })
            .collect();
        crate::differential_rasterizer::antialias(
            &cedge2vtx,
            &vtx2xyz,
            &transform_world2pix,
            img_shape,
            &mut pic2val,
            &pix2tri,
        );
        pic2val
    }

    #[test]
    fn test_nvdiffrast() {
        let img_shape = (IMG_RES, IMG_RES);
        let (tri2vtx, vtx2xyz, transform_world2ndc, dxyz) = geometry(0.);
        let transform_ndc2world =
            del_geo_core::mat4_col_major::try_inverse_with_pivot(&transform_world2ndc).unwrap();
        let transform_ndc2pix = del_geo_core::mat4_col_major::from_transform_ndc2pix(img_shape);
        let transform_world2pix = del_geo_core::mat4_col_major::mult_mat_col_major(
            &transform_ndc2pix,
            &transform_world2ndc,
        );
        let cedge2vtx = {
            //
            let num_vtx = vtx2xyz.len() / 3;
            // let (_vtx2idx, _idx2vtx) = crate::vtx2vtx::from_uniform_mesh(&tri2vtx, 3, num_vtx, false);
            let edge2vtx = crate::edge2vtx::from_triangle_mesh(&tri2vtx, num_vtx);
            let edge2tri = crate::edge2elem::from_edge2vtx_of_tri2vtx(&edge2vtx, &tri2vtx, num_vtx);
            //
            crate::edge2vtx::contour_for_triangle_mesh::<u32>(
                &tri2vtx,
                &vtx2xyz,
                &transform_world2ndc,
                &edge2vtx,
                &edge2tri,
            )
        };
        {
            // draw edge image
            let mut pix2val_edge = vec![1f32; img_shape.0 * img_shape.1];
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
                    &mut pix2val_edge,
                    img_shape.0,
                    &xy0_img,
                    &xy1_img,
                    0.,
                );
            }
            del_canvas::write_png_from_float_image_grayscale(
                "../target/differentiable_rasterizer_edge.png",
                img_shape,
                &pix2val_edge,
            )
            .unwrap();
        }
        let pix2tri = {
            let bvhnodes = crate::bvhnodes_morton::from_triangle_mesh(&tri2vtx, &vtx2xyz, 3);
            let bvhnode2aabb = crate::bvhnode2aabb3::from_uniform_mesh_with_bvh(
                0,
                &bvhnodes,
                &tri2vtx,
                3,
                &vtx2xyz,
                None,
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
        {
            let mut img_data: Vec<f32> = pix2tri
                .iter()
                .map(|&v| if v == u32::MAX { 0. } else { 1. })
                .collect();
            crate::differential_rasterizer::antialias(
                &cedge2vtx,
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
        let mut pix2rgb_diff: Vec<_> = vec![0f32; img_shape.0 * img_shape.1 * 3];
        for i_pix in 0..IMG_RES * IMG_RES {
            let dldw_img = {
                let mut dldw_pix2val = vec![0f32; IMG_RES * IMG_RES];
                dldw_pix2val[i_pix] = 1.0;
                dldw_pix2val
            };
            let mut dldw_vtx2xyz = vec![0f32; vtx2xyz.len()];
            crate::differential_rasterizer::bwd_antialias(
                &cedge2vtx,
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
        let pix2val0 = nvdiffrast_occ(-eps);
        let pix2val2 = nvdiffrast_occ(eps);
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
                0,
                &bvhnodes,
                &tri2vtx,
                3,
                &vtx2xyz,
                None,
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
            crate::differential_rasterizer::bwd_microedge(
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
        del_canvas::write_png_from_float_image_rgb(
            "../target/silhouette_diff_microedge.png",
            &img_shape,
            &pix2rgb_diff,
        )
        .unwrap();
    }
}
