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
pub fn update_image(
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

pub fn backward_wrt_vtx2xyz(
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
