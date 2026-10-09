fn fn_barycentric_pix(
    tri2vtx: &[[u32; 3]],
    vtx2xyz: &[[f32; 3]],
    pixcntr0: &[f32; 2],
    itri1: u32,
    transform_world2pix: &[f32; 16],
) -> Option<[f32; 3]> {
    if itri1 == u32::MAX {
        None
    } else {
        use del_geo_core::mat4_col_major::Mat4ColMajor;
        use del_geo_core::vec3::Vec3;
        let itri1 = itri1 as usize;
        let i0 = tri2vtx[itri1][0] as usize;
        let i1 = tri2vtx[itri1][1] as usize;
        let i2 = tri2vtx[itri1][2] as usize;
        let xyz0 = &vtx2xyz[i0];
        let xyz1 = &vtx2xyz[i1];
        let xyz2 = &vtx2xyz[i2];
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
        let b = del_geo_core::tri2::barycentric_coords(&p0, &p1, &p2, pixcntr0)?;
        Some([b.0, b.1, b.2])
    }
}

#[allow(clippy::too_many_arguments)]
fn scatter_intersection_gradient(
    node2vtx0: &[u32; 3],
    node2vtx1: &[u32; 3],
    dldw_vtx2xyz: &mut [[f32; 3]],
    transform_world2pix: &[f32; 16],
    tri0: &Tri,
    tri1: &Tri,
    pixcntr0: &[f32; 2],
    pixcntr1: &[f32; 2],
    axis: usize, // 0: horizontal movement, 1: vertical movement
    dldpa: f32,
) {
    assert!(axis < 2);

    let n0 = tri0.unormal_pix();
    let n1 = tri1.unormal_pix();

    let denom = n0[axis] * n1[2] - n0[2] * n1[axis];

    // 断面の2直線がほぼ平行なら、微分が発散するため今回はスキップ。
    let norm0 = n0[axis].hypot(n0[2]);
    let norm1 = n1[axis].hypot(n1[2]);
    let scale = norm0 * norm1;
    if !denom.is_finite() || scale == 0.0 || denom.abs() <= 1.0e-6 * scale {
        return;
    }

    let b0 = tri0.barycentric_world(pixcntr0).unwrap();
    let b1 = tri1.barycentric_world(pixcntr1).unwrap();

    let xyz0 = tri0.world_pos_from_barycentric_coord(&b0);
    let xyz1 = tri0.world_pos_from_barycentric_coord(&b1);

    let contributions = [
        (node2vtx0, xyz0, b0, n0, n1[2] / denom),
        (node2vtx1, xyz1, b1, n1, -n0[2] / denom),
    ];

    for (node2vtx, xyz, b, n, coefficient) in contributions {
        // 既存の overhang 処理と同様に、
        // 自分のピクセル中心に対応する fragment に散布する。

        let g_pix = n.map(|x| dldpa * coefficient * x);

        let j = del_geo_core::mat4_col_major::jacobian_transform(transform_world2pix, &xyz);

        // g_world = J^T g_pix
        // j[row + 3 * col] = ∂pix[row] / ∂world[col]
        let g_world: [f32; 3] = std::array::from_fn(|k| {
            j[3 * k] * g_pix[0] + j[3 * k + 1] * g_pix[1] + j[3 * k + 2] * g_pix[2]
        });

        for inode in 0..3 {
            let ivtx = node2vtx[inode] as usize;
            for k in 0..3 {
                dldw_vtx2xyz[ivtx][k] += b[inode] * g_world[k];
            }
        }
    }
}

fn fn_inside(b: Option<[f32; 3]>) -> bool {
    if let Some(b0) = b {
        if (b0[0] >= 0. && b0[1] >= 0. && b0[2] >= 0.)
            || (b0[0] <= 0. && b0[1] <= 0. && b0[2] <= 0.)
        {
            return true;
        }
        false
    } else {
        true
    }
}

#[allow(clippy::too_many_arguments)]
pub fn edge_gradient_and_type(
    tri2vtx: &[[u32; 3]],
    vtx2xyz: &[[f32; 3]],
    transform_world2pix: &[f32; 16],
    (img_w, img_h): (usize, usize),
    pix2tri: &[u32],
    num_vdim: usize,
    pix2val: &[f32],
    dldw_pix2val: &[f32],
    hedge2type: &mut [u8],
    hedge2dldr: &mut [f32],
    vedge2type: &mut [u8],
    vedge2dldr: &mut [f32],
) {
    let num_pix = img_h * img_w;
    assert_eq!(pix2tri.len(), num_pix);
    assert_eq!(pix2val.len(), num_pix * num_vdim);
    assert_eq!(dldw_pix2val.len(), pix2val.len());
    // -----------------------
    // horizontal edge
    assert_eq!(hedge2type.len(), (img_h - 1) * img_w);
    assert_eq!(hedge2type.len(), hedge2dldr.len());
    for iw in 0..img_w {
        for ih0 in 0..img_h - 1 {
            let ih1 = ih0 + 1;
            let ipix0 = ih0 * img_w + iw; // north
            let ipix1 = ih1 * img_w + iw; // south
            let i_hedge = ih0 * img_w + iw;
            {
                let itri0 = pix2tri[ipix0];
                let itri1 = pix2tri[ipix1];
                hedge2type[i_hedge] = if itri0 == itri1 {
                    // same tri/background
                    0
                }
                // no edge
                else {
                    let pixcntr0 = [iw as f32 + 0.5, ih0 as f32 + 0.5];
                    let pixcntr1 = [iw as f32 + 0.5, ih1 as f32 + 0.5];
                    let is_pixcentr0_inside_tri1 = fn_inside(fn_barycentric_pix(
                        tri2vtx,
                        vtx2xyz,
                        &pixcntr0,
                        itri1,
                        transform_world2pix,
                    ));
                    let is_pixcentr1_inside_tri0 = fn_inside(fn_barycentric_pix(
                        tri2vtx,
                        vtx2xyz,
                        &pixcntr1,
                        itri0,
                        transform_world2pix,
                    ));
                    match (is_pixcentr0_inside_tri1, is_pixcentr1_inside_tri0) {
                        (false, false) => 0, // shared edge
                        (true, false) => 1, // tri0 is in front of tri1 (only tri0 receive gradient)
                        (false, true) => 1, // tri1 is in front of tri0 (only tri1 receive gradient)
                        (true, true) => 1,  // intersection
                    }
                };
            }
            hedge2dldr[i_hedge] = 0.0;
            for i_vdim in 0..num_vdim {
                let val0 = pix2val[ipix0 * num_vdim + i_vdim];
                let val1 = pix2val[ipix1 * num_vdim + i_vdim];
                let dval0 = dldw_pix2val[ipix0 * num_vdim + i_vdim];
                let dval1 = dldw_pix2val[ipix1 * num_vdim + i_vdim];
                hedge2dldr[i_hedge] += (dval0 + dval1) * 0.5 * (val0 - val1);
            }
        }
    }

    // --------------------------
    // vertical edge
    assert_eq!(vedge2type.len(), img_h * (img_w - 1));
    assert_eq!(vedge2type.len(), vedge2dldr.len());
    for iw0 in 0..img_w - 1 {
        for ih in 0..img_h {
            let iw1 = iw0 + 1;
            let ipix0 = ih * img_w + iw0;
            let ipix1 = ih * img_w + iw1;
            let i_vedge = ih * (img_w - 1) + iw0;
            {
                let itri0 = pix2tri[ipix0];
                let itri1 = pix2tri[ipix1];
                vedge2type[i_vedge] = if itri0 == itri1 {
                    // same tri/background
                    0
                } else {
                    let pixcntr0 = [iw0 as f32 + 0.5, ih as f32 + 0.5];
                    let pixcntr1 = [iw1 as f32 + 0.5, ih as f32 + 0.5];
                    let is_pixcentr0_inside_tri1 = fn_inside(fn_barycentric_pix(
                        tri2vtx,
                        vtx2xyz,
                        &pixcntr0,
                        itri1,
                        transform_world2pix,
                    ));
                    let is_pixcentr1_inside_tri0 = fn_inside(fn_barycentric_pix(
                        tri2vtx,
                        vtx2xyz,
                        &pixcntr1,
                        itri0,
                        transform_world2pix,
                    ));
                    match (is_pixcentr0_inside_tri1, is_pixcentr1_inside_tri0) {
                        (false, false) => 0, // shared edge
                        (true, false) => 1, // tri0 is in front of tri1 (only tri0 receive gradient)
                        (false, true) => 1, // tri1 is in front of tri0 (only tri1 receive gradient)
                        (true, true) => 1,  // intersection
                    }
                };
            }
            vedge2dldr[i_vedge] = 0.0;
            for i_vdim in 0..num_vdim {
                let val0 = pix2val[ipix0 * num_vdim + i_vdim];
                let val1 = pix2val[ipix1 * num_vdim + i_vdim];
                let dval0 = dldw_pix2val[ipix0 * num_vdim + i_vdim];
                let dval1 = dldw_pix2val[ipix1 * num_vdim + i_vdim];
                vedge2dldr[i_vedge] += (dval0 + dval1) * 0.5 * (val0 - val1);
            }
        }
    }
}

pub fn interpolate_from_edges(
    (img_w, img_h): (usize, usize),
    hedge2vy: &[f32],
    vedge2vx: &[f32],
    vtx2xy: &[[f32; 2]],
    vtx2velo: &mut [[f32; 2]],
) {
    assert_eq!(hedge2vy.len(), (img_h - 1) * img_w);
    assert_eq!(vedge2vx.len(), img_h * (img_w - 1));
    let num_vtx = vtx2xy.len();
    assert_eq!(vtx2velo.len(), num_vtx);
    for i_vtx in 0..num_vtx {
        let px = vtx2xy[i_vtx][0];
        let py = vtx2xy[i_vtx][1];
        // x-velocity: bilinear from vertical edges at (iw0+1.0, ih+0.5)
        {
            let gx = px - 1.0_f32;
            let gy = py - 0.5_f32;
            let ix0 = (gx.floor() as i32).clamp(0, img_w as i32 - 2) as usize;
            let iy0 = (gy.floor() as i32).clamp(0, img_h as i32 - 2) as usize;
            let ix1 = (ix0 + 1).min(img_w - 2);
            let iy1 = (iy0 + 1).min(img_h - 1);
            let tx = (gx - ix0 as f32).clamp(0., 1.);
            let ty = (gy - iy0 as f32).clamp(0., 1.);
            let w = img_w - 1;
            vtx2velo[i_vtx][0] = (1. - tx) * (1. - ty) * vedge2vx[iy0 * w + ix0]
                + tx * (1. - ty) * vedge2vx[iy0 * w + ix1]
                + (1. - tx) * ty * vedge2vx[iy1 * w + ix0]
                + tx * ty * vedge2vx[iy1 * w + ix1];
        }
        // y-velocity: bilinear from horizontal edges at (iw+0.5, ih0+1.0)
        {
            let gx = px - 0.5_f32;
            let gy = py - 1.0_f32;
            let ix0 = (gx.floor() as i32).clamp(0, img_w as i32 - 2) as usize;
            let iy0 = (gy.floor() as i32).clamp(0, img_h as i32 - 2) as usize;
            let ix1 = (ix0 + 1).min(img_w - 1);
            let iy1 = (iy0 + 1).min(img_h - 2);
            let tx = (gx - ix0 as f32).clamp(0., 1.);
            let ty = (gy - iy0 as f32).clamp(0., 1.);
            vtx2velo[i_vtx][1] = (1. - tx) * (1. - ty) * hedge2vy[iy0 * img_w + ix0]
                + tx * (1. - ty) * hedge2vy[iy0 * img_w + ix1]
                + (1. - tx) * ty * hedge2vy[iy1 * img_w + ix0]
                + tx * ty * hedge2vy[iy1 * img_w + ix1];
        }
    }
}

struct Tri {
    node2xyz: [[f32; 3]; 3],
    node2pixxy: [[f32; 2]; 3],
    node2w: [f32; 3],
    node2pixz: [f32; 3],
}

impl Tri {
    fn new(
        i_tri: u32,
        tri2vtx: &[[u32; 3]],
        vtx2xyz: &[[f32; 3]],
        transform_world2pix: &[f32; 16],
    ) -> Option<Self> {
        if i_tri == u32::MAX {
            return None;
        }
        let node2xyz: [[f32; 3]; 3] =
            core::array::from_fn(|i| vtx2xyz[tri2vtx[i_tri as usize][i] as usize]);
        let mut node2pixxy = [[0f32; 2]; 3];
        let mut node2pixz = [0f32; 3];
        let mut node2w = [0f32; 3];
        for i_node in 0..3 {
            let xyz = &node2xyz[i_node];
            let pixh = del_geo_core::mat4_col_major::mult_vec(
                transform_world2pix,
                &[xyz[0], xyz[1], xyz[2], 1.],
            );
            node2pixxy[i_node][0] = pixh[0] / pixh[3];
            node2pixxy[i_node][1] = pixh[1] / pixh[3];
            node2pixz[i_node] = pixh[2] / pixh[3];
            node2w[i_node] = pixh[3];
        }
        Some(Self {
            node2xyz,
            node2pixxy,
            node2pixz,
            node2w,
        })
    }

    fn barycentric_pix(&self, pix: &[f32; 2]) -> Option<[f32; 3]> {
        let b = del_geo_core::tri2::barycentric_coords(
            &self.node2pixxy[0],
            &self.node2pixxy[1],
            &self.node2pixxy[2],
            pix,
        )?;
        Some([b.0, b.1, b.2])
    }
    fn is_inside(&self, pix: &[f32; 2]) -> bool {
        self.barycentric_pix(pix)
            .is_some_and(|b| b.iter().all(|&v| v >= 0.0))
    }

    fn barycentric_world(&self, pix: &[f32; 2]) -> Option<[f32; 3]> {
        let b = self.barycentric_pix(pix)?;
        let q: [f32; 3] = core::array::from_fn(|i| b[i] / self.node2w[i]);

        let sum = q.iter().sum::<f32>();
        if !sum.is_finite() || sum == 0.0 {
            return None;
        }

        let b = q.map(|v| v / sum);
        b.iter().all(|v| v.is_finite()).then_some(b)
    }

    fn world_pos_from_barycentric_coord(&self, bc: &[f32; 3]) -> [f32; 3] {
        del_geo_core::tri3::position_from_barycentric_coords(
            &self.node2xyz[0],
            &self.node2xyz[1],
            &self.node2xyz[2],
            bc,
        )
    }

    fn unormal_pix(&self) -> [f32; 3] {
        let p0 = [
            self.node2pixxy[0][0],
            self.node2pixxy[0][1],
            self.node2pixz[0],
        ];
        let p1 = [
            self.node2pixxy[1][0],
            self.node2pixxy[1][1],
            self.node2pixz[1],
        ];
        let p2 = [
            self.node2pixxy[2][0],
            self.node2pixxy[2][1],
            self.node2pixz[2],
        ];
        let (un, _area) = del_geo_core::tri3::unit_normal_area(&p0, &p1, &p2);
        un
    }
}

#[allow(clippy::too_many_arguments)]
pub fn bwd(
    tri2vtx: &[[u32; 3]],
    vtx2xyz: &[[f32; 3]],
    dldw_vtx2xyz: &mut [[f32; 3]],
    transform_world2pix: &[f32; 16],
    (img_w, img_h): (usize, usize),
    pix2tri: &[u32],
    num_vdim: usize,
    pix2val: &[f32],
    dldw_pix2val: &[f32],
) {
    assert_eq!(pix2val.len(), img_h * img_w * num_vdim);
    assert_eq!(dldw_pix2val.len(), img_h * img_w * num_vdim);
    assert_eq!(vtx2xyz.len(), dldw_vtx2xyz.len());
    // horizontal edge (vertical movement)
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
            let tri0 = Tri::new(itri0, tri2vtx, vtx2xyz, transform_world2pix);
            let tri1 = Tri::new(itri1, tri2vtx, vtx2xyz, transform_world2pix);
            let is_pixcentr0_inside_tri1 = if let Some(ref tri1) = tri1 {
                tri1.is_inside(&pixcntr0)
            } else {
                true
            };
            let is_pixcentr1_inside_tri0 = if let Some(ref tri0) = tri0 {
                tri0.is_inside(&pixcntr1)
            } else {
                true
            };
            if !is_pixcentr0_inside_tri1 && !is_pixcentr1_inside_tri0 {
                continue;
            }
            let dldpa = {
                let mut dldpa = 0.0;
                for i_vdim in 0..num_vdim {
                    let val0 = pix2val[ipix0 * num_vdim + i_vdim];
                    let val1 = pix2val[ipix1 * num_vdim + i_vdim];
                    let dval0 = dldw_pix2val[ipix0 * num_vdim + i_vdim];
                    let dval1 = dldw_pix2val[ipix1 * num_vdim + i_vdim];
                    dldpa += (dval0 + dval1) * 0.5 * (val0 - val1);
                }
                dldpa
            };
            if is_pixcentr0_inside_tri1 && is_pixcentr1_inside_tri0 {
                if itri0 == u32::MAX || itri1 == u32::MAX {
                    continue;
                }
                scatter_intersection_gradient(
                    &tri2vtx[itri0 as usize],
                    &tri2vtx[itri1 as usize],
                    dldw_vtx2xyz,
                    transform_world2pix,
                    &tri0.unwrap(),
                    &tri1.unwrap(),
                    &pixcntr0,
                    &pixcntr1,
                    1,
                    dldpa,
                );
                continue;
            } else if is_pixcentr1_inside_tri0 {
                // only tri1 receive gradient
                let tri1 = tri1.unwrap();
                let b = tri1.barycentric_world(&pixcntr1).unwrap();
                let xyz = tri1.world_pos_from_barycentric_coord(&b);
                let dpixdxyz =
                    del_geo_core::mat4_col_major::jacobian_transform(transform_world2pix, &xyz);
                let dldw_pix = [0., dldpa, 0.];
                let dldw_xyz = del_geo_core::vec3::mult_mat3_col_major(&dldw_pix, &dpixdxyz);
                for inode in 0..3 {
                    let ivtx = tri2vtx[itri1 as usize][inode] as usize;
                    dldw_vtx2xyz[ivtx][0] += b[inode] * dldw_xyz[0];
                    dldw_vtx2xyz[ivtx][1] += b[inode] * dldw_xyz[1];
                    dldw_vtx2xyz[ivtx][2] += b[inode] * dldw_xyz[2];
                }
            } else {
                // only tri0 recieve gradient
                let tri0 = tri0.unwrap();
                let b = tri0.barycentric_world(&pixcntr0).unwrap();
                let xyz = tri0.world_pos_from_barycentric_coord(&b);
                let dpixdxyz =
                    del_geo_core::mat4_col_major::jacobian_transform(transform_world2pix, &xyz);
                let dldw_pix = [0., dldpa, 0.];
                let dldw_xyz = del_geo_core::vec3::mult_mat3_col_major(&dldw_pix, &dpixdxyz);
                for inode in 0..3 {
                    let ivtx = tri2vtx[itri0 as usize][inode] as usize;
                    dldw_vtx2xyz[ivtx][0] += b[inode] * dldw_xyz[0];
                    dldw_vtx2xyz[ivtx][1] += b[inode] * dldw_xyz[1];
                    dldw_vtx2xyz[ivtx][2] += b[inode] * dldw_xyz[2];
                }
            }
        }
    }

    // vertical edge (horizontal movement)
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
            let tri0 = Tri::new(itri0, tri2vtx, vtx2xyz, transform_world2pix);
            let tri1 = Tri::new(itri1, tri2vtx, vtx2xyz, transform_world2pix);
            let is_pixcentr0_inside_tri1 = if let Some(ref tri1) = tri1 {
                tri1.is_inside(&pixcntr0)
            } else {
                true
            };
            let is_pixcentr1_inside_tri0 = if let Some(ref tri0) = tri0 {
                tri0.is_inside(&pixcntr1)
            } else {
                true
            };
            if !is_pixcentr0_inside_tri1 && !is_pixcentr1_inside_tri0 {
                continue;
            }
            let dldpa = {
                let mut dldpa = 0.0;
                for i_vdim in 0..num_vdim {
                    let val0 = pix2val[ipix0 * num_vdim + i_vdim];
                    let val1 = pix2val[ipix1 * num_vdim + i_vdim];
                    let dval0 = dldw_pix2val[ipix0 * num_vdim + i_vdim];
                    let dval1 = dldw_pix2val[ipix1 * num_vdim + i_vdim];
                    dldpa += (dval0 + dval1) * 0.5 * (val0 - val1);
                }
                dldpa
            };
            if is_pixcentr0_inside_tri1 && is_pixcentr1_inside_tri0 {
                if itri0 == u32::MAX || itri1 == u32::MAX {
                    continue;
                }
                scatter_intersection_gradient(
                    &tri2vtx[itri0 as usize],
                    &tri2vtx[itri1 as usize],
                    dldw_vtx2xyz,
                    transform_world2pix,
                    &tri0.unwrap(),
                    &tri1.unwrap(),
                    &pixcntr0,
                    &pixcntr1,
                    0,
                    dldpa,
                );
                continue;
            } else if is_pixcentr1_inside_tri0 {
                // only tri1 recieve gradient
                let tri1 = tri1.unwrap();
                let b = tri1.barycentric_world(&pixcntr1).unwrap();
                let xyz = tri1.world_pos_from_barycentric_coord(&b);
                let dpixdxyz =
                    del_geo_core::mat4_col_major::jacobian_transform(transform_world2pix, &xyz);
                let dldw_pix = [dldpa, 0., 0.];
                let dldw_xyz = del_geo_core::vec3::mult_mat3_col_major(&dldw_pix, &dpixdxyz);
                for inode in 0..3 {
                    let ivtx = tri2vtx[itri1 as usize][inode] as usize;
                    dldw_vtx2xyz[ivtx][0] += b[inode] * dldw_xyz[0];
                    dldw_vtx2xyz[ivtx][1] += b[inode] * dldw_xyz[1];
                    dldw_vtx2xyz[ivtx][2] += b[inode] * dldw_xyz[2];
                }
            } else {
                // only tri0 receive gradient
                let tri0 = tri0.unwrap();
                let b = tri0.barycentric_world(&pixcntr0).unwrap();
                let xyz = tri0.world_pos_from_barycentric_coord(&b);
                let dpixdxyz =
                    del_geo_core::mat4_col_major::jacobian_transform(transform_world2pix, &xyz);
                let dldw_pix = [dldpa, 0., 0.];
                let dldw_xyz = del_geo_core::vec3::mult_mat3_col_major(&dldw_pix, &dpixdxyz);
                for inode in 0..3 {
                    let ivtx = tri2vtx[itri0 as usize][inode] as usize;
                    dldw_vtx2xyz[ivtx][0] += b[inode] * dldw_xyz[0];
                    dldw_vtx2xyz[ivtx][1] += b[inode] * dldw_xyz[1];
                    dldw_vtx2xyz[ivtx][2] += b[inode] * dldw_xyz[2];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::trimesh3_raycast::ScalarRender;

    fn finite_difference<T>(
        tri2vtx: &[[u32; 3]],
        vtx2xyz: &[[f32; 3]],
        vtx2uvw: &[[f32; 3]],
        transform_world2ndc: &[f32; 16],
        img_shape: (usize, usize),
        _mode: &T,
        num_sample: usize,
        eps: f32,
    ) where
        T: ScalarRender<f32> + Sync,
    {
        // finite difference occlusion
        type Sampler = crate::trimesh3_raycast::BoxPixelSampler<rand_chacha::ChaChaRng>;
        let (pix2diff, pix2val0) = crate::trimesh3_raycast::finite_difference::<TriVal, Sampler>(
            &tri2vtx,
            &vtx2xyz,
            &vtx2uvw,
            &transform_world2ndc,
            img_shape,
            &TriVal,
            num_sample,
            eps,
        );
        let pix2rgb = pix2diff
            .iter()
            .map(|&d| {
                del_canvas::colormap::apply_colormap(
                    d,
                    -(img_shape.0 as f32),
                    img_shape.0 as f32,
                    del_canvas::colormap::COLORMAP_BWR,
                )
            })
            .collect::<Vec<_>>();
        let path_dir = std::path::Path::new("../target/out_del_msh_cpu");
        del_canvas::write_png_from_float_image(
            path_dir.join(format!("edgegrad_pix2val_fd.png")),
            img_shape,
            3,
            pix2rgb.as_flattened(),
        )
        .unwrap();
        del_canvas::write_png_from_float_image(
            path_dir.join(format!("edgegrad_pix2val.png")),
            img_shape,
            1,
            &pix2val0,
        )
        .unwrap();
    }

    struct TriVal;
    impl<T> ScalarRender<T> for TriVal
    where
        T: num_traits::Float,
    {
        fn fwd(&self, _: &[T; 3], i_tri: u32, _: &[[u32; 3]], _: &[[T; 3]], _: &[T; 16]) -> T {
            let one = T::one();
            if i_tri % 2 == 0 {
                one
            } else {
                one / (one + one)
            }
        }

        fn bwd(
            &self,
            _: T,
            _: &[T; 3],
            _: &[T; 3],
            _: &[T; 3],
            _: &[T; 3],
            _: &[T; 3],
            _: &[T; 16],
        ) -> ([T; 3], [T; 3], [T; 3]) {
            let zero = T::zero();
            ([zero; 3], [zero; 3], [zero; 3])
        }
    }

    #[test]
    fn test_hoge() {
        let tri2vtx = [[0, 1, 2], [3, 4, 5]];
        let vtx2xyz = [
            [-0.4, 0.1, 0.5],
            [0.6, -0.5, 0.1],
            [0.5, 0.5, 0.0],
            [0.35, -0.1, 0.5],
            [-0.4, 0.5, -0.1],
            [-0.3, -0.5, 0.0],
        ];
        let vtx2uvw = [
            [1., 0., 1.],
            [1., 0., 1.],
            [1., 0., 1.],
            [-1., 0., -1.],
            [-1., 0., -1.],
            [-1., 0., -1.],
        ];

        let img_shape = (100, 75);
        let transform_world2ndc = {
            let prj = del_geo_core::mat4_col_major::camera_perspective_blender(
                img_shape.0 as f32 / img_shape.1 as f32,
                35.0,
                1.0,
                3.0,
                true,
            );
            let ext = del_geo_core::mat4_col_major::from_translate(&[0.0, 0.0, -1.6]);
            del_geo_core::mat4_col_major::mult_mat_col_major(&prj, &ext)
        };
        let transform_ndc2world =
            del_geo_core::mat4_col_major::try_inverse(&transform_world2ndc).unwrap();
        let transform_ndc2pix = del_geo_core::mat4_col_major::from_transform_ndc2pix(img_shape);
        let transform_world2pix = del_geo_core::mat4_col_major::mult_mat_col_major(
            &transform_ndc2pix,
            &transform_world2ndc,
        );

        let _path_dir = std::path::Path::new("../target/out_del_msh_cpu");

        finite_difference(
            &tri2vtx,
            &vtx2xyz,
            &vtx2uvw,
            &transform_world2ndc,
            img_shape,
            &TriVal,
            300,
            1.0e-3,
        );

        {
            let bvhnodes = crate::bvhnodes_morton::from_triangle_mesh(&tri2vtx, &vtx2xyz);
            let bvhnode2aabb = crate::bvhnode2aabb3::from_uniform_mesh_with_bvh(
                0, &bvhnodes, &tri2vtx, &vtx2xyz, None,
            );
            let mut pix2tri = vec![u32::MAX; img_shape.0 * img_shape.1];
            crate::pix2tri::pix2tri_by_raycast(
                &mut pix2tri,
                &tri2vtx,
                &vtx2xyz,
                &bvhnodes,
                &bvhnode2aabb,
                img_shape,
                &transform_ndc2world,
            );
            let pix2val = crate::trimesh3_raycast::fwd_continuous(
                &pix2tri,
                img_shape,
                &tri2vtx,
                &vtx2xyz,
                &transform_ndc2world,
                &TriVal,
            );
            let mut pix2dval_edgegrad = vec![0f32; img_shape.0 * img_shape.1];
            for i_pix in 0..img_shape.0 * img_shape.1 {
                let pix2trg = {
                    let mut pix2trg = vec![0f32; img_shape.0 * img_shape.1];
                    pix2trg[i_pix] = 1.0;
                    pix2trg
                };
                let mut dldw_vtx2xyz = vec![[0f32; 3]; vtx2xyz.len()];
                crate::edgegrad::bwd(
                    &tri2vtx,
                    &vtx2xyz,
                    &mut dldw_vtx2xyz,
                    &transform_world2pix,
                    img_shape,
                    &pix2tri,
                    1,
                    &pix2val,
                    &pix2trg,
                );
                let grad: f32 = vtx2uvw
                    .iter()
                    .zip(dldw_vtx2xyz.iter())
                    .map(|(v0, v1)| del_geo_core::vec3::dot(v0, v1))
                    .sum();
                pix2dval_edgegrad[i_pix] = grad;
            }
            let pix2rgb = pix2dval_edgegrad
                .iter()
                .map(|&d| {
                    del_canvas::colormap::apply_colormap(
                        d,
                        -(img_shape.0 as f32),
                        img_shape.0 as f32,
                        del_canvas::colormap::COLORMAP_BWR,
                    )
                })
                .collect::<Vec<_>>();
            let path_dir = std::path::Path::new("../target/out_del_msh_cpu");
            del_canvas::write_png_from_float_image(
                path_dir.join(format!("edgegrad_pix2dval_edgegrad.png")),
                img_shape,
                3,
                pix2rgb.as_flattened(),
            )
            .unwrap()
        }
    }
}
