fn fn_barycentric(
    tri2vtx: &[u32],
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
        let i0 = tri2vtx[itri1 * 3] as usize;
        let i1 = tri2vtx[itri1 * 3 + 1] as usize;
        let i2 = tri2vtx[itri1 * 3 + 2] as usize;
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
    tri2vtx: &[u32],
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
                    let is_pixcentr0_inside_tri1 = fn_inside(fn_barycentric(
                        tri2vtx,
                        vtx2xyz,
                        &pixcntr0,
                        itri1,
                        transform_world2pix,
                    ));
                    let is_pixcentr1_inside_tri0 = fn_inside(fn_barycentric(
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
                    let is_pixcentr0_inside_tri1 = fn_inside(fn_barycentric(
                        tri2vtx,
                        vtx2xyz,
                        &pixcntr0,
                        itri1,
                        transform_world2pix,
                    ));
                    let is_pixcentr1_inside_tri0 = fn_inside(fn_barycentric(
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
    vtx2xy: &[f32],
    vtx2velo: &mut [f32],
) {
    assert_eq!(hedge2vy.len(), (img_h - 1) * img_w);
    assert_eq!(vedge2vx.len(), img_h * (img_w - 1));
    let num_vtx = vtx2xy.len() / 2;
    assert_eq!(vtx2velo.len(), num_vtx * 2);
    for i_vtx in 0..num_vtx {
        let px = vtx2xy[i_vtx * 2];
        let py = vtx2xy[i_vtx * 2 + 1];
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
            vtx2velo[i_vtx * 2] = (1. - tx) * (1. - ty) * vedge2vx[iy0 * w + ix0]
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
            vtx2velo[i_vtx * 2 + 1] = (1. - tx) * (1. - ty) * hedge2vy[iy0 * img_w + ix0]
                + tx * (1. - ty) * hedge2vy[iy0 * img_w + ix1]
                + (1. - tx) * ty * hedge2vy[iy1 * img_w + ix0]
                + tx * ty * hedge2vy[iy1 * img_w + ix1];
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn bwd(
    tri2vtx: &[u32],
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
            let is_pixcentr0_inside_tri1 = fn_inside(fn_barycentric(
                tri2vtx,
                vtx2xyz,
                &pixcntr0,
                itri1,
                transform_world2pix,
            ));
            let is_pixcentr1_inside_tri0 = fn_inside(fn_barycentric(
                tri2vtx,
                vtx2xyz,
                &pixcntr1,
                itri0,
                transform_world2pix,
            ));
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
                dbg!("todo");
                continue;
            } else if is_pixcentr1_inside_tri0 {
                // only tri1 receive gradient
                let b = fn_barycentric(tri2vtx, vtx2xyz, &pixcntr1, itri1, transform_world2pix)
                    .unwrap();
                let itri1 = itri1 as usize;
                let xyz = crate::trimesh3::to_tri3(tri2vtx.as_chunks::<3>().0, vtx2xyz, itri1)
                    .position_from_barycentric_coordinates(b[0], b[1]);
                let dpixdxyz =
                    del_geo_core::mat4_col_major::jacobian_transform(transform_world2pix, &xyz);
                let dldw_pix = [0., dldpa, 0.];
                let dldw_xyz = del_geo_core::vec3::mult_mat3_col_major(&dldw_pix, &dpixdxyz);
                for inode in 0..3 {
                    let ivtx = tri2vtx[itri1 * 3 + inode] as usize;
                    dldw_vtx2xyz[ivtx][0] += b[inode] * dldw_xyz[0];
                    dldw_vtx2xyz[ivtx][1] += b[inode] * dldw_xyz[1];
                    dldw_vtx2xyz[ivtx][2] += b[inode] * dldw_xyz[2];
                }
            } else {
                // only tri0 recieve gradient
                let b = fn_barycentric(tri2vtx, vtx2xyz, &pixcntr0, itri0, transform_world2pix)
                    .unwrap();
                let itri0 = itri0 as usize;
                let xyz = crate::trimesh3::to_tri3(tri2vtx.as_chunks::<3>().0, vtx2xyz, itri0)
                    .position_from_barycentric_coordinates(b[0], b[1]);
                let dpixdxyz =
                    del_geo_core::mat4_col_major::jacobian_transform(transform_world2pix, &xyz);
                let dldw_pix = [0., dldpa, 0.];
                let dldw_xyz = del_geo_core::vec3::mult_mat3_col_major(&dldw_pix, &dpixdxyz);
                for inode in 0..3 {
                    let ivtx = tri2vtx[itri0 * 3 + inode] as usize;
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
            let is_pixcentr0_inside_tri1 = fn_inside(fn_barycentric(
                tri2vtx,
                vtx2xyz,
                &pixcntr0,
                itri1,
                transform_world2pix,
            ));
            let is_pixcentr1_inside_tri0 = fn_inside(fn_barycentric(
                tri2vtx,
                vtx2xyz,
                &pixcntr1,
                itri0,
                transform_world2pix,
            ));
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
                dbg!("todo");
                continue;
            } else if is_pixcentr1_inside_tri0 {
                // only tri1 recieve gradient
                let b = fn_barycentric(tri2vtx, vtx2xyz, &pixcntr1, itri1, transform_world2pix)
                    .unwrap();
                let itri1 = itri1 as usize;
                let xyz = crate::trimesh3::to_tri3(tri2vtx.as_chunks::<3>().0, vtx2xyz, itri1)
                    .position_from_barycentric_coordinates(b[0], b[1]);
                let dpixdxyz =
                    del_geo_core::mat4_col_major::jacobian_transform(transform_world2pix, &xyz);
                let dldw_pix = [dldpa, 0., 0.];
                let dldw_xyz = del_geo_core::vec3::mult_mat3_col_major(&dldw_pix, &dpixdxyz);
                for inode in 0..3 {
                    let ivtx = tri2vtx[itri1 * 3 + inode] as usize;
                    dldw_vtx2xyz[ivtx][0] += b[inode] * dldw_xyz[0];
                    dldw_vtx2xyz[ivtx][1] += b[inode] * dldw_xyz[1];
                    dldw_vtx2xyz[ivtx][2] += b[inode] * dldw_xyz[2];
                }
            } else {
                // only tri0 recieve gradient
                let b = fn_barycentric(tri2vtx, vtx2xyz, &pixcntr0, itri0, transform_world2pix)
                    .unwrap();
                let itri0 = itri0 as usize;
                let xyz = crate::trimesh3::to_tri3(tri2vtx.as_chunks::<3>().0, vtx2xyz, itri0)
                    .position_from_barycentric_coordinates(b[0], b[1]);
                let dpixdxyz =
                    del_geo_core::mat4_col_major::jacobian_transform(transform_world2pix, &xyz);
                let dldw_pix = [dldpa, 0., 0.];
                let dldw_xyz = del_geo_core::vec3::mult_mat3_col_major(&dldw_pix, &dpixdxyz);
                for inode in 0..3 {
                    let ivtx = tri2vtx[itri0 * 3 + inode] as usize;
                    dldw_vtx2xyz[ivtx][0] += b[inode] * dldw_xyz[0];
                    dldw_vtx2xyz[ivtx][1] += b[inode] * dldw_xyz[1];
                    dldw_vtx2xyz[ivtx][2] += b[inode] * dldw_xyz[2];
                }
            }
        }
    }
}
