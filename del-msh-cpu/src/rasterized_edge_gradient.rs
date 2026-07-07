fn fn_barycentric(
    tri2vtx: &[u32],
    vtx2xyz: &[f32],
    pixcntr0: &[f32; 2],
    itri1: u32,
    transform_world2pix: &[f32; 16],
) -> Option<[f32; 3]>
{
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
        let Some(b) = del_geo_core::tri2::barycentric_coords(&p0, &p1, &p2, pixcntr0) else {
            return None
        };
        Some([b.0, b.1, b.2])
    }
}

fn fn_inside(b: Option<[f32;3]>) -> bool {
    if let Some(b0) = b {
        if (b0[0] >= 0. && b0[1] >= 0. && b0[2] >= 0.) || (b0[0] <= 0. && b0[1] <= 0. && b0[2] <= 0.)
        {
            return true;
        }
        false
    } else {
        true
    }
}

pub fn edge_gradient_and_type(
    tri2vtx: &[u32],
    vtx2xyz: &[f32],
    transform_world2pix: &[f32; 16],
    (img_w, img_h): (usize, usize),
    dldw_pixval: &[f32],
    pix2tri: &[u32],
    hedge2type: &mut [u8],
    hedge2dldr: &mut [f32],
    vedge2type: &mut [u8],
    vedge2dldr: &mut [f32],
)
{
    // -----------------------
    // horizontal edge
    assert_eq!(hedge2type.len(), (img_h-1)*img_w );
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
                hedge2type[i_hedge] = if itri0 == itri1 { // same tri/background
                    0
                } // no edge
                else {
                    let pixcntr0 = [iw as f32 + 0.5, ih0 as f32 + 0.5];
                    let pixcntr1 = [iw as f32 + 0.5, ih1 as f32 + 0.5];
                    let is_pixcentr0_inside_tri1 = fn_inside(fn_barycentric(tri2vtx, vtx2xyz, &pixcntr0, itri1, transform_world2pix));
                    let is_pixcentr1_inside_tri0 = fn_inside(fn_barycentric(tri2vtx, vtx2xyz, &pixcntr1, itri0, transform_world2pix));
                    match (is_pixcentr0_inside_tri1, is_pixcentr1_inside_tri0) {
                        (false, false) => {1}, // shared edge
                        (true, false) => {2}, // tri0 is in front of tri1 (only tri0 receive gradient)
                        (false, true) => {3}, // tri1 is in front of tri0 (only tri1 receive gradient)
                        (true, true) => {4} // intersection
                    }
                };
            }
            {
                let val0 = if pix2tri[ipix0] == u32::MAX { 0. } else { 1. };
                let val1 = if pix2tri[ipix1] == u32::MAX { 0. } else { 1. };
                hedge2dldr[i_hedge] = (dldw_pixval[ipix0] + dldw_pixval[ipix1]) * 0.5 * (val0 - val1);
            }
        }
    }

    // --------------------------
    // vertical edge
    assert_eq!(vedge2type.len(), img_h*(img_w-1) );
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
                vedge2type[i_vedge] = if itri0 == itri1 { // same tri/background
                    0
                }
                else {
                    let pixcntr0 = [iw0 as f32 + 0.5, ih as f32 + 0.5];
                    let pixcntr1 = [iw1 as f32 + 0.5, ih as f32 + 0.5];
                    let is_pixcentr0_inside_tri1 = fn_inside(fn_barycentric(tri2vtx, vtx2xyz, &pixcntr0, itri1, transform_world2pix));
                    let is_pixcentr1_inside_tri0 = fn_inside(fn_barycentric(tri2vtx, vtx2xyz, &pixcntr1, itri0, transform_world2pix));
                    match (is_pixcentr0_inside_tri1, is_pixcentr1_inside_tri0) {
                        (false, false) => {1} // shared edge
                        (true, false) => {2}, // tri0 is in front of tri1 (only tri0 receive gradient)
                        (false, true) => {3}, // tri1 is in front of tri0 (only tri1 receive gradient)
                        (true, true) => {4}, // intersection
                    }
                };
            }
            {
                let val0 = if pix2tri[ipix0] == u32::MAX { 0. } else { 1. };
                let val1 = if pix2tri[ipix1] == u32::MAX { 0. } else { 1. };
                vedge2dldr[i_vedge] = (dldw_pixval[ipix0] + dldw_pixval[ipix1]) * 0.5 * (val0 - val1);
            }
        }
    }
}

pub fn smooth_gradient(
    (img_w, img_h): (usize, usize),
    hedge2type: &mut [u8],
    hedge2dldr: &mut [f32],
    vedge2type: &mut [u8],
    vedge2dldr: &mut [f32])
{
    assert_eq!(hedge2type.len(), (img_h-1)*img_w );
    assert_eq!(hedge2type.len(), hedge2dldr.len());
    for iter in 0..1000 {
        for iw in 0..img_w {
            for ih in 0..img_h - 1 {
                let i_hedge_c = ih * img_w + iw;
                if hedge2type[i_hedge_c] == 2 || hedge2type[i_hedge_c] == 3 { continue; }
                let mut n_sum = 0;
                let mut v_sum: f32 = 0.;
                //
                if ih != 0 { // north
                    let i0_hedge = (ih - 1) * img_w + iw;
                    if hedge2type[i0_hedge] != 2 {
                        v_sum += hedge2dldr[i0_hedge];
                        n_sum += 1;
                    }
                }
                //
                if ih != img_h-2 { // south
                    let i0_hedge = (ih + 1) * img_w + iw;
                    if hedge2type[i0_hedge] != 3 {
                        v_sum += hedge2dldr[i0_hedge];
                        n_sum += 1;
                    }
                }

                if iw != 0 { // west
                    let i0_hedge = ih * img_w + iw - 1;
                    if hedge2type[i0_hedge] != 3 {
                        let iwn_vedge = ih * (img_w - 1) + iw - 1;
                        let iws_vedge = (ih + 1) * (img_w - 1) + iw - 1;
                        let type_wn = vedge2type[iwn_vedge];
                        let type_ws = vedge2type[iws_vedge];
                        if type_wn != 2 && type_wn != 3 && type_ws != 2 && type_ws != 3 {
                            v_sum += hedge2dldr[i0_hedge];
                            n_sum += 1;
                        }
                    }
                }
                
                if iw != img_w - 1 { // east
                    let i0_hedge = ih * img_w + iw + 1;
                    if hedge2type[i0_hedge] != 2 {
                        let ien_vedge = ih * (img_w - 1) + iw + 1;
                        let type_en = vedge2type[ien_vedge];
                        if type_en != 2 && type_en != 3 && ih != img_h - 1 && iw != img_w - 2 {
                            let ies_vedge = (ih + 1) * (img_w - 1) + iw + 1;
                            let type_es = vedge2type[ies_vedge];
                            if type_es != 2 && type_es != 3 {
                                v_sum += hedge2dldr[i0_hedge];
                                n_sum += 1;
                            }
                        }
                    }
                }

                // ----------------------------
                if n_sum != 0 {
                    hedge2dldr[i_hedge_c] = v_sum / n_sum as f32;
                }
            }
        }
    }
    //
    assert_eq!(vedge2type.len(), img_h*(img_w-1) );
    assert_eq!(vedge2type.len(), vedge2dldr.len());
}

pub fn bwd(
    tri2vtx: &[u32],
    vtx2xyz: &[f32],
    dldw_vtx2xyz: &mut [f32],
    transform_world2pix: &[f32; 16],
    (img_w, img_h): (usize, usize),
    dldw_pixval: &[f32],
    pix2tri: &[u32],
) {
    use del_geo_core::mat4_col_major::Mat4ColMajor;
    let transform_pix2world = transform_world2pix.transpose();
    // vertical edge
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
            let is_pixcentr0_inside_tri1 = fn_inside(fn_barycentric(tri2vtx, vtx2xyz, &pixcntr0, itri1, transform_world2pix));
            let is_pixcentr1_inside_tri0 = fn_inside(fn_barycentric(tri2vtx, vtx2xyz, &pixcntr1, itri0, transform_world2pix));
            if !is_pixcentr0_inside_tri1 && !is_pixcentr1_inside_tri0 {
                continue;
            }
            let val0 = if pix2tri[ipix0] == u32::MAX { 0. } else { 1. };
            let val1 = if pix2tri[ipix1] == u32::MAX { 0. } else { 1. };
            let dldpa = (dldw_pixval[ipix0] + dldw_pixval[ipix1]) * 0.5 * (val0 - val1);
            if is_pixcentr0_inside_tri1 && is_pixcentr1_inside_tri0 {
                dbg!("todo");
                continue;
            } else if is_pixcentr1_inside_tri0 {
                // only tri1 receive gradient
                let b = fn_barycentric(tri2vtx, vtx2xyz, &pixcntr1, itri1, transform_world2pix).unwrap();
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
                let b = fn_barycentric(tri2vtx, vtx2xyz, &pixcntr0, itri0, transform_world2pix).unwrap();
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
            let is_pixcentr0_inside_tri1 = fn_inside(fn_barycentric(tri2vtx, vtx2xyz, &pixcntr0, itri1, transform_world2pix));
            let is_pixcentr1_inside_tri0 = fn_inside(fn_barycentric(tri2vtx, vtx2xyz, &pixcntr1, itri0, transform_world2pix));
            if !is_pixcentr0_inside_tri1 && !is_pixcentr1_inside_tri0 {
                continue;
            }
            let val0 = if pix2tri[ipix0] == u32::MAX { 0. } else { 1. };
            let val1 = if pix2tri[ipix1] == u32::MAX { 0. } else { 1. };
            let dldpa = (dldw_pixval[ipix0] + dldw_pixval[ipix1]) * 0.5 * (val0 - val1);
            if is_pixcentr0_inside_tri1 && is_pixcentr1_inside_tri0 {
                dbg!("todo");
                continue;
            } else if is_pixcentr1_inside_tri0 {
                // only tri1 recieve gradient
                let b = fn_barycentric(tri2vtx, vtx2xyz, &pixcntr1, itri1, transform_world2pix).unwrap();
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
                let b = fn_barycentric(tri2vtx, vtx2xyz, &pixcntr0, itri0, transform_world2pix).unwrap();
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
