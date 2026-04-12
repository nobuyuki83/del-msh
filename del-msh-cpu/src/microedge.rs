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
            if (b0.0 >= 0. && b0.1 >= 0. && b0.2 >= 0.) || (b0.0 <= 0. && b0.1 <= 0. && b0.2 <= 0.)
            {
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
            } else if is_pixcentr1_inside_tri0 {
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
            } else if is_pixcentr1_inside_tri0 {
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
