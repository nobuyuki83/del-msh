fn fn_barycentric(
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
                let xyz = crate::trimesh3::to_tri3(tri2vtx, vtx2xyz, itri1)
                    .position_from_barycentric_coordinates(b[0], b[1]);
                let dpixdxyz =
                    del_geo_core::mat4_col_major::jacobian_transform(transform_world2pix, &xyz);
                let dldw_pix = [0., dldpa, 0.];
                let dldw_xyz = del_geo_core::vec3::mult_mat3_col_major(&dldw_pix, &dpixdxyz);
                for inode in 0..3 {
                    let ivtx = tri2vtx[itri1][inode] as usize;
                    dldw_vtx2xyz[ivtx][0] += b[inode] * dldw_xyz[0];
                    dldw_vtx2xyz[ivtx][1] += b[inode] * dldw_xyz[1];
                    dldw_vtx2xyz[ivtx][2] += b[inode] * dldw_xyz[2];
                }
            } else {
                // only tri0 recieve gradient
                let b = fn_barycentric(tri2vtx, vtx2xyz, &pixcntr0, itri0, transform_world2pix)
                    .unwrap();
                let itri0 = itri0 as usize;
                let xyz = crate::trimesh3::to_tri3(tri2vtx, vtx2xyz, itri0)
                    .position_from_barycentric_coordinates(b[0], b[1]);
                let dpixdxyz =
                    del_geo_core::mat4_col_major::jacobian_transform(transform_world2pix, &xyz);
                let dldw_pix = [0., dldpa, 0.];
                let dldw_xyz = del_geo_core::vec3::mult_mat3_col_major(&dldw_pix, &dpixdxyz);
                for inode in 0..3 {
                    let ivtx = tri2vtx[itri0][inode] as usize;
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
                let xyz = crate::trimesh3::to_tri3(tri2vtx, vtx2xyz, itri1)
                    .position_from_barycentric_coordinates(b[0], b[1]);
                let dpixdxyz =
                    del_geo_core::mat4_col_major::jacobian_transform(transform_world2pix, &xyz);
                let dldw_pix = [dldpa, 0., 0.];
                let dldw_xyz = del_geo_core::vec3::mult_mat3_col_major(&dldw_pix, &dpixdxyz);
                for inode in 0..3 {
                    let ivtx = tri2vtx[itri1][inode] as usize;
                    dldw_vtx2xyz[ivtx][0] += b[inode] * dldw_xyz[0];
                    dldw_vtx2xyz[ivtx][1] += b[inode] * dldw_xyz[1];
                    dldw_vtx2xyz[ivtx][2] += b[inode] * dldw_xyz[2];
                }
            } else {
                // only tri0 recieve gradient
                let b = fn_barycentric(tri2vtx, vtx2xyz, &pixcntr0, itri0, transform_world2pix)
                    .unwrap();
                let itri0 = itri0 as usize;
                let xyz = crate::trimesh3::to_tri3(tri2vtx, vtx2xyz, itri0)
                    .position_from_barycentric_coordinates(b[0], b[1]);
                let dpixdxyz =
                    del_geo_core::mat4_col_major::jacobian_transform(transform_world2pix, &xyz);
                let dldw_pix = [dldpa, 0., 0.];
                let dldw_xyz = del_geo_core::vec3::mult_mat3_col_major(&dldw_pix, &dpixdxyz);
                for inode in 0..3 {
                    let ivtx = tri2vtx[itri0][inode] as usize;
                    dldw_vtx2xyz[ivtx][0] += b[inode] * dldw_xyz[0];
                    dldw_vtx2xyz[ivtx][1] += b[inode] * dldw_xyz[1];
                    dldw_vtx2xyz[ivtx][2] += b[inode] * dldw_xyz[2];
                }
            }
        }
    }
}

#[test]
fn test_hoge() {
    let tri2vtx = [[0, 1, 2], [3, 4, 5]];
    let vtx2xyz = [
        [-0.4, 0.1, 0.5],
        [0.6, -0.5, 0.1],
        [0.5, 0.5, 0.0],
        [0.6, 0.0, 0.5],
        [-0.4, 0.5, -0.1],
        [-0.3, -0.5, 0.0],
    ];
    let vtx2uvw = [
        [-1., -1., -1.],
        [-1., -1., -1.],
        [-1., -1., -1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
    ];

    let img_shape = (200, 150);
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

    let mode = crate::pix2occlusion::Occlusion;
    {
        let num_sample = 100;
        let eps = 1.0e-3;
        type Sampler = crate::trimesh3_raycast::BoxPixelSampler<rand_chacha::ChaChaRng>;
        let pix2val0 = {
            let vtx2xyz = vtx2xyz
                .iter()
                .zip(vtx2uvw.iter())
                .map(|(x, u)| [x[0] - eps * u[0], x[1] - eps * u[1], x[2] - eps * u[2]])
                .collect::<Vec<_>>();
            crate::trimesh3_raycast::multi_sample::<_, Sampler>(
                &tri2vtx,
                &vtx2xyz,
                &transform_world2ndc,
                img_shape,
                num_sample,
                &mode,
            )
        };
        let pix2val1 = {
            let vtx2xyz = vtx2xyz
                .iter()
                .zip(vtx2uvw.iter())
                .map(|(x, u)| [x[0] + eps * u[0], x[1] + eps * u[1], x[2] + eps * u[2]])
                .collect::<Vec<_>>();
            crate::trimesh3_raycast::multi_sample::<_, Sampler>(
                &tri2vtx,
                &vtx2xyz,
                &transform_world2ndc,
                img_shape,
                num_sample,
                &mode,
            )
        };
        let pix2diff = pix2val1
            .iter()
            .zip(pix2val0.iter())
            .map(|(u, v)| (u - v) / eps)
            .collect::<Vec<_>>();
        //dbg!(pix2diff);
    }

    /*
    for i_pix in 0..img_shape.0 * img_shape.1 {
        let bvhnodes = crate::bvhnodes_morton::from_triangle_mesh(&tri2vtx, &vtx2xyz);
        let bvhnode2aabb =
            crate::bvhnode2aabb3::from_uniform_mesh_with_bvh(0, &bvhnodes, &tri2vtx, &vtx2xyz, None);
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
        let pix2occ: Vec<f32> = pix2tri
            .iter()
            .map(|&i_tri| if i_tri == u32::MAX { 0.0 } else { 1.0 })
            .collect();
        let pix2trg = {
            let mut pix2trg = vec![0f32; img_shape.0 * img_shape.1];

        };
    }
     */

    /*
    let mut pix2depth = vec![0f32; img_shape.0 * img_shape.1];
    crate::pix2depth::pix2depth_from_pix2tri(
        &mut pix2depth,
        &pix2tri,
        &tri2vtx,
        &vtx2xyz,
        img_shape,
        &transform_ndc2world,
    );
    let path_dir = std::path::Path::new("../target/out_del_msh_cpu");
    del_canvas::write_png_from_float_image(
        path_dir.join("edgegrad_depth.png"),
        img_shape,
        1,
        &pix2depth,
    )
    .unwrap();

    del_canvas::write_png_from_float_image(path_dir.join("edgegrad.png"), img_shape, 1, &pix2occ)
        .unwrap();
    use rand::RngExt;
    use rand::SeedableRng;
    let mut reng = rand_chacha::ChaChaRng::seed_from_u64(0);
    let pix2trg: Vec<f32> = (0..img_shape.0 * img_shape.1)
        .map(|_| reng.random())
        .collect();

    let loss: f32 = pix2trg
        .iter()
        .zip(pix2occ.iter())
        .map(|(t, o)| t * o)
        .sum();
    dbg!(loss);

    // d(loss)/d(pix2occ[i]) = pix2trg[i]
    let transform_ndc2pix =
        del_geo_core::mat4_col_major::from_transform_ndc2pix(img_shape);
    let transform_world2pix = del_geo_core::mat4_col_major::mult_mat_col_major(
        &transform_ndc2pix,
        &transform_world2ndc,
    );
    let mut dldw_vtx2xyz = vec![[0f32; 3]; vtx2xyz.len()];
    bwd(
        &tri2vtx,
        &vtx2xyz,
        &mut dldw_vtx2xyz,
        &transform_world2pix,
        img_shape,
        &pix2tri,
        1,
        &pix2occ,
        &pix2trg,
    );
    dbg!(&dldw_vtx2xyz);
    */
}
