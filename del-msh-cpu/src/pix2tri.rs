use num_traits::AsPrimitive;

pub fn pix2tri_by_raycast<Index>(
    pix2tri: &mut [Index],
    tri2vtx: &[[Index; 3]],
    vtx2xyz: &[[f32; 3]],
    bvhnodes: &[[Index; 3]],
    bvhnode2aabb: &[[f32; 6]],
    img_shape: (usize, usize), // (width, height)
    transform_ndc2world: &[f32; 16],
) where
    Index: num_traits::PrimInt + AsPrimitive<usize> + Sync + Send,
    usize: AsPrimitive<Index>,
{
    assert_eq!(pix2tri.len(), img_shape.0 * img_shape.1);
    let tri_for_pix = |i_pix: usize| -> Index {
        let i_h = i_pix / img_shape.0;
        let i_w = i_pix - i_h * img_shape.0;
        //
        let (ray_org, ray_dir) =
            del_geo_core::mat4_col_major::ray_from_transform_ndc2world_and_pixel_coordinates(
                (i_w as f32 + 0.5, i_h as f32 + 0.5),
                &(img_shape.0 as f32, img_shape.1 as f32),
                transform_ndc2world,
            );
        if let Some((_t, i_tri, _bc)) = crate::search_bvh3::first_intersection_ray(
            &ray_org,
            &ray_dir,
            &crate::search_bvh3::TriMesh3WithBvhRef {
                tri2vtx,
                vtx2xyz,
                bvhnode2tri_btree: bvhnodes,
                bvhnode2aabb,
            },
            0,
            f32::INFINITY,
        ) {
            i_tri.as_()
        } else {
            Index::max_value()
        }
    };
    use rayon::prelude::*;
    pix2tri
        .par_iter_mut()
        .enumerate()
        .for_each(|(i_pix, i_tri)| *i_tri = tri_for_pix(i_pix));
}

/// Linearly interpolate two vertices in homogeneous clip space at parameter `t`.
/// This preserves correct perspective when vertices are later divided by `hw`.
fn interp_hom(ndc_a: [f32; 3], hw_a: f32, ndc_b: [f32; 3], hw_b: f32, t: f32) -> ([f32; 3], f32) {
    let hw_c = hw_a + t * (hw_b - hw_a);
    let inv = 1.0 / hw_c;
    let ndc_c = [
        (ndc_a[0] * hw_a + t * (ndc_b[0] * hw_b - ndc_a[0] * hw_a)) * inv,
        (ndc_a[1] * hw_a + t * (ndc_b[1] * hw_b - ndc_a[1] * hw_a)) * inv,
        (ndc_a[2] * hw_a + t * (ndc_b[2] * hw_b - ndc_a[2] * hw_a)) * inv,
    ];
    (ndc_c, hw_c)
}

/// Rasterize one projected triangle given its NDC coordinates and clip-w values.
fn rasterize_projected_tri<Index>(
    pix2tri: &mut [Index],
    pix2depth: &mut [f32],
    img_shape: (usize, usize),
    i_tri_val: Index,
    ndc: [[f32; 3]; 3],
    hw: [f32; 3],
) where
    Index: Copy,
{
    let (width, height) = img_shape;
    let px: [[f32; 2]; 3] = [
        [
            (ndc[0][0] + 1.) * 0.5 * width as f32,
            (1. - ndc[0][1]) * 0.5 * height as f32,
        ],
        [
            (ndc[1][0] + 1.) * 0.5 * width as f32,
            (1. - ndc[1][1]) * 0.5 * height as f32,
        ],
        [
            (ndc[2][0] + 1.) * 0.5 * width as f32,
            (1. - ndc[2][1]) * 0.5 * height as f32,
        ],
    ];
    if px.iter().any(|p| !p[0].is_finite() || !p[1].is_finite()) {
        return;
    }
    let min_x = px[0][0].min(px[1][0]).min(px[2][0]);
    let max_x = px[0][0].max(px[1][0]).max(px[2][0]);
    let min_y = px[0][1].min(px[1][1]).min(px[2][1]);
    let max_y = px[0][1].max(px[1][1]).max(px[2][1]);
    if max_x < 0. || min_x >= width as f32 || max_y < 0. || min_y >= height as f32 {
        return;
    }
    for [ix, iy] in del_geo_core::tri2_scanline::TriangleScanlineIter::new(px[0], px[1], px[2]) {
        if ix < 0 || iy < 0 || ix >= width as i32 || iy >= height as i32 {
            continue;
        }
        let i_pix = iy as usize * width + ix as usize;
        let pix_center = [ix as f32 + 0.5, iy as f32 + 0.5];
        let Some((b0, b1, b2)) =
            del_geo_core::tri2::barycentric_coords(&px[0], &px[1], &px[2], &pix_center)
        else {
            continue;
        };
        if b0 < 0. || b1 < 0. || b2 < 0. {
            continue;
        }
        let q0 = b0 / hw[0];
        let q1 = b1 / hw[1];
        let q2 = b2 / hw[2];
        let inv_sum = 1.0 / (q0 + q1 + q2);
        let depth = (q0 * ndc[0][2] + q1 * ndc[1][2] + q2 * ndc[2][2]) * inv_sum;
        if depth > pix2depth[i_pix] {
            pix2depth[i_pix] = depth;
            pix2tri[i_pix] = i_tri_val;
        }
    }
}

pub fn pix2tri_by_rasterization<Index>(
    pix2tri: &mut [Index],
    pix2depth: &mut [f32],
    tri2vtx: &[[Index; 3]],
    vtx2xyz: &[[f32; 3]],
    img_shape: (usize, usize), // (width, height)
    transform_world2ndc: &[f32; 16],
) where
    Index: num_traits::PrimInt + AsPrimitive<usize> + Sync + Send,
    usize: AsPrimitive<Index>,
{
    assert_eq!(pix2tri.len(), img_shape.0 * img_shape.1);
    // m[11] is the z→w element of the projection. It equals +1 (Blender) or -1 (OpenGL)
    // for standard perspective, and 0 for orthographic.
    // For perspective: hw * t11 < 0 means "in front of camera".
    // Near-clip score c = hw * (ndc_z - t11): c >= 0 means vertex is at or beyond the near
    // clip plane (valid). This formula works for both Blender and OpenGL conventions.
    let t11 = transform_world2ndc[11];
    for (i_tri, node2vtx) in tri2vtx.iter().enumerate() {
        let i0: usize = node2vtx[0].as_();
        let i1: usize = node2vtx[1].as_();
        let i2: usize = node2vtx[2].as_();
        let Some((ndc0, hw0)) =
            del_geo_core::mat4_col_major::transform_homogeneous(transform_world2ndc, &vtx2xyz[i0])
        else {
            continue;
        };
        let Some((ndc1, hw1)) =
            del_geo_core::mat4_col_major::transform_homogeneous(transform_world2ndc, &vtx2xyz[i1])
        else {
            continue;
        };
        let Some((ndc2, hw2)) =
            del_geo_core::mat4_col_major::transform_homogeneous(transform_world2ndc, &vtx2xyz[i2])
        else {
            continue;
        };

        // Skip any triangle with a vertex behind the camera plane.
        if t11 != 0. && (hw0 * t11 >= 0. || hw1 * t11 >= 0. || hw2 * t11 >= 0.) {
            continue;
        }

        let i_tri_val: Index = i_tri.as_();

        if t11 == 0. {
            // Orthographic: no near-clip homogeneous issue.
            rasterize_projected_tri(
                pix2tri,
                pix2depth,
                img_shape,
                i_tri_val,
                [ndc0, ndc1, ndc2],
                [hw0, hw1, hw2],
            );
            continue;
        }

        // Near-clip scores (>= 0 means valid).
        let c0 = hw0 * (ndc0[2] - t11);
        let c1 = hw1 * (ndc1[2] - t11);
        let c2 = hw2 * (ndc2[2] - t11);

        if c0 >= 0. && c1 >= 0. && c2 >= 0. {
            // All vertices at or beyond near clip: rasterize whole triangle.
            rasterize_projected_tri(
                pix2tri,
                pix2depth,
                img_shape,
                i_tri_val,
                [ndc0, ndc1, ndc2],
                [hw0, hw1, hw2],
            );
        } else if c0 < 0. && c1 < 0. && c2 < 0. {
            // All vertices between camera and near clip: skip.
        } else {
            // Mixed: clip against the near plane.
            // clip_t(ca, cb) gives the parameter where c = 0 on edge a→b.
            let clip_t = |ca: f32, cb: f32| ca / (ca - cb);
            let v = [(ndc0, hw0, c0), (ndc1, hw1, c1), (ndc2, hw2, c2)];
            let n_valid = v.iter().filter(|&&(_, _, c)| c >= 0.).count();
            if n_valid == 1 {
                // 1 inside vertex → 1 output triangle.
                let (ia, ib, ic) = if v[0].2 >= 0. {
                    (0, 1, 2)
                } else if v[1].2 >= 0. {
                    (1, 0, 2)
                } else {
                    (2, 0, 1)
                };
                let (ndc_a, hw_a, ca) = v[ia];
                let (ndc_b, hw_b, cb) = v[ib];
                let (ndc_c, hw_c, cc) = v[ic];
                let (ndc_ab, hw_ab) = interp_hom(ndc_a, hw_a, ndc_b, hw_b, clip_t(ca, cb));
                let (ndc_ac, hw_ac) = interp_hom(ndc_a, hw_a, ndc_c, hw_c, clip_t(ca, cc));
                rasterize_projected_tri(
                    pix2tri,
                    pix2depth,
                    img_shape,
                    i_tri_val,
                    [ndc_a, ndc_ab, ndc_ac],
                    [hw_a, hw_ab, hw_ac],
                );
            } else {
                // 2 inside vertices → 2 output triangles.
                let (ic, ia, ib) = if v[0].2 < 0. {
                    (0, 1, 2)
                } else if v[1].2 < 0. {
                    (1, 0, 2)
                } else {
                    (2, 0, 1)
                };
                let (ndc_c, hw_c, cc) = v[ic];
                let (ndc_a, hw_a, ca) = v[ia];
                let (ndc_b, hw_b, cb) = v[ib];
                let (ndc_ac, hw_ac) = interp_hom(ndc_a, hw_a, ndc_c, hw_c, clip_t(ca, cc));
                let (ndc_bc, hw_bc) = interp_hom(ndc_b, hw_b, ndc_c, hw_c, clip_t(cb, cc));
                rasterize_projected_tri(
                    pix2tri,
                    pix2depth,
                    img_shape,
                    i_tri_val,
                    [ndc_a, ndc_b, ndc_ac],
                    [hw_a, hw_b, hw_ac],
                );
                rasterize_projected_tri(
                    pix2tri,
                    pix2depth,
                    img_shape,
                    i_tri_val,
                    [ndc_b, ndc_bc, ndc_ac],
                    [hw_b, hw_bc, hw_ac],
                );
            }
        }
    }
}

#[test]
fn test_pix2tri() {
    const IMG_RES: usize = 256;
    let path_dir_asset = std::path::Path::new("../asset/");
    let path_dir_out = std::path::Path::new("../target/out_del_msh_cpu");
    assert!(path_dir_asset.exists());
    assert!(path_dir_out.exists());

    let img_shape = (IMG_RES, IMG_RES);
    let (tri2vtx, vtx2xyz) =
        crate::io_wavefront_obj::load_tri_mesh(path_dir_asset.join("bunny_50k.obj"), Some(1.0))
            .unwrap();
    let bvhnodes = crate::bvhnodes_morton::from_triangle_mesh(&tri2vtx, &vtx2xyz);
    let bvhnode2aabb =
        crate::bvhnode2aabb3::from_uniform_mesh_with_bvh(0, &bvhnodes, &tri2vtx, &vtx2xyz, None);
    const NUM_ITR: usize = 4;
    for i_itr in 0..NUM_ITR {
        let transform_world2ndc = {
            let transform0 =
                del_geo_core::mat4_col_major::camera_perspective_blender(1.0, 35.0, 0.1, 3.0, true);
            let transform1 = del_geo_core::mat4_col_major::from_translate(&[
                0.0,
                0.0,
                -2.0 * ((NUM_ITR - i_itr - 1) as f32) / NUM_ITR as f32,
            ]);
            del_geo_core::mat4_col_major::mult_mat_col_major(&transform0, &transform1)
        };
        let transform_ndc2world =
            del_geo_core::mat4_col_major::try_inverse_with_pivot(&transform_world2ndc).unwrap();
        let pix2tri_raycast = {
            let mut pix2tri = vec![u32::MAX; IMG_RES * IMG_RES];
            pix2tri_by_raycast(
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
        let pix2tri_rasterization = {
            let mut pix2tri = vec![u32::MAX; IMG_RES * IMG_RES];
            let mut pix2depth = vec![f32::NEG_INFINITY; IMG_RES * IMG_RES];
            pix2tri_by_rasterization(
                &mut pix2tri,
                &mut pix2depth,
                &tri2vtx,
                &vtx2xyz,
                img_shape,
                &transform_world2ndc,
            );
            pix2tri
        };
        // Foreground/background must agree exactly; specific triangle identity may differ
        // by at most a handful of pixels at shared edges (boundary tie-breaking differs
        // between 2D rasterization and 3D ray casting).
        let mut num_mismatch = 0usize;
        pix2tri_raycast
            .iter()
            .zip(pix2tri_rasterization.iter())
            .for_each(|(i_tri_raycast, i_tri_rasterization)| {
                let ray_bg = *i_tri_raycast == u32::MAX;
                let rst_bg = *i_tri_rasterization == u32::MAX;
                assert_eq!(ray_bg, rst_bg, "foreground/background mismatch");
                if i_tri_raycast != i_tri_rasterization {
                    num_mismatch += 1;
                }
            });
        assert!(
            num_mismatch <= 10,
            "too many triangle-identity mismatches: {num_mismatch}"
        );
        let pix2rgb = pix2tri_raycast
            //let pix2rgb = pix2tri_rasterization
            .iter()
            .map(|&i_tri| {
                if i_tri == u32::MAX {
                    [0., 0., 0.]
                } else {
                    let d = (i_tri as f32) / tri2vtx.len() as f32;
                    del_canvas::colormap::apply_colormap(
                        d,
                        0.,
                        1.,
                        del_canvas::colormap::COLORMAP_JET,
                    )
                }
            })
            .collect::<Vec<_>>();
        del_canvas::write_png_from_float_image(
            path_dir_out.join(format!("pix2tri_{i_itr}.png")),
            img_shape,
            3,
            pix2rgb.as_flattened(),
        )
        .unwrap()
    }
}

#[allow(clippy::too_many_arguments)]
pub fn interpolate<Index, Real>(
    (img_w, img_h): (usize, usize),
    pix2tri: &[Index],
    tri2vtx: &[[Index; 3]],
    vtx2xyz: &[[Real; 3]],
    num_vdim: usize,
    vtx2val: &[Real],
    transform_ndc2world: &[Real; 16],
    pix2val: &mut [Real],
) where
    Index: AsPrimitive<usize> + num_traits::PrimInt + Sync + Send,
    Real: num_traits::Float + Send + 'static + std::marker::Sync,
    usize: AsPrimitive<Real>,
{
    assert_eq!(pix2tri.len(), img_w * img_h);
    assert!(num_vdim > 0);
    assert_eq!(vtx2xyz.len(), vtx2val.len() / num_vdim);
    assert_eq!(pix2val.len(), img_w * img_h * num_vdim);
    let one = Real::one();
    let half = one / (one + one);
    use rayon::prelude::*;
    pix2tri
        .par_iter()
        .zip(pix2val.par_chunks_mut(num_vdim))
        .enumerate()
        .for_each(|(i_pix, (&i_tri, val)): (usize, (&Index, &mut [Real]))| {
            if i_tri == Index::max_value() {
                return;
            }
            let i_h = i_pix / img_w;
            let i_w = i_pix - i_h * img_w;
            let (ray_org, ray_dir) =
                del_geo_core::mat4_col_major::ray_from_transform_ndc2world_and_pixel_coordinates(
                    (i_w.as_() + half, i_h.as_() + half),
                    &(img_w.as_(), img_h.as_()),
                    transform_ndc2world,
                );
            let i0 = tri2vtx[i_tri.as_()][0].as_();
            let i1 = tri2vtx[i_tri.as_()][1].as_();
            let i2 = tri2vtx[i_tri.as_()][2].as_();
            let p0 = &vtx2xyz[i0];
            let p1 = &vtx2xyz[i1];
            let p2 = &vtx2xyz[i2];
            let (_q, bc) = del_geo_core::tri3::intersection_plane_of_tri3_against_line(
                p0, p1, p2, &ray_org, &ray_dir,
            );
            for i_vdim in 0..num_vdim {
                val[i_vdim] = bc[0] * vtx2val[i0 * num_vdim + i_vdim]
                    + bc[1] * vtx2val[i1 * num_vdim + i_vdim]
                    + bc[2] * vtx2val[i2 * num_vdim + i_vdim];
            }
        });
}

#[allow(clippy::too_many_arguments)]
pub fn interpolate_bwd<Index, Real>(
    (img_w, img_h): (usize, usize),
    pix2tri: &[Index],
    tri2vtx: &[[Index; 3]],
    vtx2xyz: &[[Real; 3]],
    num_vdim: usize,
    vtx2val: &[Real],
    transform_ndc2world: &[Real; 16],
    dldw_pix2val: &[Real],
    dldw_vtx2xyz: &mut [[Real; 3]],
    dldw_vtx2val: &mut [Real],
) where
    Index: AsPrimitive<usize> + num_traits::PrimInt + Sync + Send + std::fmt::Debug,
    Real: num_traits::Float + 'static,
    usize: AsPrimitive<Real>,
{
    assert_eq!(pix2tri.len(), img_w * img_h);
    assert!(num_vdim > 0);
    assert_eq!(vtx2val.len() / num_vdim, vtx2xyz.len());
    assert_eq!(dldw_vtx2xyz.len(), vtx2xyz.len());
    assert_eq!(dldw_vtx2val.len() / num_vdim, vtx2xyz.len());
    let one = Real::one();
    let half = one / (one + one);
    let zero = Real::zero();
    for i_pix in 0..img_w * img_h {
        let i_tri = pix2tri[i_pix];
        if i_tri == Index::max_value() {
            continue;
        }
        let i_w = i_pix % img_w;
        let i_h = i_pix / img_w;
        let (ray_org, ray_dir) =
            del_geo_core::mat4_col_major::ray_from_transform_ndc2world_and_pixel_coordinates(
                (i_w.as_() + half, i_h.as_() + half),
                &(img_w.as_(), img_h.as_()),
                transform_ndc2world,
            );
        let i0 = tri2vtx[i_tri.as_()][0].as_();
        let i1 = tri2vtx[i_tri.as_()][1].as_();
        let i2 = tri2vtx[i_tri.as_()][2].as_();
        let p0 = &vtx2xyz[i0];
        let p1 = &vtx2xyz[i1];
        let p2 = &vtx2xyz[i2];
        let (mut dldw_bc0, mut dldw_bc1, mut dldw_bc2) = (zero, zero, zero);
        for i_vdim in 0..num_vdim {
            dldw_bc0 = dldw_bc0
                + vtx2val[i0 * num_vdim + i_vdim] * dldw_pix2val[i_pix * num_vdim + i_vdim];
            dldw_bc1 = dldw_bc1
                + vtx2val[i1 * num_vdim + i_vdim] * dldw_pix2val[i_pix * num_vdim + i_vdim];
            dldw_bc2 = dldw_bc2
                + vtx2val[i2 * num_vdim + i_vdim] * dldw_pix2val[i_pix * num_vdim + i_vdim];
        }
        dldw_bc1 = dldw_bc1 - dldw_bc0;
        dldw_bc2 = dldw_bc2 - dldw_bc0;
        let (_t, bc1, bc2, dldw_p0, dldw_p1, dldw_p2) =
            del_geo_core::tri3::intersection_against_line_bwd_wrt_tri(
                p0, p1, p2, &ray_org, &ray_dir, zero, dldw_bc1, dldw_bc2,
            );
        let bc0 = one - bc1 - bc2;
        for i_dim in 0..3 {
            dldw_vtx2xyz[i0][i_dim] = dldw_vtx2xyz[i0][i_dim] + dldw_p0[i_dim];
            dldw_vtx2xyz[i1][i_dim] = dldw_vtx2xyz[i1][i_dim] + dldw_p1[i_dim];
            dldw_vtx2xyz[i2][i_dim] = dldw_vtx2xyz[i2][i_dim] + dldw_p2[i_dim];
        }
        for i_vdim in 0..num_vdim {
            dldw_vtx2val[i0 * num_vdim + i_vdim] = dldw_vtx2val[i0 * num_vdim + i_vdim]
                + dldw_pix2val[i_pix * num_vdim + i_vdim] * bc0;
            dldw_vtx2val[i1 * num_vdim + i_vdim] = dldw_vtx2val[i1 * num_vdim + i_vdim]
                + dldw_pix2val[i_pix * num_vdim + i_vdim] * bc1;
            dldw_vtx2val[i2 * num_vdim + i_vdim] = dldw_vtx2val[i2 * num_vdim + i_vdim]
                + dldw_pix2val[i_pix * num_vdim + i_vdim] * bc2;
        }
    }
}

#[test]
fn test_interpolate() {
    const IMG_RES: usize = 128;
    type Real = f64;
    use num_traits::Zero;
    let img_shape = (IMG_RES, IMG_RES);
    //    let (tri2vtx, vtx2xyz, transform_world2ndc, dxyz) = geometry(0.);
    let (tri2vtx, vtx2xyz0) = crate::trimesh3_primitive::torus_zup::<u32, Real>(1.3, 0.4, 64, 32);
    let vtx2xyz0 = {
        let transform0 = del_geo_core::mat4_col_major::from_rot_x(1.15);
        let transform1 = del_geo_core::mat4_col_major::from_translate(&[0.01, 0.61, 0.03]);
        let transform = del_geo_core::mat4_col_major::mult_mat_col_major(&transform1, &transform0);
        crate::vtx2xyz::transform_homogeneous(&vtx2xyz0, &transform)
    };
    let transform_world2ndc = del_geo_core::mat4_col_major::from_diagonal(0.5, 0.5, 0.5, 1.0);
    let transform_ndc2world: [Real; 16] =
        del_geo_core::mat4_col_major::try_inverse_with_pivot(&transform_world2ndc).unwrap();
    let pix2tri = {
        let vtx2xyz0: Vec<_> = vtx2xyz0
            .iter()
            .map(|v| [v[0] as f32, v[1] as f32, v[2] as f32])
            .collect();
        let bvhnodes = crate::bvhnodes_morton::from_triangle_mesh(&tri2vtx, &vtx2xyz0);
        let bvhnode2aabb = crate::bvhnode2aabb3::from_uniform_mesh_with_bvh(
            0, &bvhnodes, &tri2vtx, &vtx2xyz0, None,
        );
        let mut pix2tri = vec![u32::MAX; IMG_RES * IMG_RES];
        let transform_ndc2world: [f32; 16] = std::array::from_fn(|i| transform_ndc2world[i] as f32);
        pix2tri_by_raycast(
            &mut pix2tri,
            &tri2vtx,
            &vtx2xyz0,
            &bvhnodes,
            &bvhnode2aabb,
            img_shape,
            &transform_ndc2world,
        );
        pix2tri
    };
    use rand::RngExt;
    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaChaRng::seed_from_u64(0);
    let num_vdim = 4;
    let dldw_pix2val: Vec<_> = (0..img_shape.0 * img_shape.1 * num_vdim)
        .map(|_| rng.random_range(-1. ..1.))
        .collect();
    let vtx2val0: Vec<_> = (0..vtx2xyz0.len() * num_vdim)
        .map(|_| rng.random_range(-1. ..1.))
        .collect();
    let mut pix2val0 = vec![Real::zero(); img_shape.0 * img_shape.1 * num_vdim];
    interpolate(
        img_shape,
        &pix2tri,
        &tri2vtx,
        &vtx2xyz0,
        num_vdim,
        &vtx2val0,
        &transform_ndc2world,
        &mut pix2val0,
    );
    let loss0: Real = pix2val0
        .iter()
        .zip(dldw_pix2val.iter())
        .map(|(v, w)| v * w)
        .sum();
    let mut dldw_vtx2xyz = vec![Real::zero(); vtx2xyz0.len() * 3];
    let mut dldw_vtx2val = vec![Real::zero(); vtx2val0.len()];
    interpolate_bwd(
        img_shape,
        &pix2tri,
        &tri2vtx,
        &vtx2xyz0,
        num_vdim,
        &vtx2val0,
        &transform_ndc2world,
        &dldw_pix2val,
        dldw_vtx2xyz.as_chunks_mut::<3>().0,
        &mut dldw_vtx2val,
    );
    {
        let mut max_difference = 0.0;
        let mut max_signal = 0.0;
        let eps = 1.0e-8;
        for i_vtx in 0..vtx2xyz0.len() {
            for i_dim in 0..3 {
                let mut vtx2xyz1 = vtx2xyz0.clone();
                vtx2xyz1[i_vtx][i_dim] += eps;
                let mut pix2val1 = vec![Real::zero(); img_shape.0 * img_shape.1 * num_vdim];
                interpolate(
                    img_shape,
                    &pix2tri,
                    &tri2vtx,
                    &vtx2xyz1,
                    num_vdim,
                    &vtx2val0,
                    &transform_ndc2world,
                    &mut pix2val1,
                );
                let loss1: Real = pix2val1
                    .iter()
                    .zip(dldw_pix2val.iter())
                    .map(|(v, w)| v * w)
                    .sum();
                let diff_num = (loss1 - loss0) / eps;
                let diff_ana = dldw_vtx2xyz[i_vtx * 3 + i_dim];
                //println!("{} {} --> {} {}", i_vtx, i_dim, diff_num, diff_ana);
                max_difference = (diff_num - diff_ana).abs().max(max_difference);
                max_signal = diff_num.abs().max(max_signal);
            }
        }
        //dbg!(max_difference / max_signal);
        assert!(
            max_difference / max_signal < 3.0e-5,
            "{}",
            max_difference / max_signal
        );
    }
    {
        let mut max_difference = 0.0;
        let mut max_signal = 0.0;
        let eps = 1.0e-7;
        for i_vtx in 0..vtx2xyz0.len() {
            for i_vdim in 0..num_vdim {
                let mut vtx2val1 = vtx2val0.clone();
                vtx2val1[i_vtx * num_vdim + i_vdim] += eps;
                let mut pix2val1 = vec![Real::zero(); img_shape.0 * img_shape.1 * num_vdim];
                interpolate(
                    img_shape,
                    &pix2tri,
                    &tri2vtx,
                    &vtx2xyz0,
                    num_vdim,
                    &vtx2val1,
                    &transform_ndc2world,
                    &mut pix2val1,
                );
                let loss1: Real = pix2val1
                    .iter()
                    .zip(dldw_pix2val.iter())
                    .map(|(v, w)| v * w)
                    .sum();
                let diff_num = (loss1 - loss0) / eps;
                let diff_ana = dldw_vtx2val[i_vtx * num_vdim + i_vdim];
                //println!("{} {} --> {} {}", i_vtx, i_vdim, diff_num, diff_ana);
                max_difference = (diff_num - diff_ana).abs().max(max_difference);
                max_signal = diff_num.abs().max(max_signal);
            }
        }
        // dbg!(max_difference / max_signal);
        assert!(
            max_difference / max_signal < 1.2e-7,
            "{}",
            max_difference / max_signal
        );
    }
}
