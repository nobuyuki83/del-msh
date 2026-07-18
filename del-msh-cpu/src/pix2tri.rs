use num_traits::AsPrimitive;

pub fn pix2tri_by_raycast<Index>(
    pix2tri: &mut [Index],
    tri2vtx: &[Index],
    vtx2xyz: &[f32],
    bvhnodes: &[Index],
    bvhnode2aabb: &[f32],
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
            &crate::search_bvh3::TriMeshWithBvh {
                tri2vtx,
                vtx2xyz,
                bvhnodes,
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

pub fn pix2tri_by_rasterization<Index>(
    pix2tri: &mut [Index],
    pix2depth: &mut [f32],
    tri2vtx: &[Index],
    vtx2xyz: &[f32],
    img_shape: (usize, usize), // (width, height)
    transform_ndc2world: &[f32; 16],
) where
    Index: num_traits::PrimInt + AsPrimitive<usize> + Sync + Send,
    usize: AsPrimitive<Index>,
{
    assert_eq!(pix2tri.len(), img_shape.0 * img_shape.1);
    let transform_world2ndc =
        del_geo_core::mat4_col_major::try_inverse(transform_ndc2world).unwrap();
    let (width, height) = img_shape;
    let num_tri = tri2vtx.len() / 3;
    for i_tri in 0..num_tri {
        let i0: usize = tri2vtx[i_tri * 3].as_();
        let i1: usize = tri2vtx[i_tri * 3 + 1].as_();
        let i2: usize = tri2vtx[i_tri * 3 + 2].as_();
        let p0 = arrayref::array_ref!(vtx2xyz, i0 * 3, 3);
        let p1 = arrayref::array_ref!(vtx2xyz, i1 * 3, 3);
        let p2 = arrayref::array_ref!(vtx2xyz, i2 * 3, 3);
        let Some(ndc0) =
            del_geo_core::mat4_col_major::transform_homogeneous(&transform_world2ndc, p0)
        else {
            continue;
        };
        let Some(ndc1) =
            del_geo_core::mat4_col_major::transform_homogeneous(&transform_world2ndc, p1)
        else {
            continue;
        };
        let Some(ndc2) =
            del_geo_core::mat4_col_major::transform_homogeneous(&transform_world2ndc, p2)
        else {
            continue;
        };
        let px0 = [
            (ndc0[0] + 1.) * 0.5 * width as f32,
            (1. - ndc0[1]) * 0.5 * height as f32,
        ];
        let px1 = [
            (ndc1[0] + 1.) * 0.5 * width as f32,
            (1. - ndc1[1]) * 0.5 * height as f32,
        ];
        let px2 = [
            (ndc2[0] + 1.) * 0.5 * width as f32,
            (1. - ndc2[1]) * 0.5 * height as f32,
        ];

        for [ix, iy] in del_geo_core::tri2_scanline::TriangleScanlineIter::new(px0, px1, px2) {
            if ix < 0 || iy < 0 || ix >= width as i32 || iy >= height as i32 {
                continue;
            }
            let i_pix = iy as usize * width + ix as usize;
            let pix_center = [ix as f32 + 0.5, iy as f32 + 0.5];
            let Some((b0, b1, b2)) =
                del_geo_core::tri2::barycentric_coords(&px0, &px1, &px2, &pix_center)
            else {
                continue;
            };
            let depth = b0 * ndc0[2] + b1 * ndc1[2] + b2 * ndc2[2];
            if depth > pix2depth[i_pix] {
                pix2depth[i_pix] = depth;
                pix2tri[i_pix] = i_tri.as_();
            }
        }
    }
}

#[test]
fn test_pix2tri() {
    const IMG_RES: usize = 256;
    let img_shape = (IMG_RES, IMG_RES);
    //    let (tri2vtx, vtx2xyz, transform_world2ndc, dxyz) = geometry(0.);
    let (tri2vtx, vtx2xyz) = crate::trimesh3_primitive::torus_zup::<u32, f32>(1.3, 0.4, 64, 32);
    let vtx2xyz = {
        let transform0 = del_geo_core::mat4_col_major::from_rot_x(1.15);
        let transform1 = del_geo_core::mat4_col_major::from_translate(&[0.01, 0.61, 0.03]);
        let transform = del_geo_core::mat4_col_major::mult_mat_col_major(&transform1, &transform0);
        crate::vtx2xyz::transform_homogeneous(&vtx2xyz, &transform)
    };
    let transform_world2ndc = del_geo_core::mat4_col_major::from_diagonal(0.5, 0.5, 0.5, 1.0);
    let transform_ndc2world =
        del_geo_core::mat4_col_major::try_inverse_with_pivot(&transform_world2ndc).unwrap();
    let pix2tri_raycast = {
        let bvhnodes = crate::bvhnodes_morton::from_triangle_mesh(&tri2vtx, &vtx2xyz, 3);
        let bvhnode2aabb = crate::bvhnode2aabb3::from_uniform_mesh_with_bvh(
            0, &bvhnodes, &tri2vtx, 3, &vtx2xyz, None,
        );
        let mut pix2tri = vec![u32::MAX; IMG_RES * IMG_RES];
        crate::pix2tri::pix2tri_by_raycast(
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
        crate::pix2tri::pix2tri_by_rasterization(
            &mut pix2tri,
            &mut pix2depth,
            &tri2vtx,
            &vtx2xyz,
            img_shape,
            &transform_ndc2world,
        );
        pix2tri
    };
    pix2tri_raycast
        .iter()
        .zip(pix2tri_rasterization.iter())
        .for_each(|(i_tri_raycast, i_tri_rasterization)| {
            //dbg!(i_tri_raycast, i_tri_rasterization);
            assert_eq!(i_tri_raycast, i_tri_rasterization);
        });
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
        let vtx2xyz0: Vec<_> = vtx2xyz0.iter().map(|v| *v as f32).collect();
        let bvhnodes = crate::bvhnodes_morton::from_triangle_mesh(&tri2vtx, &vtx2xyz0, 3);
        let bvhnode2aabb = crate::bvhnode2aabb3::from_uniform_mesh_with_bvh(
            0, &bvhnodes, &tri2vtx, 3, &vtx2xyz0, None,
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
    let vtx2val0: Vec<_> = (0..vtx2xyz0.len() / 3 * num_vdim)
        .map(|_| rng.random_range(-1. ..1.))
        .collect();
    let mut pix2val0 = vec![Real::zero(); img_shape.0 * img_shape.1 * num_vdim];
    interpolate(
        img_shape,
        &pix2tri,
        tri2vtx.as_chunks::<3>().0,
        vtx2xyz0.as_chunks::<3>().0,
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
    let mut dldw_vtx2xyz = vec![Real::zero(); vtx2xyz0.len()];
    let mut dldw_vtx2val = vec![Real::zero(); vtx2val0.len()];
    interpolate_bwd(
        img_shape,
        &pix2tri,
        tri2vtx.as_chunks::<3>().0,
        vtx2xyz0.as_chunks::<3>().0,
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
        for i_vtx in 0..vtx2xyz0.len() / 3 {
            for i_dim in 0..3 {
                let mut vtx2xyz1 = vtx2xyz0.clone();
                vtx2xyz1[i_vtx * 3 + i_dim] += eps;
                let mut pix2val1 = vec![Real::zero(); img_shape.0 * img_shape.1 * num_vdim];
                interpolate(
                    img_shape,
                    &pix2tri,
                    tri2vtx.as_chunks::<3>().0,
                    vtx2xyz1.as_chunks::<3>().0,
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
        for i_vtx in 0..vtx2xyz0.len() / 3 {
            for i_vdim in 0..num_vdim {
                let mut vtx2val1 = vtx2val0.clone();
                vtx2val1[i_vtx * num_vdim + i_vdim] += eps;
                let mut pix2val1 = vec![Real::zero(); img_shape.0 * img_shape.1 * num_vdim];
                interpolate(
                    img_shape,
                    &pix2tri,
                    tri2vtx.as_chunks::<3>().0,
                    vtx2xyz0.as_chunks::<3>().0,
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
            max_difference / max_signal < 1.0e-7,
            "{}",
            max_difference / max_signal
        );
    }
}
