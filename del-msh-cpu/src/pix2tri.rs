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
        if let Some((_t, i_tri)) = crate::search_bvh3::first_intersection_ray(
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
