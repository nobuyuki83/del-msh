pub trait ScalarRender<T> {
    fn fwd(
        &self,
        bc: &[T; 3],
        i_tri: u32,
        tri2vtx: &[u32],
        vtx2xyz: &[T],
        transform_world2ndc: &[T; 16],
    ) -> T;

    #[allow(clippy::too_many_arguments)]
    fn bwd(
        &self,
        dldw_val: T,
        p0: &[T; 3],
        p1: &[T; 3],
        p2: &[T; 3],
        ray_org: &[T; 3],
        ray_dir: &[T; 3],
        transform_world2ndc: &[T; 16],
    ) -> ([T; 3], [T; 3], [T; 3]);
}

#[allow(clippy::too_many_arguments)]
pub fn bwd_continuous<T: ScalarRender<f32>>(
    pix2tri: &[u32],
    tri2vtx: &[u32],
    vtx2xyz: &[f32],
    dldw_pix2val: &[f32],
    transform_ndc2world: &[f32; 16],
    img_shape: (usize, usize),
    dldw_vtx2xyz: &mut [f32],
    mode: &T,
) {
    let transform_world2ndc =
        del_geo_core::mat4_col_major::try_inverse_with_pivot(transform_ndc2world).unwrap();
    for (i_pix, &i_tri) in pix2tri.iter().enumerate() {
        if i_tri == u32::MAX {
            continue;
        }
        let i_w = i_pix % img_shape.0;
        let i_h = i_pix / img_shape.0;
        let (ray_org, ray_dir) =
            del_geo_core::mat4_col_major::ray_from_transform_ndc2world_and_pixel_coordinates(
                (i_w as f32 + 0.5, i_h as f32 + 0.5),
                &(img_shape.0 as f32, img_shape.1 as f32),
                transform_ndc2world,
            );
        let i_tri = i_tri as usize;
        let i0 = tri2vtx[i_tri * 3] as usize;
        let i1 = tri2vtx[i_tri * 3 + 1] as usize;
        let i2 = tri2vtx[i_tri * 3 + 2] as usize;
        let p0 = arrayref::array_ref![vtx2xyz, i0 * 3, 3];
        let p1 = arrayref::array_ref![vtx2xyz, i1 * 3, 3];
        let p2 = arrayref::array_ref![vtx2xyz, i2 * 3, 3];
        let dldw_val = dldw_pix2val[i_pix];
        let (dp0, dp1, dp2) = mode.bwd(
            dldw_val,
            p0,
            p1,
            p2,
            &ray_org,
            &ray_dir,
            &transform_world2ndc,
        );
        use del_geo_core::vec3::Vec3;
        arrayref::array_mut_ref![dldw_vtx2xyz, i0 * 3, 3].add_in_place(&dp0);
        arrayref::array_mut_ref![dldw_vtx2xyz, i1 * 3, 3].add_in_place(&dp1);
        arrayref::array_mut_ref![dldw_vtx2xyz, i2 * 3, 3].add_in_place(&dp2);
    }
}

pub fn fwd_continuous<T: ScalarRender<f32>>(
    pix2tri: &[u32],
    img_shape: (usize, usize),
    tri2vtx: &[u32],
    vtx2xyz: &[f32],
    transform_ndc2world: &[f32; 16],
    model: &T,
) -> Vec<f32> {
    let transform_world2ndc =
        del_geo_core::mat4_col_major::try_inverse_with_pivot(transform_ndc2world).unwrap();
    let mut pix2vin = vec![0.; pix2tri.len()];
    for (i_pix, &i_tri) in pix2tri.iter().enumerate() {
        if i_tri == u32::MAX {
            continue;
        }
        let i_w = i_pix % img_shape.0;
        let i_h = i_pix / img_shape.0;
        let (ray_org, ray_dir) =
            del_geo_core::mat4_col_major::ray_from_transform_ndc2world_and_pixel_coordinates(
                (i_w as f32 + 0.5, i_h as f32 + 0.5),
                &(img_shape.0 as f32, img_shape.1 as f32),
                transform_ndc2world,
            );
        let Some((_t, bc)) = crate::trimesh3::to_tri3(tri2vtx, vtx2xyz, i_tri as usize)
            .intersection_against_ray(&ray_org, &ray_dir)
        else {
            unreachable!()
        };
        pix2vin[i_pix] = model.fwd(&bc, i_tri, tri2vtx, vtx2xyz, &transform_world2ndc);
    }
    pix2vin
}

pub fn multi_sample<T, F, R>(
    tri2vtx: &[u32],
    vtx2xyz: &[f32],
    transform_world2ndc: &[f32; 16],
    img_shape: (usize, usize),
    num_sample: usize,
    mode: &T,
    rng_factory: F,
) -> Vec<f32>
where
    T: ScalarRender<f32> + Sync,
    F: Fn(usize) -> R + Sync,
    R: rand::Rng,
{
    use rand::RngExt;
    // let transform_ndc2world = del_geo_core::mat4_col_major::from_identity();
    let transform_ndc2world =
        del_geo_core::mat4_col_major::try_inverse_with_pivot(transform_world2ndc).unwrap();
    let bvhnodes = crate::bvhnodes_morton::from_triangle_mesh(tri2vtx, vtx2xyz, 3);
    let bvhnode2aabb =
        crate::bvhnode2aabb3::from_uniform_mesh_with_bvh(0, &bvhnodes, tri2vtx, 3, vtx2xyz, None);
    let fn_pix2val = |i_pix: usize| -> f32 {
        let mut rng = rng_factory(i_pix);
        let i_h = i_pix / img_shape.0;
        let i_w = i_pix - i_h * img_shape.0;
        //
        let mut sum = 0.0f32;
        for _itr in 0..num_sample {
            let x_offset = rng.random_range(0.0..1.);
            let y_offset = rng.random_range(0.0..1.0);
            let (ray_org, ray_dir) =
                del_geo_core::mat4_col_major::ray_from_transform_ndc2world_and_pixel_coordinates(
                    (i_w as f32 + x_offset, i_h as f32 + y_offset),
                    &(img_shape.0 as f32, img_shape.1 as f32),
                    &transform_ndc2world,
                );
            if let Some((_t, i_tri, bc)) = crate::search_bvh3::first_intersection_ray(
                &ray_org,
                &ray_dir,
                &crate::search_bvh3::TriMeshWithBvh {
                    tri2vtx,
                    vtx2xyz,
                    bvhnodes: &bvhnodes,
                    bvhnode2aabb: &bvhnode2aabb,
                },
                0,
                f32::INFINITY,
            ) {
                sum += mode.fwd(&bc, i_tri as u32, tri2vtx, vtx2xyz, transform_world2ndc);
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
