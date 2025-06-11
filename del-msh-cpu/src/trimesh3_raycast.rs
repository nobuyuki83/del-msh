use num_traits::AsPrimitive;

pub fn update_pix2tri<Index>(
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
            del_geo_core::mat4_col_major::ray_from_transform_ndc2world_and_pixel_coordinate(
                (i_w, i_h),
                &img_shape,
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

pub fn render_depth_bvh(
    image_size: (usize, usize),
    img_data: &mut [f32],
    transform_ndc2world: &[f32; 16],
    tri2vtx: &[usize],
    vtx2xyz: &[f32],
    bvhnodes: &[usize],
    bvhnode2aabb: &[f32],
) {
    let transform_world2ndc: [f32; 16] =
        del_geo_core::mat4_col_major::try_inverse(transform_ndc2world).unwrap();
    let (width, height) = image_size;
    for ih in 0..height {
        for iw in 0..width {
            let (ray_org, ray_dir) =
                del_geo_core::mat4_col_major::ray_from_transform_ndc2world_and_pixel_coordinate(
                    (iw, ih),
                    &image_size,
                    transform_ndc2world,
                );
            let mut hits = vec![];
            crate::search_bvh3::intersections_ray(
                &mut hits,
                &ray_org,
                &ray_dir,
                &crate::search_bvh3::TriMeshWithBvh {
                    tri2vtx,
                    vtx2xyz,
                    bvhnodes,
                    bvhnode2aabb,
                },
                0,
            );
            hits.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let Some(&(depth, _i_tri)) = hits.first() else {
                continue;
            };
            let pos = del_geo_core::vec3::axpy(depth, &ray_dir, &ray_org);
            let ndc =
                del_geo_core::mat4_col_major::transform_homogeneous(&transform_world2ndc, &pos)
                    .unwrap();
            let depth_ndc = (ndc[2] + 1f32) * 0.5f32;
            img_data[ih * width + iw] = depth_ndc;
        }
    }
}

pub fn render_normalmap_from_pix2tri<INDEX>(
    (img_width, img_height): (usize, usize),
    cam_modelviewd: &[f32; 16],
    tri2vtx: &[INDEX],
    vtx2xyz: &[f32],
    pix2tri: &[INDEX],
) -> Vec<f32>
where
    INDEX: num_traits::PrimInt + AsPrimitive<usize> + Sync + Send,
{
    let mut img = vec![0f32; img_height * img_width * 3];
    for ih in 0..img_height {
        for iw in 0..img_width {
            let i_tri = pix2tri[ih * img_width + iw];
            if i_tri == INDEX::max_value() {
                continue;
            }
            let i_tri: usize = i_tri.as_();
            let tri = crate::trimesh3::to_tri3(tri2vtx, vtx2xyz, i_tri);
            let nrm = tri.normal();
            let nrm = del_geo_core::mat4_col_major::transform_direction(cam_modelviewd, &nrm);
            let unrm = del_geo_core::vec3::normalize(&nrm);
            img[(ih * img_width + iw) * 3] = unrm[0] * 0.5 + 0.5;
            img[(ih * img_width + iw) * 3 + 1] = unrm[1] * 0.5 + 0.5;
            img[(ih * img_width + iw) * 3 + 2] = unrm[2] * 0.5 + 0.5;
        }
    }
    img
}

#[allow(clippy::too_many_arguments)]
pub fn render_texture_from_pix2tri<Index>(
    img_shape: (usize, usize),
    transform_ndc2world: &[f32; 16],
    tri2vtx: &[usize],
    vtx2xyz: &[f32],
    vtx2uv: &[f32],
    pix2tri: &[Index],
    tex_shape: (usize, usize),
    tex_data: &[f32],
    interpolation: &crate::grid2::Interpolation,
) -> Vec<f32>
where
    Index: num_traits::PrimInt + AsPrimitive<usize>,
{
    let (width, height) = img_shape;
    let mut img = vec![0f32; height * width * 3];
    for ih in 0..height {
        for iw in 0..width {
            let (ray_org, ray_dir) =
                del_geo_core::mat4_col_major::ray_from_transform_ndc2world_and_pixel_coordinate(
                    (iw, ih),
                    &img_shape,
                    transform_ndc2world,
                );
            let i_tri = pix2tri[ih * width + iw];
            if i_tri == Index::max_value() {
                continue;
            }
            let i_tri: usize = i_tri.as_();
            let tri = crate::trimesh3::to_tri3(tri2vtx, vtx2xyz, i_tri);
            let Some(a) = tri.intersection_against_ray(&ray_org, &ray_dir) else {
                continue;
            };
            let q = del_geo_core::vec3::axpy(a, &ray_dir, &ray_org);
            let bc = del_geo_core::tri3::to_barycentric_coords(tri.p0, tri.p1, tri.p2, &q);
            let uv0 = arrayref::array_ref!(vtx2uv, tri2vtx[i_tri * 3] * 2, 2);
            let uv1 = arrayref::array_ref!(vtx2uv, tri2vtx[i_tri * 3 + 1] * 2, 2);
            let uv2 = arrayref::array_ref!(vtx2uv, tri2vtx[i_tri * 3 + 2] * 2, 2);
            let uv = [
                uv0[0] * bc[0] + uv1[0] * bc[1] + uv2[0] * bc[2],
                uv0[1] * bc[0] + uv1[1] * bc[1] + uv2[1] * bc[2],
            ];
            let pix = [
                uv[0] * tex_shape.0 as f32,
                (1. - uv[1]) * tex_shape.1 as f32,
            ];
            let res = match interpolation {
                crate::grid2::Interpolation::Nearest => {
                    crate::grid2::nearest_integer_center::<3>(&pix, &tex_shape, tex_data)
                }
                crate::grid2::Interpolation::Bilinear => {
                    crate::grid2::bilinear_integer_center::<3>(&pix, &tex_shape, tex_data)
                }
            };
            img[(ih * width + iw) * 3] = res[0];
            img[(ih * width + iw) * 3 + 1] = res[1];
            img[(ih * width + iw) * 3 + 2] = res[2];
        }
    }
    img
}
