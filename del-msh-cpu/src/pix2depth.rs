pub struct Depth;

impl<T> crate::trimesh3_raycast::ScalarRender<T> for Depth
where
    T: num_traits::Float,
{
    fn fwd(
        &self,
        bc: &[T; 3],
        i_tri: u32,
        tri2vtx: &[u32],
        vtx2xyz: &[T],
        transform_world2ndc: &[T; 16],
    ) -> T {
        if i_tri == u32::MAX {
            return T::zero();
        };
        let q = crate::trimesh3::to_tri3(tri2vtx, vtx2xyz, i_tri as usize)
            .position_from_barycentric_coordinates(bc[0], bc[1]);
        let ndc =
            del_geo_core::mat4_col_major::transform_homogeneous(transform_world2ndc, &q).unwrap();
        let one = T::one();
        let half = one / (one + one);
        (ndc[2] + one) * half
    }

    fn bwd(
        &self,
        dldw_depth: T,
        p0: &[T; 3],
        p1: &[T; 3],
        p2: &[T; 3],
        ray_org: &[T; 3],
        ray_dir: &[T; 3],
        transform_world2ndc: &[T; 16],
    ) -> ([T; 3], [T; 3], [T; 3]) {
        let zero = T::zero();
        let one = T::one();
        let half = one / (one + one);
        let dldw_ndc = [zero, zero, half * dldw_depth];
        let (_t, bc) =
            del_geo_core::tri3::intersection_against_line(p0, p1, p2, ray_org, ray_dir).unwrap();
        let q = del_geo_core::tri3::position_from_barycentric_coords(p0, p1, p2, &bc);
        let dndcdq = del_geo_core::mat4_col_major::jacobian_transform(transform_world2ndc, &q);
        let dndcdq_t = del_geo_core::mat3_col_major::transpose(&dndcdq);
        let dldw_q = del_geo_core::mat3_col_major::mult_vec(&dndcdq_t, &dldw_ndc);
        let dldw_t = del_geo_core::vec3::dot(&ray_dir, &dldw_q);
        let (_t, _u, _v, dldw_p0, dldw_p1, dldw_p2) =
            del_geo_core::tri3::intersection_against_line_bwd_wrt_tri(
                p0, p1, p2, ray_org, ray_dir, dldw_t, zero, zero,
            );
        (dldw_p0, dldw_p1, dldw_p2)
    }
}

#[test]
fn test_hoge() {
    use del_geo_core::vec3::Vec3;
    let p0: [[f64; 3]; 3] = [[-13., -5., 8.], [14., -5., 8.], [1., 3., -3.]];
    let ray_org = [8., 11., 10.];
    let ray_dir = [1., 0., 2.].sub(&ray_org);
    let transform_world2ndc = [
        4., -1., 5., 1., 1., 3., 9., 3., 1., 4., 2., 2., -1., -2., 3., 3.,
    ];
    use crate::trimesh3_raycast::ScalarRender;
    let depth_layer = Depth;
    let (_t0, bc0) =
        del_geo_core::tri3::intersection_against_line(&p0[0], &p0[1], &p0[2], &ray_org, &ray_dir)
            .unwrap();
    let depth0 = depth_layer.fwd(&bc0, 0, &[0, 1, 2], p0.as_flattened(), &transform_world2ndc);
    let dldw_depth = 1.3;
    let l0 = depth0 * dldw_depth;
    let (dldw_p0, dldw_p1, dldw_p2) = depth_layer.bwd(
        dldw_depth,
        &p0[0],
        &p0[1],
        &p0[2],
        &ray_org,
        &ray_dir,
        &transform_world2ndc,
    );
    let eps = 1.0e-5;
    for (i_node, i_dim) in itertools::iproduct!(0..3, 0..3) {
        let p1 = {
            let mut p1 = p0;
            p1[i_node][i_dim] += eps;
            p1
        };
        let (_t, bc1) = del_geo_core::tri3::intersection_against_line(
            &p1[0], &p1[1], &p1[2], &ray_org, &ray_dir,
        )
        .unwrap();
        let depth1 = depth_layer.fwd(&bc1, 0, &[0, 1, 2], p1.as_flattened(), &transform_world2ndc);
        let l1 = depth1 * dldw_depth;
        let num_diff = (l1 - l0) / eps;
        let ana_diff = match i_node {
            0 => dldw_p0[i_dim],
            1 => dldw_p1[i_dim],
            2 => dldw_p2[i_dim],
            _ => unreachable!(),
        };
        println!("{i_node} {i_dim} {num_diff}, {ana_diff}");
        assert!((num_diff - ana_diff).abs() < 1.0e-5);
    }
}

pub fn pix2depth_from_pix2tri(
    pix2depth: &mut [f32],
    pix2tri: &[u32],
    tri2vtx: &[u32],
    vtx2xyz: &[f32],
    img_shape: (usize, usize), // (width, height)
    transform_ndc2world: &[f32; 16],
) {
    let transform_world2ndc =
        del_geo_core::mat4_col_major::try_inverse_with_pivot(transform_ndc2world).unwrap();
    let fn_pix2depth = |i_pix: usize| -> f32 {
        let i_tri = pix2tri[i_pix];
        if i_tri == u32::MAX {
            return 0f32;
        }
        let i_w = i_pix % img_shape.0;
        let i_h = i_pix / img_shape.0;
        let (ray_org, ray_dir) =
            del_geo_core::mat4_col_major::ray_from_transform_ndc2world_and_pixel_coordinates(
                (i_w as f32 + 0.5, i_h as f32 + 0.5),
                &(img_shape.0 as f32, img_shape.1 as f32),
                transform_ndc2world,
            );
        let tri = crate::trimesh3::to_tri3(tri2vtx, vtx2xyz, i_tri as usize);
        let (coeff, _bc) = del_geo_core::tri3::intersection_against_line(
            tri.p0, tri.p1, tri.p2, &ray_org, &ray_dir,
        )
        .unwrap();
        let pos_world = del_geo_core::vec3::axpy(coeff, &ray_dir, &ray_org);
        let pos_ndc =
            del_geo_core::mat4_col_major::transform_homogeneous(&transform_world2ndc, &pos_world)
                .unwrap();
        (pos_ndc[2] + 1f32) * 0.5f32
    };
    use rayon::prelude::*;
    pix2depth
        .par_iter_mut()
        .enumerate()
        .for_each(|(i_pix, depth)| *depth = fn_pix2depth(i_pix));
}

pub fn render_depth_bvh(
    image_size: (usize, usize),
    pix2depth: &mut [f32],
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
                del_geo_core::mat4_col_major::ray_from_transform_ndc2world_and_pixel_coordinates(
                    (iw as f32 + 0.5, ih as f32 + 0.5),
                    &(image_size.0 as f32, image_size.1 as f32),
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
            pix2depth[ih * width + iw] = depth_ndc;
        }
    }
}

#[test]
fn test_depthmap() {
    for i_case in 0..3 {
        let (tri2vtx, vtx2xyz) = match i_case {
            0 => {
                let vtx2xyz_polyline = crate::polyline3::helix(100, 0.05, 0.7, 0.4);
                use std::f32::consts::PI;
                let rot_y = del_geo_core::mat4_col_major::from_rot_y(PI * 0.51);
                let rot_x = del_geo_core::mat4_col_major::from_rot_x(-PI * 0.25);
                let transform = del_geo_core::mat4_col_major::mult_mat_col_major(&rot_x, &rot_y);
                let vtx2xyz_polyline =
                    crate::vtx2xyz::transform_homogeneous(&vtx2xyz_polyline, &transform);
                crate::polyline3::to_trimesh3_capsule(&vtx2xyz_polyline, 32, 32, 0.05)
            }
            1 => {
                let vtx2xyz_polyline = del_geo_core::bezier_cubic::sample_uniform_param(
                    100,
                    &[0.9, 0.0, -0.2],
                    &[-3.0, 0.9, -0.2],
                    &[0.9, -3.0, 0.2],
                    &[0.0, 0.9, 0.2],
                    true,
                    true,
                );
                use slice_of_array::SliceFlatExt;
                let vtx2xyz_polyline = vtx2xyz_polyline.flat().to_owned();
                crate::polyline3::to_trimesh3_capsule(&vtx2xyz_polyline, 32, 32, 0.05)
            }
            2 => {
                let (tri2vtx, vtx2xyz) = crate::trimesh3_primitive::torus_zup(0.8, 0.05, 32, 32);
                let transform =
                    del_geo_core::mat4_col_major::from_rot_x(std::f32::consts::PI / 12.0);
                let vtx2xyz = crate::vtx2xyz::transform_homogeneous(&vtx2xyz, &transform);
                (tri2vtx, vtx2xyz)
            }
            _ => unreachable!(),
        };
        crate::io_wavefront_obj::save_tri2vtx_vtx2xyz(
            format!("../target/trimesh3_raycast_mesh{i_case}.obj"),
            &tri2vtx,
            &vtx2xyz,
            3,
        )
        .unwrap();
        let aabb3 = crate::vtx2xyz::aabb3(&vtx2xyz, 0.);
        dbg!(aabb3);
        let bvhnodes = crate::bvhnodes_morton::from_triangle_mesh(&tri2vtx, &vtx2xyz, 3);
        let bvhnode2aabb = crate::bvhnode2aabb3::from_uniform_mesh_with_bvh(
            0, &bvhnodes, &tri2vtx, 3, &vtx2xyz, None,
        );
        let img_shape = (300, 300);
        let mut pix2depth = vec![0f32; img_shape.0 * img_shape.1];
        let transform_ndc2world = del_geo_core::mat4_col_major::from_identity();
        dbg!(&transform_ndc2world);
        render_depth_bvh(
            img_shape,
            &mut pix2depth,
            &transform_ndc2world,
            &tri2vtx,
            &vtx2xyz,
            &bvhnodes,
            &bvhnode2aabb,
        );
        pix2depth.iter_mut().for_each(|v| *v = (*v) + 0.0);
        del_canvas::write_png_from_float_image(
            format!("../target/trimesh3_raycast_depth_{i_case}.png"),
            img_shape,
            1,
            &pix2depth,
        )
        .unwrap();
        let (quad2vtx, vtx2xyz) =
            crate::grid2::to_quadmesh3_hightmap(img_shape, &pix2depth, 1.0 / img_shape.0 as f32);
        crate::io_wavefront_obj::save_quad2vtx_vtx2xyz(
            format!("../target/trimesh3_raycast_hightmap_{i_case}.obj"),
            &quad2vtx,
            &vtx2xyz,
            3,
        )
        .unwrap();
    }
}
