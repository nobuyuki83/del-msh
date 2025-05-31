//! methods related to line mesh topology

use num_traits::AsPrimitive;

/// making vertex indexes list of line mesh from vertex surrounding vertex
/// * `vtx2idx` - vertex to index list
/// * `idx2vtx` - index to vertex list
pub fn from_vtx2vtx<Index>(vtx2idx: &[Index], idx2vtx: &[Index]) -> Vec<Index>
where
    Index: AsPrimitive<usize>,
    usize: AsPrimitive<Index>,
{
    let mut line2vtx = Vec::<Index>::with_capacity(idx2vtx.len() * 2);
    for i_vtx in 0..vtx2idx.len() - 1 {
        let idx0 = vtx2idx[i_vtx].as_();
        let idx1 = vtx2idx[i_vtx + 1].as_();
        for &j_vtx in &idx2vtx[idx0..idx1] {
            line2vtx.push(i_vtx.as_());
            line2vtx.push(j_vtx);
        }
    }
    line2vtx
}

pub fn from_uniform_mesh_with_specific_edges<Index>(
    elem2vtx: &[Index],
    num_node: usize,
    edge2node: &[usize],
    num_vtx: usize,
) -> Vec<Index>
where
    Index: num_traits::PrimInt + std::ops::AddAssign + AsPrimitive<usize>,
    usize: AsPrimitive<Index>,
{
    let vtx2elem = crate::vtx2elem::from_uniform_mesh::<Index>(elem2vtx, num_node, num_vtx);
    let vtx2vtx = crate::vtx2vtx::from_specific_edges_of_uniform_mesh::<Index>(
        elem2vtx,
        num_node,
        edge2node,
        &vtx2elem.0,
        &vtx2elem.1,
        false,
    );
    from_vtx2vtx(&vtx2vtx.0, &vtx2vtx.1)
}

pub fn from_triangle_mesh<INDEX>(tri2vtx: &[INDEX], num_vtx: usize) -> Vec<INDEX>
where
    INDEX: num_traits::PrimInt + std::ops::AddAssign + num_traits::AsPrimitive<usize>,
    usize: AsPrimitive<INDEX>,
{
    from_uniform_mesh_with_specific_edges(tri2vtx, 3, &[0, 1, 1, 2, 2, 0], num_vtx)
}

/// generate line mesh from edges of mesh with polygonal elements
/// The polygonal is a mixture of triangle, quadrilateral, pentagon mesh
/// * `num_vtx` - number of vertex
pub fn from_polygon_mesh(elem2idx: &[usize], idx2vtx: &[usize], num_vtx: usize) -> Vec<usize> {
    let vtx2elem = crate::vtx2elem::from_polygon_mesh(elem2idx, idx2vtx, num_vtx);
    let vtx2vtx = crate::vtx2vtx::from_polygon_mesh_edges_with_vtx2elem(
        elem2idx,
        idx2vtx,
        &vtx2elem.0,
        &vtx2elem.1,
        false,
    );
    from_vtx2vtx(&vtx2vtx.0, &vtx2vtx.1)
}

pub fn from_polyloop(num_vtx: usize) -> Vec<usize> {
    let mut edge2vtx = Vec::<usize>::with_capacity(num_vtx * 2);
    for i in 0..num_vtx {
        edge2vtx.push(i);
        edge2vtx.push((i + 1) % num_vtx);
    }
    edge2vtx
}

// -----------

pub fn contour_for_triangle_mesh<INDEX>(
    tri2vtx: &[INDEX],
    vtx2xyz: &[f32],
    transform_world2ndc: &[f32; 16],
    edge2vtx: &[INDEX],
    edge2tri: &[INDEX],
) -> Vec<INDEX>
where
    INDEX: num_traits::PrimInt + num_traits::AsPrimitive<usize> + std::fmt::Display,
    usize: AsPrimitive<INDEX>,
{
    use del_geo_core::{mat4_col_major, vec3};
    let num_tri = tri2vtx.len() / 3;
    let transform_ndc2world = mat4_col_major::try_inverse(transform_world2ndc).unwrap();
    let mut edge2vtx_contour: Vec<INDEX> = vec![];
    for (i_edge, node2vtx) in edge2vtx.chunks(2).enumerate() {
        let (i0_vtx, i1_vtx) = (node2vtx[0].as_(), node2vtx[1].as_());
        let pos_mid: [f32; 3] =
            std::array::from_fn(|i| (vtx2xyz[i0_vtx * 3 + i] + vtx2xyz[i1_vtx * 3 + i]) * 0.5);
        let (_ray_org, ray_dir) = mat4_col_major::ray_from_transform_world2ndc(
            transform_world2ndc,
            &pos_mid,
            &transform_ndc2world,
        );
        // -----
        let i0_tri = edge2tri[i_edge * 2];
        let i1_tri = edge2tri[i_edge * 2 + 1];
        assert!(i0_tri.as_() < num_tri, "{} {}", i0_tri, tri2vtx.len() / 3);
        assert!(i1_tri.as_() < num_tri, "{} {}", i1_tri, tri2vtx.len() / 3);
        let nrm0_world = crate::trimesh3::to_tri3(tri2vtx, vtx2xyz, i0_tri.as_()).unit_normal();
        let nrm1_world = crate::trimesh3::to_tri3(tri2vtx, vtx2xyz, i1_tri.as_()).unit_normal();
        let flg0 = vec3::dot(&nrm0_world, &ray_dir) > 0.;
        let flg1 = vec3::dot(&nrm1_world, &ray_dir) > 0.;
        if flg0 == flg1 {
            continue;
        }
        edge2vtx_contour.push(i0_vtx.as_());
        edge2vtx_contour.push(i1_vtx.as_());
    }
    edge2vtx_contour
}

pub fn occluding_contour_for_triangle_mesh(
    tri2vtx: &[usize],
    vtx2xyz: &[f32],
    transform_world2ndc: &[f32; 16],
    edge2vtx: &[usize],
    edge2tri: &[usize],
    bvhnodes: &[usize],
    bvhnode2aabb: &[f32],
) -> Vec<usize> {
    use del_geo_core::{mat4_col_major, vec3};
    let transform_ndc2world = mat4_col_major::try_inverse(transform_world2ndc).unwrap();
    let mut edge2vtx_contour = vec![];
    for (i_edge, node2vtx) in edge2vtx.chunks(2).enumerate() {
        let (i0_vtx, i1_vtx) = (node2vtx[0], node2vtx[1]);
        let pos_mid: [f32; 3] =
            std::array::from_fn(|i| (vtx2xyz[i0_vtx * 3 + i] + vtx2xyz[i1_vtx * 3 + i]) * 0.5);
        let (_ray_org, ray_dir) = mat4_col_major::ray_from_transform_world2ndc(
            transform_world2ndc,
            &pos_mid,
            &transform_ndc2world,
        );
        // -----
        let i0_tri = edge2tri[i_edge * 2];
        let i1_tri = edge2tri[i_edge * 2 + 1];
        assert!(
            i0_tri < tri2vtx.len() / 3,
            "{} {}",
            i0_tri,
            tri2vtx.len() / 3
        );
        assert!(
            i1_tri < tri2vtx.len() / 3,
            "{} {}",
            i1_tri,
            tri2vtx.len() / 3
        );
        let nrm0_world = crate::trimesh3::to_tri3(tri2vtx, vtx2xyz, i0_tri).unit_normal();
        let nrm1_world = crate::trimesh3::to_tri3(tri2vtx, vtx2xyz, i1_tri).unit_normal();
        {
            let flg0 = vec3::dot(&nrm0_world, &ray_dir) > 0.;
            let flg1 = vec3::dot(&nrm1_world, &ray_dir) > 0.;
            if flg0 == flg1 {
                continue;
            }
        }
        let ray_org = {
            let nrm = vec3::normalize(&vec3::add(&nrm0_world, &nrm1_world));
            vec3::axpy(0.001, &nrm, &pos_mid)
        };
        let res = crate::search_bvh3::first_intersection_ray(
            &ray_org,
            &ray_dir,
            &crate::search_bvh3::TriMeshWithBvh {
                bvhnodes,
                bvhnode2aabb,
                tri2vtx,
                vtx2xyz,
            },
            0,
            f32::MAX,
        );
        if res.is_some() {
            continue;
        }
        edge2vtx_contour.push(i0_vtx);
        edge2vtx_contour.push(i1_vtx);
    }
    edge2vtx_contour
}

pub fn silhouette_for_triangle_mesh(
    tri2vtx: &[usize],
    vtx2xyz: &[f32],
    transform_world2ndc: &[f32; 16],
    edge2vtx: &[usize],
    edge2tri: &[usize],
    bvhnodes: &[usize],
    bvhnode2aabb: &[f32],
) -> Vec<usize> {
    use del_geo_core::{mat4_col_major, vec3};
    let transform_ndc2world = mat4_col_major::try_inverse(transform_world2ndc).unwrap();
    let mut edge2vtx_contour = vec![];
    for (i_edge, node2vtx) in edge2vtx.chunks(2).enumerate() {
        let (i0_vtx, i1_vtx) = (node2vtx[0], node2vtx[1]);
        let pos_mid: [f32; 3] =
            std::array::from_fn(|i| (vtx2xyz[i0_vtx * 3 + i] + vtx2xyz[i1_vtx * 3 + i]) * 0.5);
        let (_ray_org, ray_dir) = mat4_col_major::ray_from_transform_world2ndc(
            transform_world2ndc,
            &pos_mid,
            &transform_ndc2world,
        );
        // -------
        let i0_tri = edge2tri[i_edge * 2];
        let i1_tri = edge2tri[i_edge * 2 + 1];
        assert!(
            i0_tri < tri2vtx.len() / 3,
            "{} {}",
            i0_tri,
            tri2vtx.len() / 3
        );
        assert!(
            i1_tri < tri2vtx.len() / 3,
            "{} {}",
            i1_tri,
            tri2vtx.len() / 3
        );
        let nrm0_world = crate::trimesh3::to_tri3(tri2vtx, vtx2xyz, i0_tri).unit_normal();
        let nrm1_world = crate::trimesh3::to_tri3(tri2vtx, vtx2xyz, i1_tri).unit_normal();
        // ----------
        {
            let flg0 = vec3::dot(&nrm0_world, &ray_dir) > 0.;
            let flg1 = vec3::dot(&nrm1_world, &ray_dir) > 0.;
            if flg0 == flg1 {
                continue;
            }
        }
        let ray_org = {
            let nrm = vec3::normalize(&vec3::add(&nrm0_world, &nrm1_world));
            vec3::axpy(0.001, &nrm, &pos_mid)
        };
        let mut res: Vec<(f32, usize)> = vec![];
        crate::search_bvh3::intersections_line(
            &mut res,
            &ray_org,
            &ray_dir,
            &crate::search_bvh3::TriMeshWithBvh {
                bvhnodes,
                bvhnode2aabb,
                tri2vtx,
                vtx2xyz,
            },
            0,
        );
        if !res.is_empty() {
            continue;
        }
        edge2vtx_contour.push(i0_vtx);
        edge2vtx_contour.push(i1_vtx);
    }
    edge2vtx_contour
}

#[test]
pub fn test_contour() {
    let (tri2vtx, vtx2xyz)
        // = crate::trimesh3_primitive::sphere_yup::<usize, f32>(1., 32, 32);
        = crate::trimesh3_primitive::torus_zup(2.0, 0.5, 32, 32);
    let dir = del_geo_core::vec3::normalize(&[1f32, 1.0, 0.1]);
    let transform_world2ndc = {
        let ez = dir;
        let (ex, ey) = del_geo_core::vec3::basis_xy_from_basis_z(&ez);
        let m3 = del_geo_core::mat3_col_major::from_columns(&ex, &ey, &ez);
        let t = del_geo_core::mat4_col_major::from_mat3_col_major_adding_w(&m3);
        del_geo_core::mat4_col_major::transpose(&t)
    };
    //
    let bvhnodes = crate::bvhnodes_morton::from_triangle_mesh(&tri2vtx, &vtx2xyz, 3);
    let bvhnode2aabb = crate::bvhnode2aabb3::from_uniform_mesh_with_bvh(
        0,
        &bvhnodes,
        Some((&tri2vtx, 3)),
        &vtx2xyz,
        None,
    );
    let edge2vtx = crate::edge2vtx::from_triangle_mesh(tri2vtx.as_slice(), vtx2xyz.len() / 3);
    let edge2tri =
        crate::edge2elem::from_edge2vtx_of_tri2vtx(&edge2vtx, &tri2vtx, vtx2xyz.len() / 3);

    {
        let edge2vtx_contour = occluding_contour_for_triangle_mesh(
            &tri2vtx,
            &vtx2xyz,
            &transform_world2ndc,
            &edge2vtx,
            &edge2tri,
            &bvhnodes,
            &bvhnode2aabb,
        );
        crate::io_obj::save_edge2vtx_vtx2xyz(
            "../target/edge2vtx_countour.obj",
            &edge2vtx_contour,
            &vtx2xyz,
            3,
        )
        .unwrap();
    }
    {
        let edge2vtx_contour = silhouette_for_triangle_mesh(
            &tri2vtx,
            &vtx2xyz,
            &transform_world2ndc,
            &edge2vtx,
            &edge2tri,
            &bvhnodes,
            &bvhnode2aabb,
        );
        crate::io_obj::save_edge2vtx_vtx2xyz(
            "../target/edge2vtx_silhouette.obj",
            &edge2vtx_contour,
            &vtx2xyz,
            3,
        )
        .unwrap();
    }
    crate::io_obj::save_tri2vtx_vtx2xyz("../target/edge2vtx_trimsh.obj", &tri2vtx, &vtx2xyz, 3)
        .unwrap();
}
