//! method for 3D Bounding Volume Hierarchy

use num_traits::{AsPrimitive, PrimInt};

pub struct TriMeshWithBvh<'a, Index> {
    pub tri2vtx: &'a [Index],
    pub vtx2xyz: &'a [f32],
    pub bvhnodes: &'a [Index],
    pub bvhnode2aabb: &'a [f32],
}

pub fn intersections_ray<Index>(
    hits: &mut Vec<(f32, usize)>,
    ray_org: &[f32; 3],
    ray_dir: &[f32; 3],
    trimesh3: &TriMeshWithBvh<Index>,
    i_bvhnode: usize,
) where
    Index: PrimInt + AsPrimitive<usize>,
{
    if del_geo_core::aabb3::from_aabbs(trimesh3.bvhnode2aabb, i_bvhnode)
        .intersections_against_ray(ray_org, ray_dir)
        .is_none()
    {
        return;
    }
    assert_eq!(trimesh3.bvhnodes.len() / 3, trimesh3.bvhnode2aabb.len() / 6);
    if trimesh3.bvhnodes[i_bvhnode * 3 + 2] == Index::max_value() {
        // leaf node
        let i_tri: usize = trimesh3.bvhnodes[i_bvhnode * 3 + 1].as_();
        let Some(t) = crate::trimesh3::to_tri3(trimesh3.tri2vtx, trimesh3.vtx2xyz, i_tri)
            .intersection_against_ray(ray_org, ray_dir)
        else {
            return;
        };
        hits.push((t, i_tri));
        return;
    }
    intersections_ray(
        hits,
        ray_org,
        ray_dir,
        trimesh3,
        trimesh3.bvhnodes[i_bvhnode * 3 + 1].as_(),
    );
    intersections_ray(
        hits,
        ray_org,
        ray_dir,
        trimesh3,
        trimesh3.bvhnodes[i_bvhnode * 3 + 2].as_(),
    );
}

pub fn intersections_line<Index>(
    hits: &mut Vec<(f32, usize)>,
    line_org: &[f32; 3],
    line_dir: &[f32; 3],
    trimesh3: &TriMeshWithBvh<Index>,
    i_bvhnode: usize,
) where
    Index: PrimInt + AsPrimitive<usize>,
{
    if del_geo_core::aabb3::from_aabbs(trimesh3.bvhnode2aabb, i_bvhnode)
        .intersections_against_line(line_org, line_dir)
        .is_none()
    {
        return;
    }
    assert_eq!(trimesh3.bvhnodes.len() / 3, trimesh3.bvhnode2aabb.len() / 6);
    if trimesh3.bvhnodes[i_bvhnode * 3 + 2] == Index::max_value() {
        // leaf node
        let i_tri: usize = trimesh3.bvhnodes[i_bvhnode * 3 + 1].as_();
        let Some(t) = crate::trimesh3::to_tri3(trimesh3.tri2vtx, trimesh3.vtx2xyz, i_tri)
            .intersection_against_line(line_org, line_dir)
        else {
            return;
        };
        hits.push((t, i_tri));
        return;
    }
    crate::search_bvh3::intersections_line(
        hits,
        line_org,
        line_dir,
        trimesh3,
        trimesh3.bvhnodes[i_bvhnode * 3 + 1].as_(),
    );
    crate::search_bvh3::intersections_line(
        hits,
        line_org,
        line_dir,
        trimesh3,
        trimesh3.bvhnodes[i_bvhnode * 3 + 2].as_(),
    );
}

/// return the distance to triangle and triangle index
pub fn first_intersection_ray<Index>(
    ray_org: &[f32; 3],
    ray_dir: &[f32; 3],
    trimesh3: &TriMeshWithBvh<Index>,
    i_bvhnode: usize,
    dis: f32,
) -> Option<(f32, usize)>
where
    Index: PrimInt + AsPrimitive<usize>,
{
    assert_eq!(
        trimesh3.bvhnodes.len() / 3,
        trimesh3.tri2vtx.len() / 3 * 2 - 1
    );
    assert_eq!(trimesh3.bvhnodes.len() / 3, trimesh3.bvhnode2aabb.len() / 6);
    // culling the branch
    del_geo_core::aabb3::from_aabbs(trimesh3.bvhnode2aabb, i_bvhnode)
        .intersections_against_ray(ray_org, ray_dir)?;
    if trimesh3.bvhnodes[i_bvhnode * 3 + 2] == Index::max_value() {
        // leaf node
        let i_tri: usize = trimesh3.bvhnodes[i_bvhnode * 3 + 1].as_();
        return crate::trimesh3::to_tri3(trimesh3.tri2vtx, trimesh3.vtx2xyz, i_tri)
            .intersection_against_ray(ray_org, ray_dir)
            .filter(|&t| t < dis)
            .map(|t| (t, i_tri));
    }

    // near branch is checked first. Check which branch (left or right) is near
    let near_is_left = {
        let mut near_is_left = true;
        let t_aabb_left = del_geo_core::aabb3::from_aabbs(
            trimesh3.bvhnode2aabb,
            trimesh3.bvhnodes[i_bvhnode * 3 + 1].as_(),
        )
        .intersections_against_ray(ray_org, ray_dir);
        let t_aabb_right = del_geo_core::aabb3::from_aabbs(
            trimesh3.bvhnode2aabb,
            trimesh3.bvhnodes[i_bvhnode * 3 + 2].as_(),
        )
        .intersections_against_ray(ray_org, ray_dir);
        if let Some(t_aabb_left) = t_aabb_left {
            if let Some(t_aabb_right) = t_aabb_right {
                if t_aabb_right < t_aabb_left {
                    near_is_left = false;
                }
            }
        }
        near_is_left
    };

    // check left branch
    let res_near = {
        let idx_bvhnode_near = if near_is_left {
            i_bvhnode * 3 + 1
        } else {
            i_bvhnode * 3 + 2
        };
        first_intersection_ray(
            ray_org,
            ray_dir,
            trimesh3,
            trimesh3.bvhnodes[idx_bvhnode_near].as_(),
            dis,
        )
    };

    if let Some((t_near, _i_tri_near)) = res_near {
        // if there is a hit branch in the left
        let idx_bvhnode_far = if near_is_left {
            i_bvhnode * 3 + 2
        } else {
            i_bvhnode * 3 + 1
        };
        let i_bvhnode_far: usize = trimesh3.bvhnodes[idx_bvhnode_far].as_(); // right
        let aabb_far = arrayref::array_ref!(trimesh3.bvhnode2aabb, i_bvhnode_far * 6, 6);
        // the intersection point is closer than another bvhnode
        if let Some((t_aabb_far, _)) =
            del_geo_core::aabb::intersections_against_ray(aabb_far, ray_org, ray_dir)
        {
            if t_aabb_far > t_near {
                return res_near;
            }
        }
    }

    let res_far = {
        let idx_bvhnode_far = if near_is_left {
            i_bvhnode * 3 + 2
        } else {
            i_bvhnode * 3 + 1
        };
        first_intersection_ray(
            ray_org,
            ray_dir,
            trimesh3,
            trimesh3.bvhnodes[idx_bvhnode_far].as_(),
            dis,
        )
    };

    if let Some((t_near, i_tri_near)) = res_near {
        if let Some((t_far, i_tri_far)) = res_far {
            if t_near < t_far {
                Some((t_near, i_tri_near))
            } else {
                Some((t_far, i_tri_far))
            }
        } else {
            Some((t_near, i_tri_near))
        }
    } else if let Some((t_far, i_tri_far)) = res_far {
        Some((t_far, i_tri_far))
    } else {
        None
    }
}

/*
/// check if a point alone ray_dir is closer than an aabb
fn is_point_closer(aabb: &[f32; 6], ray_dir: &[f32; 3], t: f32) -> bool {
    // closest projection of aabb along ray
    let mut proj_aabb = 0.;
    for i_dim in 0..3 {
        if ray_dir[i_dim].abs() == 0. {
            continue;
        }

        if ray_dir[i_dim] > 0. {
            // min project
            proj_aabb += aabb[i_dim] * ray_dir[i_dim];
        } else {
            // max project
            proj_aabb += aabb[i_dim + 3] * ray_dir[i_dim];
        }
    }

    let sqlen = del_geo_core::vec3::dot(ray_dir, ray_dir);
    t * sqlen < proj_aabb
}
 */

#[test]
fn test_first_intersection_ray() {
    let (tri2vtx, vtx2xyz) = crate::trimesh3_primitive::sphere_yup::<usize, f32>(1.0, 128, 128);
    let bvhnodes = crate::bvhnodes_morton::from_triangle_mesh(&tri2vtx, &vtx2xyz, 3);
    let bvhnode2aabb = crate::bvhnode2aabb3::from_uniform_mesh_with_bvh(
        0,
        &bvhnodes,
        Some((&tri2vtx, 3)),
        &vtx2xyz,
        None,
    );
    use rand::Rng;
    use rand::SeedableRng;
    let mut reng = rand_chacha::ChaChaRng::seed_from_u64(0u64);
    for _iter in 0..1000 {
        let x = reng.random::<f32>() * 2.0 - 1.;
        let y = reng.random::<f32>() * 2.0 - 1.;
        let ray_dir = [0., 0., -1.];
        let ray_org = [x, y, 1.];
        let res_first = first_intersection_ray(
            &ray_org,
            &ray_dir,
            &TriMeshWithBvh {
                tri2vtx: &tri2vtx,
                vtx2xyz: &vtx2xyz,
                bvhnodes: &bvhnodes,
                bvhnode2aabb: &bvhnode2aabb,
            },
            0,
            f32::INFINITY,
        );
        if let Some((t, _i_tri)) = res_first {
            let p = [
                ray_org[0] + t * ray_dir[0],
                ray_org[1] + t * ray_dir[1],
                ray_org[2] + t * ray_dir[2],
            ];
            assert!(del_geo_core::vec3::norm(&p) < 1.0);
        }
        {
            // intersection all
            let mut hits = Vec::<(f32, usize)>::new();
            intersections_ray::<usize>(
                &mut hits,
                &ray_org,
                &ray_dir,
                &TriMeshWithBvh {
                    tri2vtx: &tri2vtx,
                    vtx2xyz: &vtx2xyz,
                    bvhnodes: &bvhnodes,
                    bvhnode2aabb: &bvhnode2aabb,
                },
                0,
            );
            if let Some((t_first, _i_tri_first)) = res_first {
                assert_eq!(hits.len(), 2);
                assert!(hits[0].0 == t_first || hits[1].0 == t_first);
            } else {
                assert!(hits.is_empty());
            }
        }
    }
}
