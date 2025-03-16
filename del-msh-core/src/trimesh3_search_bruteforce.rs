//! methods for query computation on 3D triangle mesh

pub fn first_intersection_ray<Index, Real>(
    ray_org: &[Real; 3],
    ray_dir: &[Real; 3],
    tri2vtx: &[Index],
    vtx2xyz: &[Real],
) -> Option<(Real, Index)>
where
    Index: num_traits::PrimInt + num_traits::AsPrimitive<usize>,
    usize: num_traits::AsPrimitive<Index>,
    Real: num_traits::Float,
{
    use num_traits::AsPrimitive;
    let mut hit_pos = Vec::<(Real, Index)>::new();
    for i_tri in 0..tri2vtx.len() / 3 {
        let Some(t) = crate::trimesh3::to_tri3(tri2vtx, vtx2xyz, i_tri)
            .intersection_against_ray(ray_org, ray_dir)
        else {
            continue;
        };
        hit_pos.push((t, i_tri.as_()));
    }
    if hit_pos.is_empty() {
        return None;
    }
    hit_pos.sort_by(|a, b| a.partial_cmp(b).unwrap());
    /*
            //let t = hit_pos[0].0;
    let a = [
        t * ray_dir[0] + ray_org[0],
        t * ray_dir[1] + ray_org[1],
        t * ray_dir[2] + ray_org[2],
    ];
     */
    Some(hit_pos[0])
}

#[allow(clippy::identity_op)]
fn triangles_in_sphere(
    pos: [f32; 3],
    rad: f32,
    itri0: usize,
    vtx2xyz: &[f32],
    tri2vtx: &[usize],
    tri2adjtri: &[usize],
) -> Vec<usize> {
    use crate::vtx2xyz::to_vec3;
    use del_geo_core::vec3;
    let mut res = Vec::<usize>::new();
    let mut searched = std::collections::BTreeSet::<usize>::new();
    let mut next0 = Vec::<usize>::new();
    next0.push(itri0);
    while let Some(iel0) = next0.pop() {
        if searched.contains(&iel0) {
            continue;
        } // already studied
        searched.insert(iel0);
        let dist_min = {
            let i0 = tri2vtx[iel0 * 3 + 0];
            let i1 = tri2vtx[iel0 * 3 + 1];
            let i2 = tri2vtx[iel0 * 3 + 2];
            let (pn, _r0, _r1) = del_geo_core::tri3::nearest_to_point3(
                to_vec3(vtx2xyz, i0),
                to_vec3(vtx2xyz, i1),
                to_vec3(vtx2xyz, i2),
                &pos,
            );
            vec3::distance(&pn, &pos)
        };
        if dist_min > rad {
            continue;
        }
        res.push(iel0);
        for ie in 0..3 {
            let iel1 = tri2adjtri[iel0 * 3 + ie];
            if iel1 == usize::MAX {
                continue;
            }
            next0.push(iel1);
        }
    }
    res
}

pub fn is_point_inside_sphere(
    smpli: &(usize, f32, f32),
    rad: f32,
    samples: &[(usize, f32, f32)],
    elem2smpl: &std::collections::HashMap<usize, Vec<usize>>,
    vtx2xyz: &[f32],
    tri2vtx: &[usize],
    tri2adjtri: &[usize],
) -> bool {
    use del_geo_core::vec3;
    let pos_i = crate::trimesh::position_from_barycentric_coordinate(
        tri2vtx, vtx2xyz, smpli.0, smpli.1, smpli.2,
    );
    let indexes_tri = triangles_in_sphere(pos_i, rad, smpli.0, vtx2xyz, tri2vtx, tri2adjtri);
    for idx_tri in indexes_tri.iter() {
        if !elem2smpl.contains_key(idx_tri) {
            continue;
        }
        for &j_smpl in elem2smpl[idx_tri].iter() {
            let smpl_j = samples[j_smpl];
            let pos_j = crate::trimesh::position_from_barycentric_coordinate(
                tri2vtx, vtx2xyz, smpl_j.0, smpl_j.1, smpl_j.2,
            );
            let dist = vec3::distance(&pos_i, &pos_j);
            if dist < rad {
                return true;
            }
        }
    }
    false
}
