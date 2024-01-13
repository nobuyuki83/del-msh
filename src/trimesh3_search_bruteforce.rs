//! methods for query computation on 3D triangle mesh

#[allow(clippy::identity_op)]
pub fn first_intersection_ray(
    ray_org: &[f32],
    ray_dir: &[f32],
    vtx2xyz: &[f32],
    tri2vtx: &[usize]) -> Option<([f32; 3], usize)> {
    use del_geo::tri3;
    let mut hit_pos = Vec::<(f32, usize)>::new();
    for itri in 0..tri2vtx.len() / 3 {
        let i0 = tri2vtx[itri * 3 + 0];
        let i1 = tri2vtx[itri * 3 + 1];
        let i2 = tri2vtx[itri * 3 + 2];
        let res = tri3::ray_triangle_intersection_(
            ray_org, ray_dir,
            &vtx2xyz[i0 * 3 + 0..i0 * 3 + 3],
            &vtx2xyz[i1 * 3 + 0..i1 * 3 + 3],
            &vtx2xyz[i2 * 3 + 0..i2 * 3 + 3]);
        match res {
            None => { continue; }
            Some(t) => {
                hit_pos.push((t, itri));
            }
        }
    }
    if hit_pos.is_empty() { return None; }
    hit_pos.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let t = hit_pos[0].0;
    let a = [
        t * ray_dir[0] + ray_org[0],
        t * ray_dir[1] + ray_org[1],
        t * ray_dir[2] + ray_org[2]];
    Some((a, hit_pos[0].1))
}

#[allow(clippy::identity_op)]
fn triangles_in_sphere(
    pos: [f32; 3],
    rad: f32,
    itri0: usize,
    vtx2xyz: &[f32],
    tri2vtx: &[usize],
    tri2adjtri: &[usize]) -> Vec<usize>
{
    use del_geo::{tri3, vec3, vec3::to_na};
    let mut res = Vec::<usize>::new();
    let mut searched = std::collections::BTreeSet::<usize>::new();
    let mut next0 = Vec::<usize>::new();
    next0.push(itri0);
    while let Some(iel0) = next0.pop() {
        if searched.contains(&iel0) { continue; } // already studied
        searched.insert(iel0);
        let dist_min = {
            let i0 = tri2vtx[iel0 * 3 + 0];
            let i1 = tri2vtx[iel0 * 3 + 1];
            let i2 = tri2vtx[iel0 * 3 + 2];
            let (pn, _r0, _r1) = tri3::nearest_to_point3(
                &to_na(vtx2xyz, i0),
                &to_na(vtx2xyz, i1),
                &to_na(vtx2xyz, i2),
                &nalgebra::Vector3::<f32>::from_row_slice(&pos));
            vec3::distance_(pn.as_slice(), &pos)
        };
        if dist_min > rad { continue; }
        res.push(iel0);
        for ie in 0..3 {
            let iel1 = tri2adjtri[iel0 * 3 + ie];
            if iel1 == usize::MAX { continue; }
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
    tri2adjtri: &[usize]) -> bool
{
    use del_geo::vec3;
    let pos_i = crate::trimesh3::position_from_barycentric_coordinate(
        tri2vtx, vtx2xyz,
        smpli.0, smpli.1, smpli.2);
    let indexes_tri = triangles_in_sphere(
        pos_i, rad,
        smpli.0, vtx2xyz, tri2vtx, tri2adjtri);
    for idx_tri in indexes_tri.iter() {
        if !elem2smpl.contains_key(idx_tri) {
            continue;
        }
        for &j_smpl in elem2smpl[idx_tri].iter() {
            let smpl_j = samples[j_smpl];
            let pos_j = crate::trimesh3::position_from_barycentric_coordinate(
                tri2vtx, vtx2xyz,
                smpl_j.0, smpl_j.1, smpl_j.2);
            let dist = vec3::distance_(&pos_i, &pos_j);
            if dist < rad { return true; }
        }
    }
    false
}




