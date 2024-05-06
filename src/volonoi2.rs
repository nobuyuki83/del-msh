
pub fn cut_convex2_by_line(
    vtx2xy: &Vec<nalgebra::Vector2<f32>>,
    vtx2info: &Vec<[usize;4]>,
    line_s: &nalgebra::Vector2<f32>,
    line_n: &nalgebra::Vector2<f32>,
    i_vtx: usize,
    j_vtx: usize)
    -> (Vec<nalgebra::Vector2::<f32>>, Vec<[usize;4]>)
{
    // negative->inside
    let depth = | p: &nalgebra::Vector2<f32> | (p-line_s).dot(&line_n);
    let mut vtx2xy_new = vec!(nalgebra::Vector2::<f32>::zeros(); 0);
    let mut vtx2info_new = vec!([usize::MAX;4]; 0);
    for i0_vtx in 0..vtx2xy.len() {
        let i1_vtx = (i0_vtx+1) % vtx2xy.len();
        let p0 = &vtx2xy[i0_vtx];
        let p1 = &vtx2xy[i1_vtx];
        let d0 = depth(p0);
        let d1 = depth(p1);
        if d0 < 0. { // p0 is inside
            vtx2xy_new.push(*p0);
            vtx2info_new.push(vtx2info[i0_vtx]);
        }
        if d0*d1 < 0. {
            let pm = p0.scale(d1 / (d1 - d0)) + p1.scale(d0 / (d0 - d1));
            vtx2xy_new.push(pm);
            let info0 = vtx2info[i0_vtx];
            let info1 = vtx2info[i1_vtx];
            let set_a = std::collections::BTreeSet::from_iter([info0[2], info0[3]]);
            let set_b = std::collections::BTreeSet::from_iter([info1[2], info1[3]]);
            let mut intersec = &set_a & &set_b;
            intersec.remove(&usize::MAX);
            if intersec.len() == 0 {
                vtx2info_new.push([info0[0], i_vtx, j_vtx, usize::MAX]);
            }
            else if intersec.len() == 1 {
                let k_vtx = intersec.first().unwrap();
                vtx2info_new.push([usize::MAX, i_vtx, *k_vtx, j_vtx]);
            }
            else { panic!(); }
        }
    }
    (vtx2xy_new, vtx2info_new)
}

pub fn volonoi_cells(
    vtxl2xy: &[nalgebra::Vector2<f32>],
    site2xy: &[nalgebra::Vector2<f32>]) -> (Vec<Vec<nalgebra::Vector2<f32>>>, Vec<Vec<[usize;4]>>)
{
    let num_site = site2xy.len();
    let mut site2vtxc2xy = vec!(
        vec!(nalgebra::Vector2::<f32>::zeros(); 0); num_site);
    let mut site2vtxc2info = vec!(
        vec!([usize::MAX; 4]; 0); num_site);
    for (i_vtxs, pos_i) in site2xy.iter().enumerate() {
        let mut vtxc2xy = Vec::<nalgebra::Vector2::<f32>>::from(vtxl2xy);
        let mut vtxc2info: Vec<[usize; 4]> = (0..vtxc2xy.len())
            .map(|v| [v, usize::MAX, usize::MAX, usize::MAX]).collect();
        for (j_vtx, pos_j) in site2xy.iter().enumerate() {
            if j_vtx == i_vtxs { continue; }
            let line_s = (pos_i + pos_j) * 0.5;
            let line_n = (pos_j - pos_i).normalize();
            (vtxc2xy, vtxc2info) = cut_convex2_by_line(
                &vtxc2xy, &vtxc2info, &line_s, &line_n, i_vtxs, j_vtx);
        }
        assert_eq!(vtxc2xy.len(), vtxc2info.len());
        site2vtxc2xy[i_vtxs] = vtxc2xy;
        site2vtxc2info[i_vtxs] = vtxc2info;
    }
    (site2vtxc2xy, site2vtxc2info)
}

pub fn indexing(
    site2vtxc2xy: &Vec<Vec<nalgebra::Vector2<f32>>>,
    site2vtxc2info: &Vec<Vec<[usize;4]>>)
    -> (Vec<Vec<usize>>,  Vec<nalgebra::Vector2<f32>>, Vec<[usize;4]>)
{
    let num_site = site2vtxc2info.len();
    let sort_info = |info: &[usize;4]| {
        let mut tmp = [info[1], info[2], info[3]];
        tmp.sort();
        [info[0], tmp[0], tmp[1], tmp[2]]
    };
    let mut info2vtxc = std::collections::BTreeMap::<[usize;4], usize>::new();
    for i_site in 0..num_site {
        let vtxc2info = &site2vtxc2info[i_site];
        for info in vtxc2info {
            let info0 = sort_info(info);
            if !info2vtxc.contains_key(&info0) {
                let i_vtxc = info2vtxc.len();
                info2vtxc.insert(info0, i_vtxc);
            }
        }
    }
    let info2vtxc = info2vtxc;
    let vtxc2info: Vec<[usize;4]> = info2vtxc.clone().into_keys().collect();
    let num_vtxc = info2vtxc.len();
    let mut vtxc2xy = vec!(nalgebra::Vector2::<f32>::zeros();num_vtxc);
    let mut site2vtxc = vec!(vec!(0usize;0);num_site);
    for i_site in 0..num_site {
        for (ind, info) in site2vtxc2info[i_site].iter().enumerate() {
            let info0 = sort_info(info);
            let i_vtxc = info2vtxc.get(&info0).unwrap();
            site2vtxc[i_site].push(*i_vtxc);
            vtxc2xy[*i_vtxc] = site2vtxc2xy[i_site][ind];
        }
    }
    (site2vtxc, vtxc2xy, vtxc2info)
}

#[test]
fn test_volonoi() {
    let vtxl2xy = vec!(0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0);
    let site2xy = crate::sampling::poisson_disk_sampling_from_polyloop2(
        &vtxl2xy, 0.15, 10);
    let num_site = site2xy.len();
    {
        let vtx2xy = crate::vtx2xyz::from_array_of_nalgebra(&site2xy);
        let mut vtxl2xy = vtxl2xy.clone();
        vtxl2xy.extend(vtx2xy);
        let _ = crate::io_obj::save_edge2vtx_vtx2xyz(
            "target/volonoi_a.obj",
            &[0, 1, 1, 2, 2, 3, 3, 0], &vtxl2xy, 2);
    }
    let vtxl2xy = crate::vtx2xyz::to_array_of_nalgebra_vector::<f32, 2>(&vtxl2xy);
    let (site2vtxc2xy, site2vtxc2info)
        = volonoi_cells(&vtxl2xy, &site2xy);
    assert_eq!(site2vtxc2xy.len(), num_site);
    assert_eq!(site2vtxc2info.len(), num_site);
    for i_site in 0..num_site {
        let vtxc2xy = &site2vtxc2xy[i_site];
        let vtxc2info = &site2vtxc2info[i_site];
        let _ = crate::io_obj::save_vtx2vecn_as_polyloop(
            format!("target/volonoi_{}.obj", i_site),
            &vtxc2xy);
        for (i_cell, info) in vtxc2info.iter().enumerate() {
            if info[1] == usize::MAX { // original point
                let cc0 = vtxl2xy[info[0]];
                let cc1 = vtxc2xy[i_cell];
                assert!((cc0 - cc1).norm() < 1.0e-5);
            } else if info[3] != usize::MAX { // three points
                let iv0 = info[1];
                let iv1 = info[2];
                let iv2 = info[3];
                let cc0 = del_geo::tri2::circumcenter(
                    &site2xy[iv0], &site2xy[iv1], &site2xy[iv2]);
                let cc1 = vtxc2xy[i_cell];
                // dbg!((iv0, iv1, iv2, &cc, &cc1));
                assert!((cc0 - cc1).norm() < 1.0e-5);
            } else { // two points against edge
                assert!(info[0] < vtxl2xy.len());
                assert_eq!(info[3], usize::MAX);
                let i1_loop = info[0];
                let i2_loop = (i1_loop + 1) % vtxl2xy.len();
                let q1 = vtxl2xy[i1_loop];
                let q2 = vtxl2xy[i2_loop];
                let i1_site = info[1]; // vtxs2vectwo
                let i2_site = info[2]; // vtxs2vectwo
                let p1 = &site2xy[i1_site];
                let p2 = &site2xy[i2_site];
                let cc0 = del_geo::line2::intersection(
                    &q1, &(q2 - q1),
                    &((p1 + p2) * 0.5), &del_geo::vec2::rotate90(&(p2 - p1)));
                let cc1 = vtxc2xy[i_cell];
                assert!((cc0 - cc1).norm() < 1.0e-5);
            }
        }
    }
    let (site2loop2vtxc, vtxc2xy, _vtxc2info)
        = indexing(&site2vtxc2xy, &site2vtxc2info);
    dbg!(&site2loop2vtxc);
    let mut edge2vtxc= vec!(0usize;0);
    for loop2vtxc in &site2loop2vtxc {
        for idx0 in 0..loop2vtxc.len() {
            let idx1 = (idx0 + 1) % loop2vtxc.len();
            edge2vtxc.push(loop2vtxc[idx0]);
            edge2vtxc.push(loop2vtxc[idx1]);
        }
    }
    // dbg!(&edge2vtxc);
    {
        let vtxc2xy = crate::vtx2xyz::from_array_of_nalgebra(&vtxc2xy);
        let _ = crate::io_obj::save_edge2vtx_vtx2xyz(
            "target/volonoi_b.obj", &edge2vtxc, &vtxc2xy, 2);
    }
}