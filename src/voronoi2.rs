pub fn cut_convex2_by_line(
    vtx2xy: &[nalgebra::Vector2<f32>],
    vtx2info: &[[usize; 4]],
    line_s: &nalgebra::Vector2<f32>,
    line_n: &nalgebra::Vector2<f32>,
    i_vtx: usize,
    j_vtx: usize)
    -> (Vec<nalgebra::Vector2::<f32>>, Vec<[usize; 4]>)
{
    // negative->inside
    let depth = |p: &nalgebra::Vector2<f32>| (p - line_s).dot(line_n);
    let mut vtx2xy_new = vec!(nalgebra::Vector2::<f32>::zeros(); 0);
    let mut vtx2info_new = vec!([usize::MAX; 4]; 0);
    for i0_vtx in 0..vtx2xy.len() {
        let i1_vtx = (i0_vtx + 1) % vtx2xy.len();
        let p0 = &vtx2xy[i0_vtx];
        let p1 = &vtx2xy[i1_vtx];
        let d0 = depth(p0);
        let d1 = depth(p1);
        if d0 < 0. { // p0 is inside
            vtx2xy_new.push(*p0);
            vtx2info_new.push(vtx2info[i0_vtx]);
        }
        if d0 * d1 < 0. {
            let pm = p0.scale(d1 / (d1 - d0)) + p1.scale(d0 / (d0 - d1));
            vtx2xy_new.push(pm);
            let info0 = vtx2info[i0_vtx];
            let info1 = vtx2info[i1_vtx];
            let set_a = std::collections::BTreeSet::from_iter([info0[2], info0[3]]);
            let set_b = std::collections::BTreeSet::from_iter([info1[2], info1[3]]);
            let mut intersec = &set_a & &set_b;
            intersec.remove(&usize::MAX);
            if intersec.is_empty() {
                vtx2info_new.push([info0[0], i_vtx, j_vtx, usize::MAX]);
            } else if intersec.len() == 1 {
                let k_vtx = intersec.first().unwrap();
                vtx2info_new.push([usize::MAX, i_vtx, *k_vtx, j_vtx]);
            } else { panic!(); }
        }
    }
    (vtx2xy_new, vtx2info_new)
}

pub fn voronoi_cells<F>(
    vtxl2xy: &[f32],
    site2xy: &[f32],
    site2isalive: F)
    -> (Vec<Vec<nalgebra::Vector2<f32>>>, Vec<Vec<[usize; 4]>>)
where F: Fn(usize) -> bool
{
    let vtxl2xy = crate::vtx2xyz::to_array_of_nalgebra_vector(vtxl2xy);
    let site2xy: Vec<nalgebra::Vector2<f32>> = crate::vtx2xyz::to_array_of_nalgebra_vector(site2xy);
    let num_site = site2xy.len();
    let mut site2vtxc2xy = vec!(
        vec!(nalgebra::Vector2::<f32>::zeros(); 0); num_site);
    let mut site2vtxc2info = vec!(
        vec!([usize::MAX; 4]; 0); num_site);
    for (i_site, pos_i) in site2xy.iter().enumerate() {
        if !site2isalive(i_site) { continue; }
        let mut vtxc2xy = vtxl2xy.clone();
        let mut vtxc2info: Vec<[usize; 4]> = (0..vtxc2xy.len())
            .map(|v| [v, usize::MAX, usize::MAX, usize::MAX]).collect();
        for (j_site, pos_j) in site2xy.iter().enumerate() {
            if !site2isalive(j_site) { continue; }
            if j_site == i_site { continue; }
            let line_s = (pos_i + pos_j) * 0.5;
            let line_n = (pos_j - pos_i).normalize();
            (vtxc2xy, vtxc2info) = cut_convex2_by_line(
                &vtxc2xy, &vtxc2info, &line_s, &line_n, i_site, j_site);
        }
        assert_eq!(vtxc2xy.len(), vtxc2info.len());
        site2vtxc2xy[i_site] = vtxc2xy;
        site2vtxc2info[i_site] = vtxc2info;
    }
    (site2vtxc2xy, site2vtxc2info)
}

pub fn indexing(
    site2vtxc2xy: &[Vec<nalgebra::Vector2<f32>>],
    site2vtxc2info: &[Vec<[usize; 4]>])
    -> (Vec<usize>, Vec<usize>, Vec<nalgebra::Vector2<f32>>, Vec<[usize; 4]>)
{
    let num_site = site2vtxc2info.len();
    let sort_info = |info: &[usize; 4]| {
        let mut tmp = [info[1], info[2], info[3]];
        tmp.sort();
        [info[0], tmp[0], tmp[1], tmp[2]]
    };
    let mut info2vtxv = std::collections::BTreeMap::<[usize; 4], usize>::new();
    let mut vtxv2info : Vec<[usize;4]> = vec!();
    for vtxc2info in site2vtxc2info {
        for info in vtxc2info {
            let info0 = sort_info(info);
            if !info2vtxv.contains_key(&info0) {
                let i_vtxc = info2vtxv.len();
                info2vtxv.insert(info0, i_vtxc);
                vtxv2info.push(info0);
            }
        }
    }
    let info2vtxv = info2vtxv;
    let vtxv2info = vtxv2info;
    let num_vtxv = info2vtxv.len();
    let mut vtxv2xy = vec!(nalgebra::Vector2::<f32>::zeros(); num_vtxv);
    let mut site2idx = vec!(0; 1);
    let mut idx2vtxc = vec!(0usize; 0);
    for i_site in 0..num_site {
        for (ind, info) in site2vtxc2info[i_site].iter().enumerate() {
            let info0 = sort_info(info);
            let i_vtxv = info2vtxv.get(&info0).unwrap();
            idx2vtxc.push(*i_vtxv);
            vtxv2xy[*i_vtxv] = site2vtxc2xy[i_site][ind];
        }
        site2idx.push(idx2vtxc.len());
    }
    assert_eq!(site2idx.len(), num_site + 1);
    (site2idx, idx2vtxc, vtxv2xy, vtxv2info)
}

pub fn position_of_voronoi_vertex(
    info: &[usize;4],
    vtxl2xy: &[f32],
    site2xy: &[f32]) -> nalgebra::Vector2<f32>
{
    if info[1] == usize::MAX { // original point
        del_geo::vec2::to_na(vtxl2xy, info[0])
    }
    else if info[3] == usize::MAX { // two points against edge
        let num_vtxl =  vtxl2xy.len() / 2;
        assert!(info[0] < num_vtxl);
        let i1_loop = info[0];
        let i2_loop = (i1_loop + 1) % num_vtxl;
        let l1 = del_geo::vec2::to_na(vtxl2xy, i1_loop);
        let l2 = del_geo::vec2::to_na(vtxl2xy, i2_loop);
        let s1 = &del_geo::vec2::to_na(site2xy, info[1]);
        let s2 = &del_geo::vec2::to_na(site2xy, info[2]);
        return del_geo::line2::intersection(
            &l1, &(l2 - l1),
            &((s1 + s2) * 0.5),
            &del_geo::vec2::rotate90(&(s2 - s1)));
    } else { // three points
        assert_eq!(info[0], usize::MAX);
        return del_geo::tri2::circumcenter(
            &del_geo::vec2::to_na(site2xy, info[1]),
            &del_geo::vec2::to_na(site2xy, info[2]),
            &del_geo::vec2::to_na(site2xy, info[3]));
    }
}

#[test]
fn test_voronoi_concave() {
    let vtxl2xy = vec!(
        0.0, 0.0,
        1.0, 0.0,
        1.0, 0.5,
        0.5, 0.5,
        0.5, 1.0,
        0.0, 1.0);
    let site2xy = crate::sampling::poisson_disk_sampling_from_polyloop2(
        &vtxl2xy, 0.15, 10);
    let num_site = site2xy.len() / 2;
    {   // save boundary loop and input points
        let num_vtxl = vtxl2xy.len()/2;
        let mut vtxl2xy = vtxl2xy.clone();
        vtxl2xy.extend(site2xy.clone());
        let _ = crate::io_obj::save_edge2vtx_vtx2xyz(
            "target/voronoi_concave_input.obj",
            &crate::edge2vtx::from_polyloop(num_vtxl), &vtxl2xy, 2);
    }
    let (site2vtxc2xy, _site2vtxc2info)
        = voronoi_cells(&vtxl2xy, &site2xy, |_isite| true );
    { // write each cell
        let mut edge2vtxo = vec!(0usize; 0);
        let mut vtxo2xy = vec!(0f32; 0);
        for i_site in 0..num_site {
            let vtxc2xy = &site2vtxc2xy[i_site];
            let vtxc2xy = crate::vtx2xyz::from_array_of_nalgebra(&vtxc2xy);
            let edge2vtxc = crate::edge2vtx::from_polyloop(vtxc2xy.len() / 2);
            crate::uniform_mesh::merge(
                &mut edge2vtxo, &mut vtxo2xy,
                &edge2vtxc, &vtxc2xy, 2);
        }
        let _ = crate::io_obj::save_edge2vtx_vtx2xyz(
            "target/voronoi_concave_cells.obj",
            &edge2vtxo, &vtxo2xy, 2);
    }
}

#[test]
fn test_voronoi_convex() {
    let vtxl2xy = vec!(0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0);
    let site2xy = crate::sampling::poisson_disk_sampling_from_polyloop2(
        &vtxl2xy, 0.15, 10);
    let num_site = site2xy.len() / 2;
    {
        let mut vtxl2xy = vtxl2xy.clone();
        vtxl2xy.extend(site2xy.clone());
        let _ = crate::io_obj::save_edge2vtx_vtx2xyz(
            "target/voronoi_convex_input.obj",
            &[0, 1, 1, 2, 2, 3, 3, 0], &vtxl2xy, 2);
    }
    let (site2vtxc2xy, site2vtxc2info)
        = voronoi_cells(&vtxl2xy, &site2xy, |_isite| true );
    assert_eq!(site2vtxc2xy.len(), num_site);
    assert_eq!(site2vtxc2info.len(), num_site);
    { // write each cell
        let mut edge2vtxo = vec!(0usize; 0);
        let mut vtxo2xy = vec!(0f32; 0);
        for i_site in 0..num_site {
            let vtxc2xy = &site2vtxc2xy[i_site];
            let vtxc2xy = crate::vtx2xyz::from_array_of_nalgebra(&vtxc2xy);
            let edge2vtxc = crate::edge2vtx::from_polyloop(vtxc2xy.len() / 2);
            crate::uniform_mesh::merge(
                &mut edge2vtxo, &mut vtxo2xy,
                &edge2vtxc, &vtxc2xy, 2);
        }
        let _ = crate::io_obj::save_edge2vtx_vtx2xyz(
            "target/voronoi_convex_cells.obj",
            &edge2vtxo, &vtxo2xy, 2);
    }
    // check if the info and the coordinates of vtxc is OK
    for i_site in 0..num_site {
        let vtxc2xy = &site2vtxc2xy[i_site];
        let vtxc2info = &site2vtxc2info[i_site];
        for (i_vtxc, info) in vtxc2info.iter().enumerate() {
            let cc0 = position_of_voronoi_vertex(info,  &vtxl2xy, &site2xy);
            let cc1 = vtxc2xy[i_vtxc];
            assert!((cc0 - cc1).norm() < 1.0e-5);
        }
    }
    let (site2idx, idx2vtxc, vtxc2xy, vtxc2info)
        = indexing(&site2vtxc2xy, &site2vtxc2info);
    for (i_vtxc, info) in vtxc2info.iter().enumerate() {
        let cc0 = position_of_voronoi_vertex(info, &vtxl2xy, &site2xy);
        let cc1 = vtxc2xy[i_vtxc];
        assert!((cc0 - cc1).norm() < 1.0e-5);
    }
    { // write edges to file
        let edge2vtxc = crate::edge2vtx::from_polygon_mesh(
            &site2idx, &idx2vtxc, vtxc2xy.len());
        let vtxc2xy = crate::vtx2xyz::from_array_of_nalgebra(&vtxc2xy);
        let _ = crate::io_obj::save_edge2vtx_vtx2xyz(
            "target/voronoi_convex_indexed.obj", &edge2vtxc, &vtxc2xy, 2);
    }
}

#[test]
fn test_voronoi_sites_on_edge() {
    let vtxl2xy = vec!(0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0);
    let (tri2vtx, vtx2xy)
        = crate::trimesh2_dynamic::meshing_from_polyloop2::<usize, f32>(
        &vtxl2xy, 0.08, 0.08);
    let tri2xycc = crate::trimesh2::tri2circumcenter(&tri2vtx, &vtx2xy);
    let (bedge2vtx, tri2triedge)
        = crate::trimesh_topology::boundaryedge2vtx(&tri2vtx, vtx2xy.len()/2);
    //
    let bedge2xymp = {
        let mut bedge2xymp = vec!(0f32; bedge2vtx.len());
        for (i_bedge, node2vtx) in bedge2vtx.chunks(2).enumerate() {
            let (i0_vtx, i1_vtx) = (node2vtx[0], node2vtx[1]);
            bedge2xymp[i_bedge*2+0] = (vtx2xy[i0_vtx*2+0] + vtx2xy[i1_vtx*2+0])*0.5;
            bedge2xymp[i_bedge*2+1] = (vtx2xy[i0_vtx*2+1] + vtx2xy[i1_vtx*2+1])*0.5;
        }
        bedge2xymp
    };
    let pnt2xy = {
        let mut pnt2xy = tri2xycc;
        pnt2xy.extend(bedge2xymp);
        pnt2xy
    };
    let vedge2pnt = {
        let mut vedge2pnt = vec!(0usize;0);
        for (i_tri, node2triedge) in tri2triedge.chunks(3).enumerate() {
            for i_node in 0..3 {
                let i_triedge = node2triedge[i_node];
                if i_triedge <= i_tri { continue; }
                vedge2pnt.extend(&[i_tri, i_triedge]);
            }
        }
        let (face2idx, idx2node)
            = crate::elem2elem::face2node_of_simplex_element(2);
        let bedge2bedge = crate::elem2elem::from_uniform_mesh(
            &bedge2vtx, 2,
            &face2idx, &idx2node, vtx2xy.len()/2);
        let num_tri = tri2vtx.len() / 3;
        for (i_bedge, node2bedge) in bedge2bedge.chunks(2).enumerate() {
            for i_node in 0..2 {
                let j_bedge = node2bedge[i_node];
                if i_bedge > j_bedge { continue; }
                vedge2pnt.extend(&[i_bedge + num_tri, j_bedge + num_tri]);
            }
        }
        vedge2pnt
    };
    let _ = crate::io_obj::save_edge2vtx_vtx2xyz(
        "target/voronoi_sites_on_edge.obj", &vedge2pnt, &pnt2xy, 2);
}