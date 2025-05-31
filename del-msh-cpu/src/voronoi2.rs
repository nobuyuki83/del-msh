//! methods for 2D Voronoi diagram

#[derive(Clone)]
pub struct Cell {
    pub vtx2xy: Vec<f32>,
    pub vtx2info: Vec<[usize; 4]>,
}

impl Cell {
    fn is_inside(&self, p: &[f32; 2]) -> bool {
        let wn = crate::polyloop2::winding_number(&self.vtx2xy, p);
        (wn - 1.0).abs() < 0.1
    }

    fn area(&self) -> f32 {
        crate::polyloop2::area(&self.vtx2xy)
    }

    fn new_from_polyloop2(vtx2xy_in: &[f32]) -> Self {
        let vtx2info = (0..vtx2xy_in.len() / 2)
            .map(|v| [v, usize::MAX, usize::MAX, usize::MAX])
            .collect();
        Cell {
            vtx2xy: vtx2xy_in.to_owned(),
            vtx2info,
        }
    }

    fn new_empty() -> Self {
        let vtx2xy: Vec<f32> = vec![];
        let vtx2info = vec![[usize::MAX; 4]; 0];
        Cell { vtx2xy, vtx2info }
    }
}

fn hoge(
    vtx2xy: &[f32],
    vtx2info: &[[usize; 4]],
    vtxnews: &[(f32, usize, [f32; 2], [usize; 4])],
    vtx2vtxnew: &[usize],
    vtxnew2isvisisted: &mut [bool],
) -> Option<Cell> {
    let num_vtx = vtx2xy.len() / 2;
    let mut vtx2xy_new: Vec<f32> = vec![];
    let mut vtx2info_new = vec![[usize::MAX; 4]; 0];
    let (i_vtx0, is_new0) = {
        let i_vtxnew_start = vtxnew2isvisisted
            .iter()
            .position(|is_visited| !is_visited)?;
        (i_vtxnew_start, true)
    };
    let (mut i_vtx, mut is_new) = (i_vtx0, is_new0);
    let mut is_entry = true;
    loop {
        // dbg!(i_vtx, is_new, is_entry, i_vtx0, is_new0);
        if is_new {
            vtx2xy_new.push(vtxnews[i_vtx].2[0]);
            vtx2xy_new.push(vtxnews[i_vtx].2[1]);
            vtx2info_new.push(vtxnews[i_vtx].3);
            vtxnew2isvisisted[i_vtx] = true;
            if is_entry {
                i_vtx = vtxnews[i_vtx].1;
                i_vtx = (i_vtx + 1) % num_vtx;
                // assert!(depth(&vtx2xy[i_vtx]) < 0., "{}", depth(&vtx2xy[i_vtx]));
                is_new = false;
                is_entry = false;
            } else {
                i_vtx -= 1;
                is_new = true;
                is_entry = true;
            }
        } else {
            vtx2xy_new.push(vtx2xy[i_vtx * 2]);
            vtx2xy_new.push(vtx2xy[i_vtx * 2 + 1]);
            vtx2info_new.push(vtx2info[i_vtx]);
            if vtx2vtxnew[i_vtx] == usize::MAX {
                i_vtx = (i_vtx + 1) % num_vtx;
                is_new = false;
            } else {
                i_vtx = vtx2vtxnew[i_vtx];
                is_new = true;
                is_entry = false;
            }
        }
        if i_vtx == i_vtx0 && is_new == is_new0 {
            break;
        }
    }
    Some(Cell {
        vtx2xy: vtx2xy_new,
        vtx2info: vtx2info_new,
    })
}

/// vtx2xy should be counter-clockwise
pub fn cut_polygon_by_line(
    cell: &Cell,
    line_s: &[f32; 2],
    line_n: &[f32; 2],
    i_vtx: usize,
    j_vtx: usize,
) -> Vec<Cell> {
    use del_geo_core::vec2::Vec2;
    // negative->inside
    let depth = |p: &[f32; 2]| p.sub(line_s).dot(line_n);
    let num_vtx = cell.vtx2xy.len() / 2;
    let (vtxnews, is_inside) = {
        let line_t = del_geo_core::vec2::rotate90(line_n);
        let mut is_inside = false;
        let mut vtxnews: Vec<(f32, usize, [f32; 2], [usize; 4])> = vec![];
        for i0_vtx in 0..num_vtx {
            let i1_vtx = (i0_vtx + 1) % num_vtx;
            let p0 = crate::vtx2xy::to_vec2(&cell.vtx2xy, i0_vtx);
            let p1 = crate::vtx2xy::to_vec2(&cell.vtx2xy, i1_vtx);
            let d0 = depth(p0);
            if d0 < 0. {
                is_inside = true;
            }
            let d1 = depth(p1);
            assert_ne!(d0 * d0, 0., "{} {}", d0, d1);
            if d0 * d1 > 0. {
                continue;
            }
            let pm = p0.scale(d1 / (d1 - d0)).add(&p1.scale(d0 / (d0 - d1)));
            let t0 = line_t.dot(&pm);
            //
            let info0 = cell.vtx2info[i0_vtx];
            let info1 = cell.vtx2info[i1_vtx];
            let set_a = std::collections::BTreeSet::from_iter([info0[2], info0[3]]);
            let set_b = std::collections::BTreeSet::from_iter([info1[2], info1[3]]);
            let mut intersec = &set_a & &set_b;
            intersec.remove(&usize::MAX);
            let info = if intersec.is_empty() {
                [info0[0], i_vtx, j_vtx, usize::MAX]
            } else if intersec.len() == 1 {
                let k_vtx = intersec.first().unwrap();
                [usize::MAX, i_vtx, *k_vtx, j_vtx]
            } else {
                panic!();
            };
            //
            vtxnews.push((-t0, i0_vtx, pm, info));
        }
        vtxnews.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        (vtxnews, is_inside)
    };
    if vtxnews.is_empty() {
        // no intersection
        return if is_inside {
            vec![cell.clone()]
        } else {
            vec![]
        };
    }
    assert_eq!(vtxnews.len() % 2, 0);
    let vtx2vtxnew = {
        let mut vtx2vtxnew = vec![usize::MAX; num_vtx];
        for (i_vtxnew, vtxnew) in vtxnews.iter().enumerate() {
            vtx2vtxnew[vtxnew.1] = i_vtxnew;
        }
        vtx2vtxnew
    };
    let mut vtxnew2isvisisted = vec![false; vtxnews.len()];
    let mut cells: Vec<Cell> = vec![];
    loop {
        let c0 = hoge(
            &cell.vtx2xy,
            &cell.vtx2info,
            &vtxnews,
            &vtx2vtxnew,
            &mut vtxnew2isvisisted,
        );
        let Some(cell) = c0 else {
            break;
        };
        cells.push(cell);
    }
    cells
}

pub fn voronoi_cells<F>(vtxl2xy: &[f32], site2xy: &[f32], site2isalive: F) -> Vec<Cell>
where
    F: Fn(usize) -> bool,
{
    use del_geo_core::vec2::Vec2;
    let num_site = site2xy.len() / 2;
    let mut site2cell = vec![Cell::new_empty(); num_site];
    for (i_site, pos_i) in site2xy.chunks(2).enumerate() {
        let pos_i = arrayref::array_ref![pos_i, 0, 2];
        if !site2isalive(i_site) {
            continue;
        }
        let mut cell_stack = vec![Cell::new_from_polyloop2(vtxl2xy)];
        for (j_site, pos_j) in site2xy.chunks(2).enumerate() {
            let pos_j = arrayref::array_ref![pos_j, 0, 2];
            if !site2isalive(j_site) {
                continue;
            }
            if j_site == i_site {
                continue;
            }
            let line_s = pos_i.add(pos_j).scale(0.5);
            let line_n = pos_j.sub(pos_i).normalize();
            let mut cell_stack_new = vec![];
            for cell_in in cell_stack {
                let cells = cut_polygon_by_line(&cell_in, &line_s, &line_n, i_site, j_site);
                cell_stack_new.extend(cells);
            }
            cell_stack = cell_stack_new;
        }
        if cell_stack.is_empty() {
            site2cell[i_site] = Cell::new_empty();
            continue;
        }
        if cell_stack.len() == 1 {
            site2cell[i_site] = cell_stack[0].clone();
            continue;
        }
        let mut depthcell: Vec<(f32, usize)> = vec![];
        for (i_cell, cell) in cell_stack.iter().enumerate() {
            let is_inside = cell.is_inside(crate::vtx2xy::to_vec2(site2xy, i_site));
            let dist = if is_inside { 0. } else { 1.0 / cell.area() };
            depthcell.push((dist, i_cell));
        }
        depthcell.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let i_cell = depthcell[0].1;
        assert!(!cell_stack[i_cell].vtx2xy.is_empty());
        site2cell[i_site] = cell_stack[i_cell].clone();
    }
    site2cell
}

pub struct VoronoiMesh {
    pub site2idx: Vec<usize>,
    pub idx2vtxv: Vec<usize>,
    pub vtxv2xy: Vec<[f32; 2]>,
    pub vtxv2info: Vec<[usize; 4]>,
}

pub fn indexing(site2cell: &[Cell]) -> VoronoiMesh {
    let num_site = site2cell.len();
    let sort_info = |info: &[usize; 4]| {
        let mut tmp = [info[1], info[2], info[3]];
        tmp.sort();
        [info[0], tmp[0], tmp[1], tmp[2]]
    };
    let mut info2vtxv = std::collections::BTreeMap::<[usize; 4], usize>::new();
    let mut vtxv2info: Vec<[usize; 4]> = vec![];
    for cell in site2cell.iter() {
        for info in &cell.vtx2info {
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
    let mut vtxv2xy = vec![[0f32; 2]; num_vtxv];
    let mut site2idx = vec![0; 1];
    let mut idx2vtxc = vec![0usize; 0];
    for cell in site2cell.iter() {
        for (ind, info) in cell.vtx2info.iter().enumerate() {
            let info0 = sort_info(info);
            let i_vtxv = info2vtxv.get(&info0).unwrap();
            idx2vtxc.push(*i_vtxv);
            vtxv2xy[*i_vtxv] = *crate::vtx2xy::to_vec2(&cell.vtx2xy, ind);
        }
        site2idx.push(idx2vtxc.len());
    }
    assert_eq!(site2idx.len(), num_site + 1);
    VoronoiMesh {
        site2idx,
        idx2vtxv: idx2vtxc,
        vtxv2xy,
        vtxv2info,
    }
}

pub fn position_of_voronoi_vertex(info: &[usize; 4], vtxl2xy: &[f32], site2xy: &[f32]) -> [f32; 2] {
    use del_geo_core::vec2::Vec2;
    if info[1] == usize::MAX {
        // original point
        *crate::vtx2xy::to_vec2(vtxl2xy, info[0])
    } else if info[3] == usize::MAX {
        // two points against edge
        let num_vtxl = vtxl2xy.len() / 2;
        assert!(info[0] < num_vtxl);
        let i1_loop = info[0];
        let i2_loop = (i1_loop + 1) % num_vtxl;
        let l1 = crate::vtx2xy::to_vec2(vtxl2xy, i1_loop);
        let l2 = crate::vtx2xy::to_vec2(vtxl2xy, i2_loop);
        let s1 = &crate::vtx2xy::to_vec2(site2xy, info[1]);
        let s2 = &crate::vtx2xy::to_vec2(site2xy, info[2]);
        return del_geo_core::line2::intersection(
            l1,
            &l2.sub(l1),
            &s1.add(s2).scale(0.5),
            &del_geo_core::vec2::rotate90(&s2.sub(s1)),
        );
    } else {
        // three points
        assert_eq!(info[0], usize::MAX);
        return del_geo_core::tri2::circumcenter(
            crate::vtx2xy::to_vec2(site2xy, info[1]),
            crate::vtx2xy::to_vec2(site2xy, info[2]),
            crate::vtx2xy::to_vec2(site2xy, info[3]),
        );
    }
}

#[test]
fn test_voronoi_concave() {
    let mut reng = rand::rng();
    // let vtxl2xy = vec![0.0, 0.0, 1.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5, 1.0, 0.0, 1.0];
    let vtxl2xy = vec![
        0.0, 0.0, 1.0, 0.0, 1.0, 0.2, 0.2, 0.2, 0.2, 0.5, 1.0, 0.5, 1.0, 1.0, 0.0, 1.0,
    ];
    let site2xy = crate::polyloop2::poisson_disk_sampling(&vtxl2xy, 0.15, 30, &mut reng);
    let num_site = site2xy.len() / 2;
    {
        // save boundary loop and input points
        let num_vtxl = vtxl2xy.len() / 2;
        let mut vtxl2xy = vtxl2xy.clone();
        vtxl2xy.extend(site2xy.clone());
        let _ = crate::io_obj::save_edge2vtx_vtx2xyz(
            "../target/voronoi_concave_input.obj",
            &crate::edge2vtx::from_polyloop(num_vtxl),
            &vtxl2xy,
            2,
        );
    }
    let site2cell = voronoi_cells(&vtxl2xy, &site2xy, |_isite| true);
    {
        // write each cell
        let mut edge2vtxo = vec![0usize; 0];
        let mut vtxo2xy = vec![0f32; 0];
        for i_site in 0..num_site {
            let vtxc2xy = &site2cell[i_site].vtx2xy;
            let edge2vtxc = crate::edge2vtx::from_polyloop(vtxc2xy.len() / 2);
            crate::uniform_mesh::merge(&mut edge2vtxo, &mut vtxo2xy, &edge2vtxc, vtxc2xy, 2);
        }
        crate::io_obj::save_edge2vtx_vtx2xyz(
            "../target/voronoi_concave_cells.obj",
            &edge2vtxo,
            &vtxo2xy,
            2,
        )
        .unwrap();
    }
}

#[test]
fn test_voronoi_convex() {
    use del_geo_core::vec2::Vec2;
    let mut reng = rand::rng();
    let vtxl2xy = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
    let site2xy = crate::polyloop2::poisson_disk_sampling(&vtxl2xy, 0.15, 30, &mut reng);
    let num_site = site2xy.len() / 2;
    {
        let mut vtxl2xy = vtxl2xy.clone();
        vtxl2xy.extend(site2xy.clone());
        crate::io_obj::save_edge2vtx_vtx2xyz(
            "../target/voronoi_convex_input.obj",
            &[0, 1, 1, 2, 2, 3, 3, 0],
            &vtxl2xy,
            2,
        )
        .unwrap();
    }
    let site2cell = voronoi_cells(&vtxl2xy, &site2xy, |_isite| true);
    assert_eq!(site2cell.len(), num_site);
    {
        // write each cell
        let mut edge2vtxo = vec![0usize; 0];
        let mut vtxo2xy = vec![0f32; 0];
        for i_site in 0..num_site {
            let vtxc2xy = &site2cell[i_site].vtx2xy;
            let edge2vtxc = crate::edge2vtx::from_polyloop(vtxc2xy.len() / 2);
            crate::uniform_mesh::merge(&mut edge2vtxo, &mut vtxo2xy, &edge2vtxc, vtxc2xy, 2);
        }
        let _ = crate::io_obj::save_edge2vtx_vtx2xyz(
            "../target/voronoi_convex_cells.obj",
            &edge2vtxo,
            &vtxo2xy,
            2,
        );
    }
    // check if the info and the coordinates of vtxc is OK
    for i_site in 0..num_site {
        let vtxc2xy = &site2cell[i_site].vtx2xy;
        let vtxc2info = &site2cell[i_site].vtx2info;
        for (i_vtxc, info) in vtxc2info.iter().enumerate() {
            let cc0 = position_of_voronoi_vertex(info, &vtxl2xy, &site2xy);
            let cc1 = crate::vtx2xy::to_vec2(vtxc2xy, i_vtxc);
            assert!(cc0.sub(cc1).norm() < 1.0e-5);
        }
    }
    let voronoi_mesh = indexing(&site2cell);
    for (i_vtxc, info) in voronoi_mesh.vtxv2info.iter().enumerate() {
        let cc0 = position_of_voronoi_vertex(info, &vtxl2xy, &site2xy);
        let cc1 = voronoi_mesh.vtxv2xy[i_vtxc];
        assert!(cc0.sub(&cc1).norm() < 1.0e-5);
    }
    {
        // write edges to file
        let edge2vtxc = crate::edge2vtx::from_polygon_mesh(
            &voronoi_mesh.site2idx,
            &voronoi_mesh.idx2vtxv,
            voronoi_mesh.vtxv2xy.len(),
        );
        use slice_of_array::SliceFlatExt;
        let vtxc2xy = voronoi_mesh.vtxv2xy.flat();
        crate::io_obj::save_edge2vtx_vtx2xyz(
            "../target/voronoi_convex_indexed.obj",
            &edge2vtxc,
            vtxc2xy,
            2,
        )
        .unwrap();
    }
}

#[test]
fn test_voronoi_sites_on_edge() {
    let vtxl2xy = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
    let (tri2vtx, vtx2xy) =
        crate::trimesh2_dynamic::meshing_from_polyloop2::<usize, f32>(&vtxl2xy, 0.08, 0.08);
    let tri2xycc = crate::trimesh2::tri2circumcenter(&tri2vtx, &vtx2xy);
    let (bedge2vtx, tri2triedge) =
        crate::trimesh_topology::boundaryedge2vtx(&tri2vtx, vtx2xy.len() / 2);
    //
    let bedge2xymp = {
        let mut bedge2xymp = vec![0f32; bedge2vtx.len()];
        for (i_bedge, node2vtx) in bedge2vtx.chunks(2).enumerate() {
            let (i0_vtx, i1_vtx) = (node2vtx[0], node2vtx[1]);
            bedge2xymp[i_bedge * 2] = (vtx2xy[i0_vtx * 2] + vtx2xy[i1_vtx * 2]) * 0.5;
            bedge2xymp[i_bedge * 2 + 1] = (vtx2xy[i0_vtx * 2 + 1] + vtx2xy[i1_vtx * 2 + 1]) * 0.5;
        }
        bedge2xymp
    };
    let pnt2xy = {
        let mut pnt2xy = tri2xycc;
        pnt2xy.extend(bedge2xymp);
        pnt2xy
    };
    let vedge2pnt = {
        let mut vedge2pnt = vec![0usize; 0];
        for (i_tri, node2triedge) in tri2triedge.chunks(3).enumerate() {
            for i_node in 0..3 {
                let i_triedge = node2triedge[i_node];
                if i_triedge <= i_tri {
                    continue;
                }
                vedge2pnt.extend(&[i_tri, i_triedge]);
            }
        }
        let (face2idx, idx2node) = crate::elem2elem::face2node_of_simplex_element(2);
        let bedge2bedge = crate::elem2elem::from_uniform_mesh(
            &bedge2vtx,
            2,
            &face2idx,
            &idx2node,
            vtx2xy.len() / 2,
        );
        let num_tri = tri2vtx.len() / 3;
        for (i_bedge, node2bedge) in bedge2bedge.chunks(2).enumerate() {
            for i_node in 0..2 {
                let j_bedge = node2bedge[i_node];
                if i_bedge > j_bedge {
                    continue;
                }
                vedge2pnt.extend(&[i_bedge + num_tri, j_bedge + num_tri]);
            }
        }
        vedge2pnt
    };
    crate::io_obj::save_edge2vtx_vtx2xyz(
        "../target/voronoi_sites_on_edge.obj",
        &vedge2pnt,
        &pnt2xy,
        2,
    )
    .unwrap();
}
