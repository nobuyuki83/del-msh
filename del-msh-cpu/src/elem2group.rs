//! group elements using the connectivity information

/// mark elements with "idx_group" that is connected to "idx_elem_kernel"
/// * `elem_group` - array of group index for element
/// * `elem_adjelem` - array of adjacent element for element
pub fn mark_connected_elements_for_uniform_mesh<const NFACE: usize>(
    elem2group: &mut [usize],
    idx_elem_kernel: usize,
    idx_group: usize,
    elem2elem_adj: &[[usize; NFACE]],
) {
    assert_eq!(elem2group.len(), elem2elem_adj.len());
    elem2group[idx_elem_kernel] = idx_group;
    let mut next = vec![idx_elem_kernel];
    while let Some(i_elem0) = next.pop() {
        for &i_elem_adj in elem2elem_adj[i_elem0].iter() {
            if i_elem_adj == usize::MAX {
                continue;
            }
            if elem2group[i_elem_adj] != idx_group {
                elem2group[i_elem_adj] = idx_group;
                next.push(i_elem_adj);
            }
        }
    }
}

/// group uniform mesh
/// * `elem2vtx` - the indices of vertex for each element
/// * `num_node` - number of vertices par element
/// * `elem2elem_adj` - the adjacent element index of an element
pub fn from_uniform_mesh_with_elem2elem<const NNODE: usize, const NFACE: usize>(
    elem2vtx: &[[usize; NNODE]],
    elem2adjelem: &[[usize; NFACE]],
) -> (usize, Vec<usize>) {
    assert_eq!(elem2vtx.len(), elem2adjelem.len());
    let nelem = elem2vtx.len();
    let mut elem2group = vec![usize::MAX; nelem];
    let mut i_group = 0;
    loop {
        let mut itri_ker = usize::MAX;
        for (i_tri, &group) in elem2group.iter().enumerate() {
            if group != usize::MAX {
                continue;
            }
            itri_ker = i_tri;
            break;
        }
        if itri_ker == usize::MAX {
            break;
        }
        mark_connected_elements_for_uniform_mesh(&mut elem2group, itri_ker, i_group, elem2adjelem);
        i_group += 1;
    }
    (i_group, elem2group)
}

pub fn from_triangle_mesh(tri2vtx: &[usize], num_vtx: usize) -> (usize, Vec<usize>) {
    let tri2tri = crate::elem2elem::from_uniform_mesh::<_, 3, 3>(
        tri2vtx.as_chunks::<3>().0,
        &del_geo_core::tri::FACE2IDX,
        &del_geo_core::tri::IDX2NODE,
        num_vtx,
    );
    from_uniform_mesh_with_elem2elem(tri2vtx.as_chunks::<3>().0, &tri2tri)
}

/// mark elements with "idx_group" that is connected to "idx_elem_kernel"
/// * `elem2group` - array of group index for element
/// * `elemface2adjelem` - function takes i_elem and i_face and output index of adjacent element
pub fn mark_connected_elements_for_polygon_mesh<F>(
    elem2group: &mut [usize],
    idx_elem_kernel: usize,
    idx_group: usize,
    elem2idx: &[usize],
    elemface2adjelem: F,
) where
    F: Fn(usize, usize) -> usize,
{
    let num_elem = elem2group.len();
    assert_eq!(num_elem, elem2idx.len() - 1);
    elem2group[idx_elem_kernel] = idx_group;
    let mut next = vec![idx_elem_kernel];
    while let Some(i_elem0) = next.pop() {
        let num_adjelem_for_ielem0 = elem2idx[i_elem0 + 1] - elem2idx[i_elem0];
        for i_face0 in 0..num_adjelem_for_ielem0 {
            let j_elem = elemface2adjelem(i_elem0, i_face0);
            if j_elem == usize::MAX {
                continue;
            }
            if elem2group[j_elem] != idx_group {
                elem2group[j_elem] = idx_group;
                next.push(j_elem);
            }
        }
    }
}

/// group uniform mesh
/// * `elem2vtx` - the indices of vertex for each element
/// * `num_node` - number of vertices par element
/// * `elem2elem_adj` - the adjacent element index of an element
pub fn from_polygon_mesh<F>(elem2idx: &[usize], elemface2adjelem: F) -> (usize, Vec<usize>)
where
    F: Fn(usize, usize) -> usize,
{
    let nelem = elem2idx.len() - 1;
    let mut elem2group = vec![usize::MAX; nelem];
    let mut i_group = 0;
    loop {
        let mut itri_ker = usize::MAX;
        for (i_tri, &group) in elem2group.iter().enumerate() {
            if group != usize::MAX {
                continue;
            }
            itri_ker = i_tri;
            break;
        }
        if itri_ker == usize::MAX {
            break;
        }
        mark_connected_elements_for_polygon_mesh(
            &mut elem2group,
            itri_ker,
            i_group,
            elem2idx,
            &elemface2adjelem,
        );
        i_group += 1;
    }
    (i_group, elem2group)
}
