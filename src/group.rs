//! group elements using the connectivity information

/// mark elements with "idx_group" that is connected to "idx_elem_kernel"
/// * `elem_group` - array of group index for element
/// * `elem_adjelem` - array of adjacent element for element
pub fn mark_connected_elements(
    elem2group: &mut [usize],
    idx_elem_kernel: usize,
    idx_group: usize,
    elem2elem_adj: &[usize]) {
    let num_elem = elem2group.len();
    assert_eq!(elem2elem_adj.len() % num_elem, 0);
    let num_face_par_elem = elem2elem_adj.len() / num_elem;
    elem2group[idx_elem_kernel] = idx_group;
    let mut next = vec!(idx_elem_kernel);
    while !next.is_empty() {
        let i_elem0 = next.pop().unwrap();
        for ie in 0..num_face_par_elem {
            let ita = elem2elem_adj[i_elem0 * num_face_par_elem + ie];
            if ita == usize::MAX {
                continue;
            }
            if elem2group[ita] != idx_group {
                elem2group[ita] = idx_group;
                next.push(ita);
            }
        }
    }
}

/// group uniform mesh
/// * `elem2vtx` - the indices of vertex for each element
/// * `num_node` - number of vertices par element
/// * `elem2elem_adj` - the adjacent element index of an element
pub fn make_group_elem(
    elem2vtx: &[usize],
    num_node: usize,
    elem2elem_adj: &[usize]) -> (usize, Vec<usize>)
{
    let nelem = elem2vtx.len() / num_node;
    let mut elem2group = vec!(usize::MAX; nelem);
    let mut i_group = 0;
    loop {
        let mut itri_ker = usize::MAX;
        for itri in 0..nelem {
            if elem2group[itri] == usize::MAX {
                itri_ker = itri;
                break;
            }
        }
        if itri_ker == usize::MAX { break; }
        mark_connected_elements(
            &mut elem2group,
            itri_ker, i_group, elem2elem_adj);
        i_group += 1;
    }
    (i_group + 1, elem2group)
}