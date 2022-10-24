
/// mark elements with "idx_group" that is connected to "idx_elem_kernel"
/// * elem_group - array of group index for element
/// * elem_adjelem - array of adjacent element for element
pub fn mark_connected_elements(
    elem_group: &mut [usize],
    idx_elem_kernel: usize,
    idx_group: usize,
    elem_adjelem: &[usize]) {
    let num_elem = elem_group.len();
    let num_face_par_elem = elem_adjelem.len() / num_elem;
    elem_group[idx_elem_kernel] = idx_group;
    let mut next= vec!(idx_elem_kernel);
    while !next.is_empty() {
        let i_elem0 = next.pop().unwrap();
        for ie in 0..num_face_par_elem {
            let ita = elem_adjelem[i_elem0 * num_face_par_elem + ie];
            if ita == usize::MAX {
                continue;
            }
            if elem_group[ita] != idx_group {
                elem_group[ita] = idx_group;
                next.push(ita);
            }
        }
    }
}


/// * num_node - number of vertices par element
pub fn make_group_elem(
    elem_vtx: &[usize],
    num_node: usize,
    elem_adjelem: &[usize]) -> (usize, Vec<usize>)
{
    let nelem = elem_vtx.len() / num_node;
    let mut elem_group = vec!(usize::MAX; nelem);
    let mut i_group = 0;
    loop {
        let mut itri_ker = usize::MAX;
        for itri in 0..nelem {
            if elem_group[itri] == usize::MAX {
                itri_ker = itri;
                break;
            }
        }
        if itri_ker == usize::MAX { break; }
        mark_connected_elements(
            &mut elem_group,
            itri_ker, i_group, elem_adjelem);
        i_group += 1;
    }
    (i_group + 1, elem_group)
}