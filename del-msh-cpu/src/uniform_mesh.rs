pub fn merge<T>(
    out_elem2vtx: &mut Vec<usize>,
    out_vtx2xyz: &mut Vec<T>,
    elem2vtx: &[usize],
    vtx2xyz: &[T],
    num_dim: usize,
) where
    T: Copy,
{
    let num_vtx0 = out_vtx2xyz.len() / num_dim;
    elem2vtx
        .iter()
        .for_each(|&v| out_elem2vtx.push(num_vtx0 + v));
    vtx2xyz.iter().for_each(|&v| out_vtx2xyz.push(v));
}

pub fn merge_with_vtx2rgb<T>(
    out_elem2vtx: &mut Vec<usize>,
    out_vtx2xyz: &mut Vec<T>,
    out_vtx2rgb: &mut Vec<T>,
    elem2vtx: &[usize],
    vtx2xyz: &[T],
    vtx2rgb: &[T],
    num_dim: usize,
) where
    T: Copy,
{
    let num_vtx0 = out_vtx2xyz.len() / num_dim;
    elem2vtx
        .iter()
        .for_each(|&v| out_elem2vtx.push(num_vtx0 + v));
    vtx2xyz.iter().for_each(|&v| out_vtx2xyz.push(v));
    vtx2rgb.iter().for_each(|&v| out_vtx2rgb.push(v));
}

pub fn vtx2vtx(
    elem2vtx: &[usize],
    num_node: usize,
    num_vtx: usize,
    is_self: bool,
) -> (Vec<usize>, Vec<usize>) {
    match num_node {
        2 => crate::vtx2vtx::from_uniform_mesh(elem2vtx.as_chunks::<2>().0, num_vtx, is_self),
        3 => crate::vtx2vtx::from_uniform_mesh(elem2vtx.as_chunks::<3>().0, num_vtx, is_self),
        4 => crate::vtx2vtx::from_uniform_mesh(elem2vtx.as_chunks::<4>().0, num_vtx, is_self),
        _ => panic!("unsupported num_node: {num_node}"),
    }
}

/// Compute vertex-to-element adjacency.
/// Returns (vtx2idx, idx2elem) where vtx2idx gives index ranges for each vertex's adjacent elements.
pub fn vtx2elem(elem2vtx: &[usize], num_node: usize, num_vtx: usize) -> (Vec<usize>, Vec<usize>) {
    match num_node {
        2 => crate::vtx2elem::from_uniform_mesh(elem2vtx.as_chunks::<2>().0, num_vtx),
        3 => crate::vtx2elem::from_uniform_mesh(elem2vtx.as_chunks::<3>().0, num_vtx),
        4 => crate::vtx2elem::from_uniform_mesh(elem2vtx.as_chunks::<4>().0, num_vtx),
        _ => panic!("unsupported num_node: {num_node}"),
    }
}

/// Compute element-to-element adjacency through shared faces.
/// Returns flattened array where each element has one neighbor index per face.
pub fn elem2elem(
    elem2vtx: &[usize],
    num_node: usize,
    face2idx_offset: &[usize],
    idx2node: &[usize],
    num_vtx: usize,
) -> Vec<usize> {
    match num_node {
        2 => crate::elem2elem::from_uniform_mesh(
            elem2vtx.as_chunks::<2>().0,
            face2idx_offset,
            idx2node,
            num_vtx,
        ),
        3 => crate::elem2elem::from_uniform_mesh(
            elem2vtx.as_chunks::<3>().0,
            face2idx_offset,
            idx2node,
            num_vtx,
        ),
        4 => crate::elem2elem::from_uniform_mesh(
            elem2vtx.as_chunks::<4>().0,
            face2idx_offset,
            idx2node,
            num_vtx,
        ),
        _ => panic!("unsupported num_node: {num_node}"),
    }
}
