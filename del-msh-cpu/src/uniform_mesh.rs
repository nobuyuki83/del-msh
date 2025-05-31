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
    crate::vtx2vtx::from_uniform_mesh(elem2vtx, num_node, num_vtx, is_self)
}

pub fn vtx2elem(elem2vtx: &[usize], num_node: usize, num_vtx: usize) -> (Vec<usize>, Vec<usize>) {
    crate::vtx2elem::from_uniform_mesh(elem2vtx, num_node, num_vtx)
}

pub fn elem2elem(
    elem2vtx: &[usize],
    num_node: usize,
    face2idx: &[usize],
    idx2node: &[usize],
    num_vtx: usize,
) -> Vec<usize> {
    crate::elem2elem::from_uniform_mesh(elem2vtx, num_node, face2idx, idx2node, num_vtx)
}
