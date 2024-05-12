pub fn merge<T>(
    out_elem2vtx: &mut Vec<usize>,
    out_vtx2xyz: &mut Vec<T>,
    elem2vtx: &[usize],
    vtx2xyz: &[T],
    num_dim: usize)
    where T: Copy
{
    let num_vtx0 = out_vtx2xyz.len() / num_dim;
    elem2vtx.iter().for_each(|&v| out_elem2vtx.push(num_vtx0 + v));
    vtx2xyz.iter().for_each(|&v| out_vtx2xyz.push(v));
}