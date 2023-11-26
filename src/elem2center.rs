pub fn from_uniform_mesh<T>(
    elem2vtx: &[usize],
    num_node: usize,
    vtx2xyz: &[T],
    num_dim: usize) -> Vec<T>
    where T: num_traits::Float + 'static + Copy + std::ops::AddAssign,
          f64: num_traits::AsPrimitive<T>,
        usize: num_traits::AsPrimitive<T>
{
    use num_traits::AsPrimitive;
    let num_elem = elem2vtx.len() / num_node;
    assert_eq!(elem2vtx.len(), num_elem*num_node);
    let mut elem2cog = Vec::<T>::with_capacity(num_elem*num_dim);
    let ratio: T = T::one() / num_node.as_();
    let mut cog = vec!(T::zero();num_node);
    for node2vtx in elem2vtx.chunks(num_node) {
        for idim in 0..num_dim {
            cog[idim] = 0_f64.as_();
        }
        for inode in 0..num_node {
            let i_vtx = node2vtx[inode];
            for idim in 0..num_dim {
                cog[idim] += vtx2xyz[i_vtx * num_node + idim];
            }
        }
        for idim in 0..num_dim {
            elem2cog.push(cog[idim] * ratio);
        }
    }
    elem2cog
}