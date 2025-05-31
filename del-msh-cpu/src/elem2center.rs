//! compute the centers of elements in the mesh used mainly for constructing spatial hash

use num_traits::AsPrimitive;

// TODO: implement from_polygon_mesh_as_edges
// TODO: implement from_polygon_mesh_as_faces

pub fn update_from_uniform_mesh_as_points<Index, T>(
    elem2center: &mut [T],
    elem2vtx: &[Index],
    num_node: usize,
    vtx2xyz: &[T],
    num_dim: usize,
) where
    T: num_traits::Float + 'static + Copy + std::ops::AddAssign,
    usize: AsPrimitive<T>,
    Index: AsPrimitive<usize>,
{
    let num_elem = elem2vtx.len() / num_node;
    assert_eq!(elem2center.len(), num_elem * num_dim);
    assert_eq!(vtx2xyz.len() % num_dim, 0);
    assert_eq!(elem2vtx.len(), num_elem * num_node);
    let ratio: T = T::one() / num_node.as_();
    let mut cog = vec![T::zero(); num_dim];
    for (i_elem, node2vtx) in elem2vtx.chunks(num_node).enumerate() {
        cog.fill(T::zero());
        for i_vtx in &node2vtx[0..num_node] {
            let i_vtx: usize = i_vtx.as_();
            for idim in 0..num_dim {
                cog[idim] += vtx2xyz[i_vtx * num_dim + idim];
            }
        }
        for idim in 0..num_dim {
            elem2center[i_elem * num_dim + idim] = cog[idim] * ratio;
        }
    }
}

pub fn from_uniform_mesh_as_points<Index, T>(
    elem2vtx: &[Index],
    num_node: usize,
    vtx2xyz: &[T],
    num_dim: usize,
) -> Vec<T>
where
    T: num_traits::Float + 'static + Copy + std::ops::AddAssign,
    usize: AsPrimitive<T>,
    Index: AsPrimitive<usize>,
{
    assert_eq!(vtx2xyz.len() % num_dim, 0);
    let num_elem = elem2vtx.len() / num_node;
    assert_eq!(elem2vtx.len(), num_elem * num_node);
    let mut elem2center = vec![T::zero(); num_elem * num_dim];
    update_from_uniform_mesh_as_points(&mut elem2center, elem2vtx, num_node, vtx2xyz, num_dim);
    elem2center
}

/// the center of gravity of each element where mass is lumped at the vertices
pub fn from_polygon_mesh_as_points<T>(
    elem2idx: &[usize],
    idx2vtx: &[usize],
    vtx2xyz: &[T],
    num_dim: usize,
) -> Vec<T>
where
    T: num_traits::Float + 'static + Copy + std::ops::AddAssign,
    usize: AsPrimitive<T>,
{
    let mut cog = vec![T::zero(); num_dim];
    let num_elem = elem2idx.len() - 1;
    let mut elem2cog = Vec::<T>::with_capacity(num_elem * num_dim);
    for i_elem in 0..num_elem {
        cog.fill(T::zero());
        let num_vtx_in_elem = elem2idx[i_elem + 1] - elem2idx[i_elem];
        for i_vtx0 in &idx2vtx[elem2idx[i_elem]..elem2idx[i_elem + 1]] {
            for idim in 0..num_dim {
                cog[idim] += vtx2xyz[i_vtx0 * num_dim + idim];
            }
        }
        let ratio = if num_vtx_in_elem == 0 {
            T::zero()
        } else {
            T::one() / num_vtx_in_elem.as_()
        };
        cog.iter().for_each(|&v| elem2cog.push(v * ratio));
    }
    elem2cog
}
