//! compute the centers of elements in the mesh used mainly for constructing spatial hash

use num_traits::AsPrimitive;

// TODO: implement from_polygon_mesh_as_edges
// TODO: implement from_polygon_mesh_as_faces

pub fn update_from_uniform_mesh_as_points<Index, T, const NNODE: usize, const NDIM: usize>(
    elem2center: &mut [[T; NDIM]],
    elem2vtx: &[[Index; NNODE]],
    vtx2xyz: &[[T; NDIM]],
) where
    T: num_traits::Float + 'static + Copy + std::ops::AddAssign,
    usize: AsPrimitive<T>,
    Index: AsPrimitive<usize>,
{
    assert_eq!(elem2center.len(), elem2vtx.len());
    let ratio: T = T::one() / NNODE.as_();
    let mut cog = [T::zero(); NDIM];
    for (i_elem, node2vtx) in elem2vtx.iter().enumerate() {
        cog.fill(T::zero());
        for i_vtx in node2vtx.iter() {
            let i_vtx: usize = i_vtx.as_();
            for idim in 0..NDIM {
                cog[idim] += vtx2xyz[i_vtx][idim];
            }
        }
        for idim in 0..NDIM {
            elem2center[i_elem][idim] = cog[idim] * ratio;
        }
    }
}

pub fn from_uniform_mesh_as_points<Index, T, const NNODE: usize, const NDIM: usize>(
    elem2vtx: &[[Index; NNODE]],
    vtx2xyz: &[[T; NDIM]],
) -> Vec<[T; NDIM]>
where
    T: num_traits::Float + 'static + Copy + std::ops::AddAssign,
    usize: AsPrimitive<T>,
    Index: AsPrimitive<usize>,
{
    let mut elem2center = vec![[T::zero(); NDIM]; elem2vtx.len()];
    update_from_uniform_mesh_as_points(&mut elem2center, elem2vtx, vtx2xyz);
    elem2center
}

/// the center of gravity of each element where mass is lumped at the vertices
pub fn from_polygon_mesh_as_points<T, IDX, const NDIM: usize>(
    elem2idx_offset: &[IDX],
    idx2vtx: &[IDX],
    vtx2xyz: &[[T; NDIM]],
) -> Vec<[T; NDIM]>
where
    IDX: num_traits::PrimInt + AsPrimitive<usize>,
    T: num_traits::Float + 'static + Copy + std::ops::AddAssign,
    usize: AsPrimitive<T>,
{
    let mut cog = [T::zero(); NDIM];
    let num_elem = elem2idx_offset.len() - 1;
    let mut elem2cog = Vec::<[T; NDIM]>::with_capacity(num_elem);
    for i_elem in 0..num_elem {
        cog.fill(T::zero());
        let num_vtx_in_elem = (elem2idx_offset[i_elem + 1] - elem2idx_offset[i_elem]).as_();
        for &i_vtx0 in &idx2vtx[elem2idx_offset[i_elem].as_()..elem2idx_offset[i_elem + 1].as_()] {
            let i_vtx0: usize = i_vtx0.as_();
            for idim in 0..NDIM {
                cog[idim] += vtx2xyz[i_vtx0][idim];
            }
        }
        let ratio = if num_vtx_in_elem == 0 {
            T::zero()
        } else {
            T::one() / num_vtx_in_elem.as_()
        };
        elem2cog.push(cog.map(|v| v * ratio));
    }
    elem2cog
}
