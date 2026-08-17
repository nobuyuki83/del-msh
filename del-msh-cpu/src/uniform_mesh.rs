pub fn merge<T, const NNODE: usize, const NDIM: usize>(
    out_elem2vtx: &mut Vec<[usize; NNODE]>,
    out_vtx2xyz: &mut Vec<[T; NDIM]>,
    elem2vtx: &[[usize; NNODE]],
    vtx2xyz: &[[T; NDIM]],
) where
    T: Copy,
{
    let num_vtx0 = out_vtx2xyz.len();
    elem2vtx
        .iter()
        .for_each(|elem| out_elem2vtx.push(elem.map(|v| v + num_vtx0)));
    out_vtx2xyz.extend_from_slice(vtx2xyz);
}

pub fn merge_with_vtx2rgb<T, const NNODE: usize, const NDIM: usize, const NCHANNEL: usize>(
    out_elem2vtx: &mut Vec<[usize; NNODE]>,
    out_vtx2xyz: &mut Vec<[T; NDIM]>,
    out_vtx2rgb: &mut Vec<[T; NCHANNEL]>,
    elem2vtx: &[[usize; NNODE]],
    vtx2xyz: &[[T; NDIM]],
    vtx2rgb: &[[T; NCHANNEL]],
) where
    T: Copy,
{
    let num_vtx0 = out_vtx2xyz.len();
    elem2vtx
        .iter()
        .for_each(|elem| out_elem2vtx.push(elem.map(|v| v + num_vtx0)));
    out_vtx2xyz.extend_from_slice(vtx2xyz);
    out_vtx2rgb.extend_from_slice(vtx2rgb);
}

pub fn vtx2vtx<const NNODE: usize>(
    elem2vtx: &[[usize; NNODE]],
    num_vtx: usize,
    is_self: bool,
) -> (Vec<usize>, Vec<usize>) {
    crate::vtx2vtx::from_uniform_mesh(elem2vtx, num_vtx, is_self)
}

/// Compute vertex-to-element adjacency.
/// Returns (vtx2idx, idx2elem) where vtx2idx gives index ranges for each vertex's adjacent elements.
pub fn vtx2elem<const NNODE: usize>(
    elem2vtx: &[[usize; NNODE]],
    num_vtx: usize,
) -> (Vec<usize>, Vec<usize>) {
    crate::vtx2elem::from_uniform_mesh(elem2vtx, num_vtx)
}

/// Compute element-to-element adjacency through shared faces.
/// Returns flattened array where each element has one neighbor index per face.
pub fn elem2elem<const NNODE: usize, const NFACE: usize>(
    elem2vtx: &[[usize; NNODE]],
    face2idx_offset: &[usize],
    idx2node: &[usize],
    num_vtx: usize,
) -> Vec<usize> {
    crate::elem2elem::from_uniform_mesh::<_, NNODE, NFACE>(
        elem2vtx,
        face2idx_offset,
        idx2node,
        num_vtx,
    )
    .into_flattened()
}
