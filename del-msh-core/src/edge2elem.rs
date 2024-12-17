use num_traits::AsPrimitive;

/// two adjacent triangles indices for each edge
/// assuming there is no t-junction
pub fn from_edge2vtx_of_tri2vtx_with_vtx2vtx<INDEX>(
    edge2vtx: &[INDEX],
    tri2vtx: &[INDEX],
    vtx2idx: &[INDEX],
    idx2tri: &[INDEX],
) -> Vec<INDEX>
where
    INDEX: num_traits::PrimInt + 'static + AsPrimitive<usize>,
    usize: num_traits::AsPrimitive<INDEX>,
{
    use num_traits::AsPrimitive;
    let mut edge2tri = vec![INDEX::max_value(); edge2vtx.len()];
    for (i_edge, node2vtx) in edge2vtx.chunks(2).enumerate() {
        let (i0_vtx, i1_vtx) = (node2vtx[0], node2vtx[1]);
        let (i0_vtx, i1_vtx): (usize, usize) = (i0_vtx.as_(), i1_vtx.as_());
        let mut i_cnt = 0;
        for &i_tri in &idx2tri[vtx2idx[i0_vtx].as_()..vtx2idx[i0_vtx + 1].as_()] {
            let i_tri = i_tri.as_();
            let (j0_vtx, j1_vtx, j2_vtx) = (
                tri2vtx[i_tri * 3],
                tri2vtx[i_tri * 3 + 1],
                tri2vtx[i_tri * 3 + 2],
            );
            let (j0_vtx, j1_vtx, j2_vtx) = (j0_vtx.as_(), j1_vtx.as_(), j2_vtx.as_());
            let is_adjacent_edge = match (i0_vtx == j0_vtx, i0_vtx == j1_vtx, i0_vtx == j2_vtx) {
                (true, false, false) => (j1_vtx == i1_vtx) || (j2_vtx == i1_vtx),
                (false, true, false) => (j2_vtx == i1_vtx) || (j0_vtx == i1_vtx),
                (false, false, true) => (j0_vtx == i1_vtx) || (j1_vtx == i1_vtx),
                _ => unreachable!(),
            };
            if !is_adjacent_edge {
                continue;
            }
            edge2tri[i_edge * 2 + i_cnt] = i_tri.as_();
            i_cnt += 1;
            if i_cnt == 2 {
                break;
            }
        }
    }
    edge2tri
}

pub fn from_edge2vtx_of_tri2vtx<INDEX>(
    edge2vtx: &[INDEX],
    tri2vtx: &[INDEX],
    num_vtx: usize,
) -> Vec<INDEX>
where
    INDEX: num_traits::PrimInt + std::ops::AddAssign + num_traits::AsPrimitive<usize>,
    usize: num_traits::AsPrimitive<INDEX>,
{
    let (vtx2idx, idx2tri) = crate::vtx2elem::from_uniform_mesh(tri2vtx, 3, num_vtx);
    from_edge2vtx_of_tri2vtx_with_vtx2vtx(edge2vtx, tri2vtx, &vtx2idx, &idx2tri)
}

#[test]
pub fn test_edge2tri() {
    let (tri2vtx, vtx2xyz) : (Vec<usize>, Vec<f32>)
        //= crate::trimesh3_primitive::capsule_yup(1., 2., 32, 32, 8);
        = crate::trimesh3_primitive::sphere_yup(1., 32, 32);
    let edge2vtx = crate::edge2vtx::from_triangle_mesh(tri2vtx.as_slice(), vtx2xyz.len() / 3);
    let edge2tri = from_edge2vtx_of_tri2vtx(&edge2vtx, &tri2vtx, vtx2xyz.len() / 3);
    edge2tri
        .iter()
        .for_each(|&i_tri| assert_ne!(i_tri, usize::MAX));
}
