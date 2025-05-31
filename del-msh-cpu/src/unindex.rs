//! methods related to unindexed mesh (e.g., array of vertex coordinates)

pub fn unidex_vertex_attribute_for_triangle_mesh<Index>(
    tri2vtx: &[Index],
    vtx2val: &[f32],
    num_val: usize,
) -> Vec<f32>
where
    Index: num_traits::AsPrimitive<usize>,
{
    let num_tri = tri2vtx.len() / 3;
    let mut tri2node2val = vec![0_f32; num_tri * 3 * num_val];
    for i_tri in 0..num_tri {
        for i_node in 0..3 {
            let i_vtx: usize = tri2vtx[i_tri * 3 + i_node].as_();
            for i_val in 0..num_val {
                let i0 = i_tri * 3 * num_val + i_node * num_val + i_val;
                tri2node2val[i0] = vtx2val[i_vtx * num_val + i_val];
            }
        }
    }
    tri2node2val
}
