//! methods related to unindexed mesh (e.g., array of vertex coordinates)

pub fn unidex_vertex_attribute_for_triangle_mesh<Index>(
    tri2vtx: &[[Index; 3]],
    vtx2val: &[[f32; 3]],
) -> Vec<[f32; 3]>
where
    Index: num_traits::AsPrimitive<usize>,
{
    let num_tri = tri2vtx.len();
    let mut tri2node2val = vec![[0f32; 3]; num_tri * 3];
    for i_tri in 0..num_tri {
        for i_node in 0..3 {
            let i_vtx: usize = tri2vtx[i_tri][i_node].as_();
            tri2node2val[i_tri * 3 + i_node] = vtx2val[i_vtx];
        }
    }
    tri2node2val
}
