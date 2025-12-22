use num_traits::AsPrimitive;

/// Compute the two adjacent triangle indices for each edge in a triangle mesh.
///
/// This function finds which triangles share each edge, returning up to 2 triangle indices
/// per edge. For boundary edges, only one triangle will be found.
///
/// # Arguments
/// * `edge2vtx` - Flattened array where each pair represents an edge (vertex indices)
/// * `tri2vtx` - Triangle connectivity array (3 vertices per triangle)  
/// * `vtx2idx` - Index array for vertex-to-triangle lookup table
/// * `idx2tri` - Triangle indices for each vertex's adjacent triangles
///
/// # Returns
/// * `Vec<INDEX>` - Flattened array of triangle indices (2 per edge, INDEX::max_value() for missing)
///
/// # Assumptions
/// * No T-junctions in the mesh (each edge shared by at most 2 triangles)
/// * Valid mesh topology with consistent vertex ordering
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
    // Initialize result array with max values (indicating no triangle found)
    let mut edge2tri = vec![INDEX::max_value(); edge2vtx.len()];

    // Process each edge
    for (i_edge, node2vtx) in edge2vtx.chunks(2).enumerate() {
        let (i0_vtx, i1_vtx) = (node2vtx[0], node2vtx[1]);
        let (i0_vtx, i1_vtx): (usize, usize) = (i0_vtx.as_(), i1_vtx.as_());
        let mut i_cnt = 0; // Count of adjacent triangles found for this edge
                           // Iterate through triangles adjacent to the first vertex of the edge
        for &i_tri in &idx2tri[vtx2idx[i0_vtx].as_()..vtx2idx[i0_vtx + 1].as_()] {
            let i_tri = i_tri.as_();
            // Get the three vertices of the current triangle
            let (j0_vtx, j1_vtx, j2_vtx) = (
                tri2vtx[i_tri * 3],
                tri2vtx[i_tri * 3 + 1],
                tri2vtx[i_tri * 3 + 2],
            );
            let (j0_vtx, j1_vtx, j2_vtx) = (j0_vtx.as_(), j1_vtx.as_(), j2_vtx.as_());
            // Check if this triangle contains the edge by finding which vertex position
            // matches i0_vtx, then checking if either of the other two matches i1_vtx
            let is_adjacent_edge = match (i0_vtx == j0_vtx, i0_vtx == j1_vtx, i0_vtx == j2_vtx) {
                (true, false, false) => (j1_vtx == i1_vtx) || (j2_vtx == i1_vtx), // i0_vtx at position 0
                (false, true, false) => (j2_vtx == i1_vtx) || (j0_vtx == i1_vtx), // i0_vtx at position 1
                (false, false, true) => (j0_vtx == i1_vtx) || (j1_vtx == i1_vtx), // i0_vtx at position 2
                _ => unreachable!(), // i0_vtx must be exactly one of the triangle vertices
            };
            if !is_adjacent_edge {
                continue;
            }
            // Store the triangle index for this edge
            edge2tri[i_edge * 2 + i_cnt] = i_tri.as_();
            i_cnt += 1;
            // Stop after finding 2 triangles (manifold edge assumption)
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
