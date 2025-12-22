//! Methods for creating edge connectivity (line mesh topology) from various mesh types

use num_traits::AsPrimitive;

/// Create edge connectivity from vertex-to-vertex adjacency information.
///
/// Converts vertex adjacency data into a flattened edge list where each pair of
/// consecutive elements represents an edge between two vertices.
///
/// # Arguments
/// * `vtx2idx` - Index boundaries for each vertex's adjacency list
/// * `idx2vtx` - Flattened list of adjacent vertices for all vertices
///
/// # Returns
/// * `Vec<Index>` - Flattened edge list (pairs of vertex indices)
pub fn from_vtx2vtx<Index>(vtx2idx: &[Index], idx2vtx: &[Index]) -> Vec<Index>
where
    Index: AsPrimitive<usize>,
    usize: AsPrimitive<Index>,
{
    // Pre-allocate assuming each vertex connects to multiple others
    let mut line2vtx = Vec::<Index>::with_capacity(idx2vtx.len() * 2);

    // Process each vertex and its adjacent vertices
    for i_vtx in 0..vtx2idx.len() - 1 {
        let idx0 = vtx2idx[i_vtx].as_();
        let idx1 = vtx2idx[i_vtx + 1].as_();
        // Add an edge from current vertex to each of its neighbors
        for &j_vtx in &idx2vtx[idx0..idx1] {
            line2vtx.push(i_vtx.as_()); // Source vertex
            line2vtx.push(j_vtx); // Target vertex
        }
    }
    line2vtx
}

/// Extract specific edges from a uniform mesh (elements with same number of nodes).
///
/// Creates edge connectivity by extracting only the specified edge patterns from
/// uniform mesh elements (e.g., triangles, quads, tetrahedra).
///
/// # Arguments
/// * `elem2vtx` - Element connectivity (flattened, num_node vertices per element)
/// * `num_node` - Number of nodes per element
/// * `edge2node` - Edge pattern as pairs of local node indices within elements
/// * `num_vtx` - Total number of vertices in the mesh
///
/// # Returns
/// * `Vec<Index>` - Flattened edge connectivity
pub fn from_uniform_mesh_with_specific_edges<Index>(
    elem2vtx: &[Index],
    num_node: usize,
    edge2node: &[usize],
    num_vtx: usize,
) -> Vec<Index>
where
    Index: num_traits::PrimInt + std::ops::AddAssign + AsPrimitive<usize>,
    usize: AsPrimitive<Index>,
{
    // Build vertex-to-element adjacency lookup
    let vtx2elem = crate::vtx2elem::from_uniform_mesh::<Index>(elem2vtx, num_node, num_vtx);
    // Extract vertex-to-vertex connectivity for specified edge patterns
    let vtx2vtx = crate::vtx2vtx::from_specific_edges_of_uniform_mesh::<Index>(
        elem2vtx,
        num_node,
        edge2node,
        &vtx2elem.0,
        &vtx2elem.1,
        false, // Don't include duplicate edges
    );
    // Convert vertex adjacency to edge list
    from_vtx2vtx(&vtx2vtx.0, &vtx2vtx.1)
}

/// Extract all edges from a triangle mesh.
///
/// Creates edge connectivity from triangle mesh by extracting all three edges
/// of each triangle: (0,1), (1,2), and (2,0).
///
/// # Arguments
/// * `tri2vtx` - Triangle connectivity (3 vertices per triangle)
/// * `num_vtx` - Total number of vertices
///
/// # Returns
/// * `Vec<INDEX>` - Flattened edge connectivity
pub fn from_triangle_mesh<INDEX>(tri2vtx: &[INDEX], num_vtx: usize) -> Vec<INDEX>
where
    INDEX: num_traits::PrimInt + std::ops::AddAssign + num_traits::AsPrimitive<usize>,
    usize: AsPrimitive<INDEX>,
{
    // Extract triangle edges: vertex 0-1, 1-2, 2-0
    from_uniform_mesh_with_specific_edges(tri2vtx, 3, &[0, 1, 1, 2, 2, 0], num_vtx)
}

/// Extract edges from a mesh with mixed polygonal elements.
///
/// Handles meshes containing a mixture of triangles, quadrilaterals, pentagons, etc.
/// Each polygon contributes its boundary edges to the result.
///
/// # Arguments
/// * `elem2idx` - Element boundaries (start index for each element's vertices)
/// * `idx2vtx` - Flattened vertex indices for all elements
/// * `num_vtx` - Total number of vertices
///
/// # Returns
/// * `Vec<usize>` - Flattened edge connectivity
pub fn from_polygon_mesh(elem2idx: &[usize], idx2vtx: &[usize], num_vtx: usize) -> Vec<usize> {
    // Build vertex-to-element adjacency for polygon mesh
    let vtx2elem = crate::vtx2elem::from_polygon_mesh(elem2idx, idx2vtx, num_vtx);
    // Extract vertex-to-vertex connectivity from polygon edges
    let vtx2vtx = crate::vtx2vtx::from_polygon_mesh_edges_with_vtx2elem(
        elem2idx,
        idx2vtx,
        &vtx2elem.0,
        &vtx2elem.1,
        false, // Don't include duplicate edges
    );
    // Convert to edge list
    from_vtx2vtx(&vtx2vtx.0, &vtx2vtx.1)
}

/// Create edge connectivity for a closed polygon loop.
///
/// Generates edges connecting consecutive vertices in a loop, with the last vertex
/// connecting back to the first vertex to close the loop.
///
/// # Arguments
/// * `num_vtx` - Number of vertices in the loop
///
/// # Returns
/// * `Vec<usize>` - Edge connectivity for closed loop
pub fn from_polyloop(num_vtx: usize) -> Vec<usize> {
    let mut edge2vtx = Vec::<usize>::with_capacity(num_vtx * 2);
    for i in 0..num_vtx {
        edge2vtx.push(i);
        edge2vtx.push((i + 1) % num_vtx); // Wrap around to close the loop
    }
    edge2vtx
}

/// Create edge connectivity for an open polyline.
///
/// Generates edges connecting consecutive vertices in sequence, without closing
/// the loop (no edge from last to first vertex).
///
/// # Arguments
/// * `num_vtx` - Number of vertices in the polyline
///
/// # Returns
/// * `Vec<usize>` - Edge connectivity for open polyline
pub fn from_polyline(num_vtx: usize) -> Vec<usize> {
    let mut edge2vtx = Vec::<usize>::with_capacity((num_vtx - 1) * 2);
    for i in 0..num_vtx - 1 {
        edge2vtx.push(i);
        edge2vtx.push(i + 1); // Connect to next vertex (no wrap-around)
    }
    edge2vtx
}

// -----------
// Contour and silhouette extraction functions
// -----------

/// Extract contour edges from a triangle mesh based on viewing direction.
///
/// Finds edges where adjacent triangles have opposite orientations relative to the
/// viewing direction (one facing toward, one facing away from the viewer).
///
/// # Arguments
/// * `tri2vtx` - Triangle connectivity
/// * `vtx2xyz` - Vertex coordinates (3D)
/// * `transform_world2ndc` - World-to-NDC transformation matrix (4x4)
/// * `edge2vtx` - Edge connectivity
/// * `edge2tri` - Edge-to-triangle adjacency
///
/// # Returns
/// * `Vec<INDEX>` - Contour edge connectivity
pub fn contour_for_triangle_mesh<INDEX>(
    tri2vtx: &[INDEX],
    vtx2xyz: &[f32],
    transform_world2ndc: &[f32; 16],
    edge2vtx: &[INDEX],
    edge2tri: &[INDEX],
) -> Vec<INDEX>
where
    INDEX: num_traits::PrimInt + num_traits::AsPrimitive<usize> + std::fmt::Display,
    usize: AsPrimitive<INDEX>,
{
    use del_geo_core::{mat4_col_major, vec3};
    let num_tri = tri2vtx.len() / 3;
    let transform_ndc2world = mat4_col_major::try_inverse(transform_world2ndc).unwrap();
    let mut edge2vtx_contour: Vec<INDEX> = vec![];

    // Check each edge for contour condition
    for (i_edge, node2vtx) in edge2vtx.chunks(2).enumerate() {
        let (i0_vtx, i1_vtx) = (node2vtx[0].as_(), node2vtx[1].as_());
        // Calculate midpoint of edge for ray casting
        let pos_mid: [f32; 3] =
            std::array::from_fn(|i| (vtx2xyz[i0_vtx * 3 + i] + vtx2xyz[i1_vtx * 3 + i]) * 0.5);
        // Get viewing direction at this point
        let (_ray_org, ray_dir) = mat4_col_major::ray_from_transform_world2ndc(
            transform_world2ndc,
            &pos_mid,
            &transform_ndc2world,
        );

        // Get the two triangles adjacent to this edge
        let i0_tri = edge2tri[i_edge * 2];
        let i1_tri = edge2tri[i_edge * 2 + 1];
        assert!(i0_tri.as_() < num_tri, "{} {}", i0_tri, tri2vtx.len() / 3);
        assert!(i1_tri.as_() < num_tri, "{} {}", i1_tri, tri2vtx.len() / 3);

        // Calculate normal vectors of adjacent triangles
        let nrm0_world = crate::trimesh3::to_tri3(tri2vtx, vtx2xyz, i0_tri.as_()).unit_normal();
        let nrm1_world = crate::trimesh3::to_tri3(tri2vtx, vtx2xyz, i1_tri.as_()).unit_normal();

        // Check if triangles face opposite directions relative to viewing direction
        let flg0 = vec3::dot(&nrm0_world, &ray_dir) > 0.; // Triangle 0 facing toward/away from viewer
        let flg1 = vec3::dot(&nrm1_world, &ray_dir) > 0.; // Triangle 1 facing toward/away from viewer

        // Include edge if triangles have opposite orientations (contour edge)
        if flg0 == flg1 {
            continue; // Both triangles face same direction, not a contour edge
        }
        edge2vtx_contour.push(i0_vtx.as_());
        edge2vtx_contour.push(i1_vtx.as_());
    }
    edge2vtx_contour
}

/// Extract occluding contour edges (visible contour) from a triangle mesh.
///
/// Similar to contour extraction but additionally performs occlusion testing to
/// include only edges that are visible from the viewing direction.
///
/// # Arguments
/// * `tri2vtx` - Triangle connectivity
/// * `vtx2xyz` - Vertex coordinates (3D)
/// * `transform_world2ndc` - World-to-NDC transformation matrix
/// * `edge2vtx` - Edge connectivity
/// * `edge2tri` - Edge-to-triangle adjacency
/// * `bvhnodes` - BVH acceleration structure nodes
/// * `bvhnode2aabb` - BVH node bounding boxes
///
/// # Returns
/// * `Vec<usize>` - Visible contour edge connectivity
pub fn occluding_contour_for_triangle_mesh(
    tri2vtx: &[usize],
    vtx2xyz: &[f32],
    transform_world2ndc: &[f32; 16],
    edge2vtx: &[usize],
    edge2tri: &[usize],
    bvhnodes: &[usize],
    bvhnode2aabb: &[f32],
) -> Vec<usize> {
    use del_geo_core::{mat4_col_major, vec3};
    let transform_ndc2world = mat4_col_major::try_inverse(transform_world2ndc).unwrap();
    let mut edge2vtx_contour = vec![];

    // Process each edge for visibility and contour conditions
    for (i_edge, node2vtx) in edge2vtx.chunks(2).enumerate() {
        let (i0_vtx, i1_vtx) = (node2vtx[0], node2vtx[1]);
        // Calculate edge midpoint for testing
        let pos_mid: [f32; 3] =
            std::array::from_fn(|i| (vtx2xyz[i0_vtx * 3 + i] + vtx2xyz[i1_vtx * 3 + i]) * 0.5);
        // Get viewing direction
        let (_ray_org, ray_dir) = mat4_col_major::ray_from_transform_world2ndc(
            transform_world2ndc,
            &pos_mid,
            &transform_ndc2world,
        );

        // Get adjacent triangles for this edge
        let i0_tri = edge2tri[i_edge * 2];
        let i1_tri = edge2tri[i_edge * 2 + 1];
        assert!(
            i0_tri < tri2vtx.len() / 3,
            "{} {}",
            i0_tri,
            tri2vtx.len() / 3
        );
        assert!(
            i1_tri < tri2vtx.len() / 3,
            "{} {}",
            i1_tri,
            tri2vtx.len() / 3
        );
        // Calculate triangle normals
        let nrm0_world = crate::trimesh3::to_tri3(tri2vtx, vtx2xyz, i0_tri).unit_normal();
        let nrm1_world = crate::trimesh3::to_tri3(tri2vtx, vtx2xyz, i1_tri).unit_normal();

        // First check if this is a contour edge (triangles face opposite directions)
        {
            let flg0 = vec3::dot(&nrm0_world, &ray_dir) > 0.;
            let flg1 = vec3::dot(&nrm1_world, &ray_dir) > 0.;
            if flg0 == flg1 {
                continue; // Not a contour edge
            }
        }

        // Perform occlusion test: cast ray from slightly offset position
        let ray_org = {
            let nrm = vec3::normalize(&vec3::add(&nrm0_world, &nrm1_world));
            vec3::axpy(0.001, &nrm, &pos_mid) // Small offset along average normal
        };
        let res = crate::search_bvh3::first_intersection_ray(
            &ray_org,
            &ray_dir,
            &crate::search_bvh3::TriMeshWithBvh {
                bvhnodes,
                bvhnode2aabb,
                tri2vtx,
                vtx2xyz,
            },
            0,
            f32::MAX,
        );
        if res.is_some() {
            continue; // Edge is occluded by another part of the mesh
        }
        edge2vtx_contour.push(i0_vtx);
        edge2vtx_contour.push(i1_vtx);
    }
    edge2vtx_contour
}

/// Extract silhouette edges from a triangle mesh.
///
/// Similar to occluding contour but uses a different occlusion test that checks for
/// any intersections along the ray (not just the first intersection).
///
/// # Arguments
/// * `tri2vtx` - Triangle connectivity
/// * `vtx2xyz` - Vertex coordinates (3D)  
/// * `transform_world2ndc` - World-to-NDC transformation matrix
/// * `edge2vtx` - Edge connectivity
/// * `edge2tri` - Edge-to-triangle adjacency
/// * `bvhnodes` - BVH acceleration structure nodes
/// * `bvhnode2aabb` - BVH node bounding boxes
///
/// # Returns
/// * `Vec<usize>` - Silhouette edge connectivity
pub fn silhouette_for_triangle_mesh(
    tri2vtx: &[usize],
    vtx2xyz: &[f32],
    transform_world2ndc: &[f32; 16],
    edge2vtx: &[usize],
    edge2tri: &[usize],
    bvhnodes: &[usize],
    bvhnode2aabb: &[f32],
) -> Vec<usize> {
    use del_geo_core::{mat4_col_major, vec3};
    let transform_ndc2world = mat4_col_major::try_inverse(transform_world2ndc).unwrap();
    let mut edge2vtx_contour = vec![];
    for (i_edge, node2vtx) in edge2vtx.chunks(2).enumerate() {
        let (i0_vtx, i1_vtx) = (node2vtx[0], node2vtx[1]);
        let pos_mid: [f32; 3] =
            std::array::from_fn(|i| (vtx2xyz[i0_vtx * 3 + i] + vtx2xyz[i1_vtx * 3 + i]) * 0.5);
        let (_ray_org, ray_dir) = mat4_col_major::ray_from_transform_world2ndc(
            transform_world2ndc,
            &pos_mid,
            &transform_ndc2world,
        );
        // -------
        let i0_tri = edge2tri[i_edge * 2];
        let i1_tri = edge2tri[i_edge * 2 + 1];
        assert!(
            i0_tri < tri2vtx.len() / 3,
            "{} {}",
            i0_tri,
            tri2vtx.len() / 3
        );
        assert!(
            i1_tri < tri2vtx.len() / 3,
            "{} {}",
            i1_tri,
            tri2vtx.len() / 3
        );
        let nrm0_world = crate::trimesh3::to_tri3(tri2vtx, vtx2xyz, i0_tri).unit_normal();
        let nrm1_world = crate::trimesh3::to_tri3(tri2vtx, vtx2xyz, i1_tri).unit_normal();
        // Check contour condition first
        {
            let flg0 = vec3::dot(&nrm0_world, &ray_dir) > 0.;
            let flg1 = vec3::dot(&nrm1_world, &ray_dir) > 0.;
            if flg0 == flg1 {
                continue; // Not a contour edge
            }
        }

        // Perform comprehensive occlusion test using line intersection
        let ray_org = {
            let nrm = vec3::normalize(&vec3::add(&nrm0_world, &nrm1_world));
            vec3::axpy(0.001, &nrm, &pos_mid) // Small offset along average normal
        };
        let mut res: Vec<(f32, usize)> = vec![];
        crate::search_bvh3::intersections_line(
            &mut res,
            &ray_org,
            &ray_dir,
            &crate::search_bvh3::TriMeshWithBvh {
                bvhnodes,
                bvhnode2aabb,
                tri2vtx,
                vtx2xyz,
            },
            0,
        );
        if !res.is_empty() {
            continue; // Edge is occluded (any intersection found)
        }
        edge2vtx_contour.push(i0_vtx);
        edge2vtx_contour.push(i1_vtx);
    }
    edge2vtx_contour
}

#[test]
pub fn test_contour() {
    let (tri2vtx, vtx2xyz)
        // = crate::trimesh3_primitive::sphere_yup::<usize, f32>(1., 32, 32);
        = crate::trimesh3_primitive::torus_zup(2.0, 0.5, 32, 32);
    let dir = del_geo_core::vec3::normalize(&[1f32, 1.0, 0.1]);
    let transform_world2ndc = {
        let ez = dir;
        let (ex, ey) = del_geo_core::vec3::basis_xy_from_basis_z(&ez);
        let m3 = del_geo_core::mat3_col_major::from_columns(&ex, &ey, &ez);
        let t = del_geo_core::mat4_col_major::from_mat3_col_major_adding_w(&m3, 1.0);
        del_geo_core::mat4_col_major::transpose(&t)
    };
    //
    let bvhnodes = crate::bvhnodes_morton::from_triangle_mesh(&tri2vtx, &vtx2xyz, 3);
    let bvhnode2aabb = crate::bvhnode2aabb3::from_uniform_mesh_with_bvh(
        0,
        &bvhnodes,
        Some((&tri2vtx, 3)),
        &vtx2xyz,
        None,
    );
    let edge2vtx = crate::edge2vtx::from_triangle_mesh(tri2vtx.as_slice(), vtx2xyz.len() / 3);
    let edge2tri =
        crate::edge2elem::from_edge2vtx_of_tri2vtx(&edge2vtx, &tri2vtx, vtx2xyz.len() / 3);

    {
        let edge2vtx_contour = occluding_contour_for_triangle_mesh(
            &tri2vtx,
            &vtx2xyz,
            &transform_world2ndc,
            &edge2vtx,
            &edge2tri,
            &bvhnodes,
            &bvhnode2aabb,
        );
        crate::io_wavefront_obj::save_edge2vtx_vtx2xyz(
            "../target/edge2vtx_countour.obj",
            &edge2vtx_contour,
            &vtx2xyz,
            3,
        )
        .unwrap();
    }
    {
        let edge2vtx_contour = silhouette_for_triangle_mesh(
            &tri2vtx,
            &vtx2xyz,
            &transform_world2ndc,
            &edge2vtx,
            &edge2tri,
            &bvhnodes,
            &bvhnode2aabb,
        );
        crate::io_wavefront_obj::save_edge2vtx_vtx2xyz(
            "../target/edge2vtx_silhouette.obj",
            &edge2vtx_contour,
            &vtx2xyz,
            3,
        )
        .unwrap();
    }
    crate::io_wavefront_obj::save_tri2vtx_vtx2xyz(
        "../target/edge2vtx_trimsh.obj",
        &tri2vtx,
        &vtx2xyz,
        3,
    )
    .unwrap();
}
