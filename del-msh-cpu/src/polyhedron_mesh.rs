use num_traits::AsPrimitive;
use std::collections::HashMap;
pub fn vtx2elem<INDEX>(
    elem2idx_offset: &[INDEX],
    idx2vtx: &[INDEX],
    num_vtx: usize,
) -> (Vec<INDEX>, Vec<INDEX>)
where
    INDEX: num_traits::PrimInt + num_traits::AsPrimitive<usize>,
    usize: AsPrimitive<INDEX>,
{
    crate::vtx2elem::from_polygon_mesh(elem2idx_offset, idx2vtx, num_vtx)
}

/// Compute the volume of each element in a mixed polyhedron mesh.
///
/// Supported element types (determined by node count per element):
/// - 4 nodes: tetrahedron
/// - 5 nodes: pyramid (square base + apex)
/// - 6 nodes: triangular prism
/// - 8 nodes: hex mesh
///
/// # Arguments
/// * `elem2idx_offset` - CSR offset array of length `num_elem + 1`; element `i` owns
///   vertices `idx2vtx[elem2idx_offset[i]..elem2idx_offset[i+1]]`
/// * `idx2vtx` - flat list of vertex indices for all elements
/// * `vtx2xyz` - vertex positions, stored as `[x, y, z, x, y, z, ...]`
/// * `i_gauss_degree` - Gauss quadrature degree used for pyramid and prism volume integration
/// * `elem2volume` - output slice of length `num_elem`; each entry is set to the element's volume
pub fn elem2volume<IDX, Real>(
    elem2idx_offset: &[IDX],
    idx2vtx: &[IDX],
    vtx2xyz: &[Real],
    i_gauss_degree: usize,
    elem2volume: &mut [Real],
) where
    IDX: num_traits::AsPrimitive<usize>,
    Real: num_traits::Float + 'static + std::fmt::Debug + std::iter::Sum,
    del_geo_core::quadrature_line::Quad<Real>: del_geo_core::quadrature_line::QuadratureLine<Real>,
    del_geo_core::quadrature_tri::Quad<Real>: del_geo_core::quadrature_tri::QuadratureTri<Real>,
{
    let num_elem = elem2idx_offset.len() - 1;
    assert_eq!(elem2volume.len(), num_elem);
    for i_elem in 0..elem2idx_offset.len() - 1 {
        let node2vtx = &idx2vtx[elem2idx_offset[i_elem].as_()..elem2idx_offset[i_elem + 1].as_()];
        let p = |i: usize| arrayref::array_ref![vtx2xyz, node2vtx[i].as_() * 3, 3];
        match node2vtx.len() {
            4 => {
                elem2volume[i_elem] = del_geo_core::tet::volume(p(0), p(1), p(2), p(3));
            }
            5 => {
                elem2volume[i_elem] =
                    del_geo_core::pyramid::volume(p(0), p(1), p(2), p(3), p(4), i_gauss_degree);
            }
            6 => {
                elem2volume[i_elem] =
                    del_geo_core::prism::volume(p(0), p(1), p(2), p(3), p(4), p(5), i_gauss_degree);
            }
            8 => {
                elem2volume[i_elem] = del_geo_core::hex::volume(
                    p(0),
                    p(1),
                    p(2),
                    p(3),
                    p(4),
                    p(5),
                    p(6),
                    p(7),
                    i_gauss_degree,
                );
            }
            _ => {
                unreachable!()
            }
        }
    }
}

/// Returns the nearest point on the element and the 3 parametric coordinates at that point.
/// - Tet (4):    (r0, r1, r2); r3 = 1-r0-r1-r2 is implicit
/// - Pyramid (5): (r, s, t) with r,s in [0,1], t in [0,1]
/// - Prism (6):   (r, s, t) with r,s>=0, r+s<=1, t in [0,1]
fn parametric_coord(query: &[f32; 3], node2vtx: &[u32], vtx2xyz: &[f32]) -> Option<[f32; 3]> {
    let shift = |i: usize| -> [f32; 3] {
        let p = arrayref::array_ref!(vtx2xyz, i * 3, 3);
        [p[0] - query[0], p[1] - query[1], p[2] - query[2]]
    };
    match node2vtx.len() {
        4 => {
            del_geo_core::tet::barycentric_coord_for_origin(
                &shift(node2vtx[0] as usize),
                &shift(node2vtx[1] as usize),
                &shift(node2vtx[2] as usize),
                &shift(node2vtx[3] as usize),
            ).map(|bc| [bc.0, bc.1, bc.2])
        }
        5 => del_geo_core::pyramid::parametric_coord_for_origin(
            &shift(node2vtx[0] as usize),
            &shift(node2vtx[1] as usize),
            &shift(node2vtx[2] as usize),
            &shift(node2vtx[3] as usize),
            &shift(node2vtx[4] as usize),
        ),
        6 => del_geo_core::prism::parametric_coord_for_origin(
            &shift(node2vtx[0] as usize),
            &shift(node2vtx[1] as usize),
            &shift(node2vtx[2] as usize),
            &shift(node2vtx[3] as usize),
            &shift(node2vtx[4] as usize),
            &shift(node2vtx[5] as usize),
        ),
        8 => del_geo_core::hex::parametric_coord_for_origin(
            &shift(node2vtx[0] as usize),
            &shift(node2vtx[1] as usize),
            &shift(node2vtx[2] as usize),
            &shift(node2vtx[3] as usize),
            &shift(node2vtx[4] as usize),
            &shift(node2vtx[5] as usize),
            &shift(node2vtx[6] as usize),
            &shift(node2vtx[7] as usize),
        ),
        n => panic!("unsupported element type with {n} vertices"),
    }
}

fn find_edges_for_subdiv<const NEDGE: usize>(
    edge2node: &[[usize; 2]; NEDGE],
    i_elem: usize,
    elem2idx_offset: &[u32],
    idx2vtx: &[u32],
    map_vtx2edge: &HashMap<[u32; 2], usize>,
    offset: usize,
) -> [u32; NEDGE] {
    std::array::from_fn(|i_edge| {
        let i0_node = edge2node[i_edge][0];
        let i1_node = edge2node[i_edge][1];
        let mut node2vtx = [
            idx2vtx[elem2idx_offset[i_elem] as usize + i0_node],
            idx2vtx[elem2idx_offset[i_elem] as usize + i1_node],
        ];
        node2vtx.sort();
        let i_edge = map_vtx2edge.get(&node2vtx).unwrap();
        (i_edge + offset) as u32
    })
}

pub fn subdivide(
    elem2idx_offset: &[u32],
    idx2vtx: &[u32],
    vtx2xyz: &[f32],
) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
    let vtx2elem = vtx2elem(elem2idx_offset, idx2vtx, vtx2xyz.len() / 3);
    let edge2vtx = edge2vtx_with_vtx2elem(elem2idx_offset, idx2vtx, &vtx2elem.0, &vtx2elem.1);
    let elem2elem =
        elem2elem_with_vtx2elem::<u32>(elem2idx_offset, idx2vtx, &vtx2elem.0, &vtx2elem.1);
    let quad2vtx = extract_quad_face(elem2idx_offset, idx2vtx, &elem2elem.0, &elem2elem.1);
    let num_elem = elem2idx_offset.len() - 1;
    let hex2elem: Vec<usize> = (0..num_elem)
        .filter(|&i_elem| elem2idx_offset[i_elem + 1] - elem2idx_offset[i_elem] == 8)
        .collect();
    //
    let num_vtx0 = vtx2xyz.len() / 3;
    let num_edge0 = edge2vtx.len() / 2;
    let num_quad0 = quad2vtx.len() / 4;
    let num_wtx = num_vtx0 + num_edge0 + num_quad0 + hex2elem.len();
    let wtx2xyz = {
        let mut wtx2xyz = Vec::<f32>::with_capacity(num_wtx * 3);
        wtx2xyz.extend_from_slice(vtx2xyz);
        use del_geo_core::vec3::Vec3;
        edge2vtx.chunks(2).for_each(|vtxs| {
            let p0 = arrayref::array_ref![vtx2xyz, vtxs[0] as usize * 3, 3];
            let p1 = arrayref::array_ref![vtx2xyz, vtxs[1] as usize * 3, 3];
            wtx2xyz.extend_from_slice(&p0.add(p1).scale(0.5));
        });
        quad2vtx.chunks(4).for_each(|vtxs| {
            let p0 = arrayref::array_ref![vtx2xyz, vtxs[0] as usize * 3, 3];
            let p1 = arrayref::array_ref![vtx2xyz, vtxs[1] as usize * 3, 3];
            let p2 = arrayref::array_ref![vtx2xyz, vtxs[2] as usize * 3, 3];
            let p3 = arrayref::array_ref![vtx2xyz, vtxs[3] as usize * 3, 3];
            wtx2xyz.extend_from_slice(&p0.add(p1).add(p2).add(p3).scale(0.25));
        });
        hex2elem.iter().for_each(|&i_elem| {
            let idx0 = elem2idx_offset[i_elem] as usize;
            let idx1 = elem2idx_offset[i_elem + 1] as usize;
            let vtxs = &idx2vtx[idx0..idx1];
            let p0 = arrayref::array_ref![vtx2xyz, vtxs[0] as usize * 3, 3];
            let p1 = arrayref::array_ref![vtx2xyz, vtxs[1] as usize * 3, 3];
            let p2 = arrayref::array_ref![vtx2xyz, vtxs[2] as usize * 3, 3];
            let p3 = arrayref::array_ref![vtx2xyz, vtxs[3] as usize * 3, 3];
            let p4 = arrayref::array_ref![vtx2xyz, vtxs[4] as usize * 3, 3];
            let p5 = arrayref::array_ref![vtx2xyz, vtxs[5] as usize * 3, 3];
            let p6 = arrayref::array_ref![vtx2xyz, vtxs[6] as usize * 3, 3];
            let p7 = arrayref::array_ref![vtx2xyz, vtxs[7] as usize * 3, 3];
            let p0123 = p0.add(p1).add(p2).add(p3);
            let p4567 = p4.add(p5).add(p6).add(p7);
            wtx2xyz.extend_from_slice(&p0123.add(&p4567).scale(0.125));
        });
        wtx2xyz
    };
    //
    let map_elem2hex: HashMap<usize, usize> = hex2elem
        .iter()
        .enumerate()
        .map(|(i_hex, &i_elem)| (i_elem, i_hex))
        .collect();
    let map_vtx2edge: HashMap<[u32; 2], usize> = edge2vtx
        .chunks(2)
        .enumerate()
        .map(|(i_edge, node2vtx)| {
            let mut a = [node2vtx[0], node2vtx[1]];
            a.sort();
            (a, i_edge)
        })
        .collect();
    let map_vtx2quad: HashMap<[u32; 4], usize> = quad2vtx
        .chunks(4)
        .enumerate()
        .map(|(i_edge, node2vtx)| {
            let mut a = [node2vtx[0], node2vtx[1], node2vtx[2], node2vtx[3]];
            a.sort();
            (a, i_edge)
        })
        .collect();
    // -------

    // -------
    let mut welem2jdx_offset: Vec<u32> = vec![0];
    let mut jdx2wtx: Vec<u32> = vec![];
    for i_elem in 0..num_elem {
        let num_node = elem2idx_offset[i_elem + 1] - elem2idx_offset[i_elem];
        match num_node {
            4 => {
                let tetcorner2wtx = &idx2vtx
                    [elem2idx_offset[i_elem] as usize..elem2idx_offset[i_elem + 1] as usize];
                let tetedge2wtx = find_edges_for_subdiv(
                    &del_geo_core::tet::EDGE2NODE,
                    i_elem,
                    elem2idx_offset,
                    idx2vtx,
                    &map_vtx2edge,
                    num_vtx0,
                );
                let tet2wtx = del_geo_core::tet::subdivide(
                    arrayref::array_ref![tetcorner2wtx, 0, 4],
                    &tetedge2wtx,
                );
                tet2wtx.iter().for_each(|a| {
                    welem2jdx_offset
                        .push((welem2jdx_offset.last().unwrap() + a.len() as u32) as u32)
                });
                jdx2wtx.extend_from_slice(tet2wtx.as_flattened());
            }
            5 => {
                let pyrmdcorner2wtx = &idx2vtx
                    [elem2idx_offset[i_elem] as usize..elem2idx_offset[i_elem + 1] as usize];
                let pyrmdedge2wtx = find_edges_for_subdiv(
                    &del_geo_core::pyramid::EDGE2NODE,
                    i_elem,
                    elem2idx_offset,
                    idx2vtx,
                    &map_vtx2edge,
                    num_vtx0,
                );
                let pyrmdquad2wtx: Vec<_> = (0..5)
                    .filter_map(|i_face| {
                        let idx0 = del_geo_core::pyramid::FACE2IDX[i_face];
                        let idx1 = del_geo_core::pyramid::FACE2IDX[i_face + 1];
                        if idx1 - idx0 != 4 {
                            return None;
                        }
                        let nofa = &del_geo_core::pyramid::IDX2NODE[idx0..idx1];
                        let mut node2vtx = [
                            idx2vtx[elem2idx_offset[i_elem] as usize + nofa[0]],
                            idx2vtx[elem2idx_offset[i_elem] as usize + nofa[1]],
                            idx2vtx[elem2idx_offset[i_elem] as usize + nofa[2]],
                            idx2vtx[elem2idx_offset[i_elem] as usize + nofa[3]],
                        ];
                        node2vtx.sort();
                        let i_quad = map_vtx2quad.get(&node2vtx).unwrap();
                        Some((num_vtx0 + num_edge0 + i_quad) as u32)
                    })
                    .collect();
                assert_eq!(pyrmdquad2wtx.len(), 1);
                let (pyramid2wtx, tet2wtx) = del_geo_core::pyramid::subdivide(
                    arrayref::array_ref![pyrmdcorner2wtx, 0, 5],
                    &pyrmdedge2wtx,
                    pyrmdquad2wtx[0],
                );
                pyramid2wtx.iter().for_each(|a| {
                    welem2jdx_offset
                        .push((welem2jdx_offset.last().unwrap() + a.len() as u32) as u32)
                });
                jdx2wtx.extend_from_slice(pyramid2wtx.as_flattened());
                tet2wtx.iter().for_each(|a| {
                    welem2jdx_offset
                        .push((welem2jdx_offset.last().unwrap() + a.len() as u32) as u32)
                });
                jdx2wtx.extend_from_slice(tet2wtx.as_flattened());
            }
            6 => {
                let prismcorner2wtx = &idx2vtx
                    [elem2idx_offset[i_elem] as usize..elem2idx_offset[i_elem + 1] as usize];
                let prismedge2wtx = find_edges_for_subdiv(
                    &del_geo_core::prism::EDGE2NODE,
                    i_elem,
                    elem2idx_offset,
                    idx2vtx,
                    &map_vtx2edge,
                    num_vtx0,
                );
                let prismquad2wtx: Vec<_> = (0..5)
                    .filter_map(|i_face| {
                        let idx0 = del_geo_core::prism::FACE2IDX[i_face];
                        let idx1 = del_geo_core::prism::FACE2IDX[i_face + 1];
                        if idx1 - idx0 != 4 {
                            return None;
                        }
                        let nofa = &del_geo_core::prism::IDX2NODE[idx0..idx1];
                        let mut node2vtx = [
                            idx2vtx[elem2idx_offset[i_elem] as usize + nofa[0]],
                            idx2vtx[elem2idx_offset[i_elem] as usize + nofa[1]],
                            idx2vtx[elem2idx_offset[i_elem] as usize + nofa[2]],
                            idx2vtx[elem2idx_offset[i_elem] as usize + nofa[3]],
                        ];
                        node2vtx.sort();
                        let i_quad = map_vtx2quad.get(&node2vtx).unwrap();
                        Some((num_vtx0 + num_edge0 + i_quad) as u32)
                    })
                    .collect();
                let prism2wtx = del_geo_core::prism::subdivide(
                    arrayref::array_ref![prismcorner2wtx, 0, 6],
                    &prismedge2wtx,
                    arrayref::array_ref![prismquad2wtx, 0, 3],
                );
                prism2wtx.iter().for_each(|a| {
                    welem2jdx_offset
                        .push((welem2jdx_offset.last().unwrap() + a.len() as u32) as u32)
                });
                jdx2wtx.extend_from_slice(prism2wtx.as_flattened());
            }
            8 => {
                let hexcorner2wtx = &idx2vtx
                    [elem2idx_offset[i_elem] as usize..elem2idx_offset[i_elem + 1] as usize];
                let hexedge2wtx = find_edges_for_subdiv(
                    &del_geo_core::hex::EDGE2NODE,
                    i_elem,
                    elem2idx_offset,
                    idx2vtx,
                    &map_vtx2edge,
                    num_vtx0,
                );
                let hexquad2wtx: Vec<_> = (0..6)
                    .map(|i_face| {
                        let idx0 = del_geo_core::hex::FACE2IDX[i_face];
                        let idx1 = del_geo_core::hex::FACE2IDX[i_face + 1];
                        let nofa = &del_geo_core::hex::IDX2NODE[idx0..idx1];
                        let mut node2vtx = [
                            idx2vtx[elem2idx_offset[i_elem] as usize + nofa[0]],
                            idx2vtx[elem2idx_offset[i_elem] as usize + nofa[1]],
                            idx2vtx[elem2idx_offset[i_elem] as usize + nofa[2]],
                            idx2vtx[elem2idx_offset[i_elem] as usize + nofa[3]],
                        ];
                        node2vtx.sort();
                        let i_quad = map_vtx2quad.get(&node2vtx).unwrap();
                        (num_vtx0 + num_edge0 + i_quad) as u32
                    })
                    .collect();
                let hexcntr2wtx =
                    map_elem2hex.get(&i_elem).unwrap() + num_vtx0 + num_edge0 + num_quad0;
                let hex2wtx = del_geo_core::hex::subdivide(
                    arrayref::array_ref![hexcorner2wtx, 0, 8],
                    &hexedge2wtx,
                    arrayref::array_ref![hexquad2wtx, 0, 6],
                    hexcntr2wtx as u32,
                );
                hex2wtx.iter().for_each(|a| {
                    welem2jdx_offset
                        .push((welem2jdx_offset.last().unwrap() + a.len() as u32) as u32)
                });
                jdx2wtx.extend_from_slice(hex2wtx.as_flattened());
            }
            _ => {}
        }
    }
    (welem2jdx_offset, jdx2wtx, wtx2xyz)
}

#[test]
pub fn test_elem2volume() {
    let (elem2idx_offset, idx2vtx, vtx2xyz) = {
        let data = crate::io_cfd_mesh_txt::read::<_, u32>("../asset/cfd_mesh.txt").unwrap();
        let num_elem = data.tet2vtx.len() / 4
            + data.pyrmd2vtx.len() / 5
            + data.prism2vtx.len() / 6
            + data.hex2vtx.len() / 8;
        let num_idx =
            data.tet2vtx.len() + data.pyrmd2vtx.len() + data.prism2vtx.len() + data.hex2vtx.len();
        let mut elem2idx_offset = vec![0u32; num_elem + 1];
        let mut idx2vtx = vec![0u32; num_idx];
        crate::mixed_mesh::to_polyhedron_mesh(
            &data.tet2vtx,
            &data.pyrmd2vtx,
            &data.prism2vtx,
            &data.hex2vtx,
            &mut elem2idx_offset,
            &mut idx2vtx,
        );
        (elem2idx_offset, idx2vtx, data.vtx2xyz)
    };
    let volume_total: f32 = {
        let mut elem2volume = vec![0f32; elem2idx_offset.len() - 1];
        crate::polyhedron_mesh::elem2volume(
            &elem2idx_offset,
            &idx2vtx,
            &vtx2xyz,
            1,
            &mut elem2volume,
        );
        assert!((elem2volume[0] - 1.0 / 12.0).abs() < 1.0e-8);
        assert!((elem2volume[1] - 1.0 / 6.0).abs() < 1.0e-5);
        assert!((elem2volume[2] - 1.0 / 2.0).abs() < 1.0e-8);
        assert!((elem2volume[3] - 1.0).abs() < 1.0e-8);
        elem2volume.iter().sum()
    };
    dbg!(volume_total);
    {
        let (mut elem2idx0_offset, mut idx2vtx0, mut vtx2xyz0) =
            (elem2idx_offset.clone(), idx2vtx.clone(), vtx2xyz.clone());
        for _itr in 0..4 {
            let (elem2idx1_offset, idx2vtx1, vtx2xyz1) =
                subdivide(&elem2idx0_offset, &idx2vtx0, &vtx2xyz0);
            (elem2idx0_offset, idx2vtx0, vtx2xyz0) =
                (elem2idx1_offset.clone(), idx2vtx1.clone(), vtx2xyz1.clone());
        }
        let mut elem2volume0 = vec![0f32; elem2idx0_offset.len() - 1];
        elem2volume(
            &elem2idx0_offset,
            &idx2vtx0,
            &vtx2xyz0,
            1,
            &mut elem2volume0,
        );
        let volume_total1: f32 = elem2volume0.iter().sum();
        dbg!(volume_total1);
        {
            let mut file = std::fs::File::create("../target/subdiv.vtk").expect("file not found.");
            crate::io_vtk::write_vtk_points(&mut file, "hoge", &vtx2xyz0, 3).unwrap();
            crate::io_vtk::write_vtk_cells_polyhedron(&mut file, &elem2idx0_offset, &idx2vtx0)
                .unwrap();
        }
    }
    {
        // let query = [1.3f32; 3];
        let query = [0.5, 0.5, 0.3];
        for i_elem in 0..elem2idx_offset.len() - 1 {
            let node2vtx =
                &idx2vtx[elem2idx_offset[i_elem] as usize..elem2idx_offset[i_elem + 1] as usize];
            let _bc = parametric_coord(&query, node2vtx, &vtx2xyz);
        }
    }
}

/// Recursive BVH traversal that updates `best_dist_sq`, `best_elem`, and `best_weights` with the
/// closest element found so far. Visits the nearer child first to prune the farther one early.
#[allow(clippy::too_many_arguments)]
fn search_elem_contains_query_using_bvh(
    query: &[f32; 3],
    bvhnodes: &[u32],
    bvhnode2aabb: &[f32],
    elem2idx_offset: &[u32],
    idx2vtx: &[u32],
    vtx2xyz: &[f32],
    i_bvhnode: usize,
    best_elem: &mut usize,
    best_param: &mut [f32; 3],
) {
    let aabb = arrayref::array_ref!(bvhnode2aabb, i_bvhnode * 6, 6);
    if !del_geo_core::aabb3::is_contain(aabb, query) {
        return;
    }
    if bvhnodes[i_bvhnode * 3 + 2] == u32::MAX {
        // leaf node
        let i_elem = bvhnodes[i_bvhnode * 3 + 1] as usize;
        let i0 = elem2idx_offset[i_elem] as usize;
        let i1 = elem2idx_offset[i_elem + 1] as usize;
        let res = parametric_coord(query, &idx2vtx[i0..i1], vtx2xyz);
        if let Some(bc) = res {
            *best_elem = i_elem;
            *best_param = bc;
        }
    } else {
        let i_child0 = bvhnodes[i_bvhnode * 3 + 1] as usize;
        let i_child1 = bvhnodes[i_bvhnode * 3 + 2] as usize;
        let d0 = del_geo_core::aabb3::min_sq_dist_to_point3(
            arrayref::array_ref!(bvhnode2aabb, i_child0 * 6, 6),
            query,
        );
        let d1 = del_geo_core::aabb3::min_sq_dist_to_point3(
            arrayref::array_ref!(bvhnode2aabb, i_child1 * 6, 6),
            query,
        );
        let (first, second) = if d0 <= d1 {
            (i_child0, i_child1)
        } else {
            (i_child1, i_child0)
        };
        search_elem_contains_query_using_bvh(
            query,
            bvhnodes,
            bvhnode2aabb,
            elem2idx_offset,
            idx2vtx,
            vtx2xyz,
            first,
            best_elem,
            best_param,
        );
        search_elem_contains_query_using_bvh(
            query,
            bvhnodes,
            bvhnode2aabb,
            elem2idx_offset,
            idx2vtx,
            vtx2xyz,
            second,
            best_elem,
            best_param,
        );
    }
}

/// For each query point in `wtx2xyz` (shape N×3, flat), find the nearest polyhedron element
/// using BVH acceleration. Returns `(wtx2elem, wtx2param)` where:
/// - `wtx2elem`: (N,) - nearest element index per query point
/// - `wtx2param`: (N×6, flat) - shape function weights within the nearest element;
///   padded with zeros for element types with fewer than 6 nodes
pub fn nearest_elem_for_points(
    bvhnodes: &[u32],
    bvhnode2aabb: &[f32],
    elem2idx_offset: &[u32],
    idx2vtx: &[u32],
    vtx2xyz: &[f32],
    wtx2xyz: &[f32],
) -> (Vec<u32>, Vec<f32>) {
    let num_wtx = wtx2xyz.len() / 3;
    use rayon::prelude::*;
    let mut wtx2elem = vec![u32::MAX; num_wtx];
    let mut wtx2param = vec![0f32; num_wtx * 3];
    wtx2elem
        .par_iter_mut()
        .zip(wtx2param.par_chunks_mut(3))
        .zip(wtx2xyz.par_chunks(3))
        .for_each(|((e, p), q)| {
            let query: &[f32; 3] = q.try_into().unwrap();
            let mut best_elem = usize::MAX;
            let mut best_weights = [0f32; 3];
            search_elem_contains_query_using_bvh(
                query,
                bvhnodes,
                bvhnode2aabb,
                elem2idx_offset,
                idx2vtx,
                vtx2xyz,
                0,
                &mut best_elem,
                &mut best_weights,
            );
            if best_elem == usize::MAX {
                *e = u32::MAX;
            } else {
                *e = best_elem as u32;
            }
            p.copy_from_slice(&best_weights);
        });
    (wtx2elem, wtx2param)
}

pub fn node_in_face(num_noel: usize, i_face: usize) -> &'static [usize] {
    match num_noel {
        4 => {
            use del_geo_core::tet::{FACE2IDX, IDX2NODE};
            &IDX2NODE[FACE2IDX[i_face]..FACE2IDX[i_face + 1]]
        }
        5 => {
            use del_geo_core::pyramid::{FACE2IDX, IDX2NODE};
            &IDX2NODE[FACE2IDX[i_face]..FACE2IDX[i_face + 1]]
        }
        6 => {
            use del_geo_core::prism::{FACE2IDX, IDX2NODE};
            &IDX2NODE[FACE2IDX[i_face]..FACE2IDX[i_face + 1]]
        }
        8 => {
            use del_geo_core::hex::{FACE2IDX, IDX2NODE};
            &IDX2NODE[FACE2IDX[i_face]..FACE2IDX[i_face + 1]]
        }
        _ => {
            unreachable!("{}", num_noel)
        }
    }
}

/// Returns the global vertex indices of face `i_face` on element `i_elem`, in local node order.
pub fn vertices_on_face<INDEX>(
    i_elem: usize,
    i_face: usize,
    elem2idx_offset: &[INDEX],
    idx2vtx: &[INDEX],
) -> Vec<INDEX>
where
    INDEX: num_traits::PrimInt + num_traits::AsPrimitive<usize>,
    usize: AsPrimitive<INDEX>,
{
    let num_noel: usize = (elem2idx_offset[i_elem + 1] - elem2idx_offset[i_elem]).as_();
    let nofa2node = node_in_face(num_noel, i_face);
    nofa2node
        .iter()
        .map(|i_node| {
            let idx: usize = elem2idx_offset[i_elem].as_() + i_node;
            idx2vtx[idx]
        })
        .collect::<Vec<_>>()
}

/// Build the face-adjacency graph for a mixed polyhedron mesh, using a precomputed
/// vertex-to-element map for efficiency. Returns `(elem2jdx_offset, jdx2elem)` in CSR form:
/// entry `jdx2elem[elem2jdx_offset[i] + f]` is the element sharing face `f` of element `i`,
/// or `usize::MAX` if the face is on the boundary.
pub fn elem2elem_with_vtx2elem<INDEX>(
    elem2idx_offset: &[INDEX],
    idx2vtx: &[INDEX],
    vtx2kdx_offset: &[INDEX],
    kdx2elem: &[INDEX],
) -> (Vec<INDEX>, Vec<INDEX>)
where
    INDEX: num_traits::PrimInt + num_traits::AsPrimitive<usize> + for<'a> std::iter::Sum<&'a INDEX>,
    usize: AsPrimitive<INDEX>,
{
    let num_elem = elem2idx_offset.len() - 1;
    let elem2jdx_offset = {
        let mut elem2jdx_offset = Vec::<INDEX>::with_capacity(num_elem + 1);
        elem2jdx_offset.push(INDEX::zero());
        for i_elem in 0..num_elem {
            let num_noel = elem2idx_offset[i_elem + 1] - elem2idx_offset[i_elem];
            let num_face: usize = match num_noel.as_() {
                4 => 4, // tet
                5 => 5, // pyramid
                6 => 5, // prism
                8 => 6, // hex
                _ => {
                    todo!()
                }
            };
            let jdx1: INDEX = *elem2jdx_offset.last().unwrap() + num_face.as_();
            elem2jdx_offset.push(jdx1);
        }
        elem2jdx_offset
    };
    let &num_jdx = elem2jdx_offset.last().unwrap();
    let num_jdx: usize = num_jdx.as_();
    let mut jdx2elem = vec![INDEX::max_value(); num_jdx];
    for i_elem in 0..num_elem {
        let num_face_i: usize = (elem2jdx_offset[i_elem + 1] - elem2jdx_offset[i_elem]).as_();
        for i_face in 0..num_face_i {
            if jdx2elem[elem2jdx_offset[i_elem].as_() + i_face] != INDEX::max_value() {
                continue;
            }
            let mut vtxs_i = vertices_on_face(i_elem, i_face, elem2idx_offset, idx2vtx);
            let sum_i: INDEX = vtxs_i.iter().sum();
            vtxs_i.sort();
            let i0_vtx: usize = vtxs_i[0].as_();
            let k0_idx: usize = vtx2kdx_offset[i0_vtx].as_();
            let k1_idx: usize = vtx2kdx_offset[i0_vtx + 1].as_();
            for &j_elem in &kdx2elem[k0_idx..k1_idx] {
                let j_elem: usize = j_elem.as_();
                if j_elem == i_elem {
                    continue;
                }
                let num_face_j: usize =
                    (elem2jdx_offset[j_elem + 1] - elem2jdx_offset[j_elem]).as_();
                for j_face in 0..num_face_j {
                    let mut vtxs_j = crate::polyhedron_mesh::vertices_on_face(
                        j_elem,
                        j_face,
                        elem2idx_offset,
                        idx2vtx,
                    );
                    if vtxs_i.len() != vtxs_j.len() {
                        continue;
                    }
                    if vtxs_j.iter().sum::<INDEX>() != sum_i {
                        continue;
                    }
                    vtxs_j.sort();
                    if vtxs_i != vtxs_j {
                        continue;
                    }
                    jdx2elem[elem2jdx_offset[i_elem].as_() + i_face] = j_elem.as_();
                    jdx2elem[elem2jdx_offset[j_elem].as_() + j_face] = i_elem.as_();
                    break;
                }
                if jdx2elem[elem2jdx_offset[i_elem].as_() + i_face] != INDEX::max_value() {
                    break;
                }
            }
        }
    }
    (elem2jdx_offset, jdx2elem)
}

/// Extract the boundary surface of a mixed polyhedron mesh as a polygon mesh in CSR form.
/// A face is on the boundary when its adjacency entry in `jdx2elem` is `usize::MAX`.
/// Returns `(oelem2ldx_offset, ldx2vtx)`.
pub fn extract_boundary(
    elem2idx_offset: &[usize],
    idx2vtx: &[usize],
    elem2jdx_offset: &[usize],
    jdx2elem: &[usize],
) -> (Vec<usize>, Vec<usize>) {
    let num_elem = elem2idx_offset.len() - 1;
    let mut oelem2ldx = vec![0usize; 1];
    let mut ldx2vtx = Vec::<usize>::new();
    for i_elem in 0..num_elem {
        let num_noel = elem2idx_offset[i_elem + 1] - elem2idx_offset[i_elem];
        let num_face = elem2jdx_offset[i_elem + 1] - elem2jdx_offset[i_elem];
        for i_face in 0..num_face {
            let j_elem = jdx2elem[elem2jdx_offset[i_elem] + i_face];
            if j_elem != usize::MAX {
                continue;
            }
            let nofa = node_in_face(num_noel, i_face);
            let num_nofa = nofa.len();
            oelem2ldx.push(oelem2ldx.last().unwrap() + num_nofa);
            nofa.iter().for_each(|i_node| {
                let i_vtx = idx2vtx[elem2idx_offset[i_elem] + i_node];
                ldx2vtx.push(i_vtx);
            });
        }
    }
    (oelem2ldx, ldx2vtx)
}

pub fn extract_quad_face<INDEX>(
    elem2idx_offset: &[INDEX],
    idx2vtx: &[INDEX],
    elem2jdx_offset: &[INDEX],
    jdx2elem: &[INDEX],
) -> Vec<INDEX>
where
    INDEX: num_traits::PrimInt + num_traits::AsPrimitive<usize>,
    usize: AsPrimitive<INDEX>,
{
    let num_elem = elem2idx_offset.len() - 1;
    let mut quad2vtx = Vec::<INDEX>::new();
    for i_elem in 0..num_elem {
        let num_noel: usize = (elem2idx_offset[i_elem + 1] - elem2idx_offset[i_elem]).as_();
        let num_face: usize = (elem2jdx_offset[i_elem + 1] - elem2jdx_offset[i_elem]).as_();
        for i_face in 0..num_face {
            let j_elem = jdx2elem[elem2jdx_offset[i_elem].as_() + i_face];
            if j_elem != INDEX::max_value() && j_elem.as_() < i_elem {
                continue;
            }
            let nofa = node_in_face(num_noel, i_face);
            if nofa.len() != 4 {
                continue;
            }
            quad2vtx.extend(
                nofa.iter()
                    .map(|i_node| idx2vtx[elem2idx_offset[i_elem].as_() + i_node]),
            );
        }
    }
    quad2vtx
}

pub fn vtx2vtx_with_vtx2elem<INDEX>(
    elem2idx: &[INDEX],
    idx2vtx: &[INDEX],
    vtx2jdx: &[INDEX],
    jdx2elem: &[INDEX],
    is_bidirectional: bool,
) -> (Vec<INDEX>, Vec<INDEX>)
where
    INDEX: num_traits::PrimInt + num_traits::AsPrimitive<usize>,
    usize: AsPrimitive<INDEX>,
{
    let nvtx = vtx2jdx.len() - 1;

    let mut vtx2kdx = vec![INDEX::zero(); nvtx + 1];
    let mut kdx2vtx = Vec::<INDEX>::new();

    for i_vtx in 0..nvtx {
        let mut set_vtx_idx = std::collections::BTreeSet::<usize>::new();
        let idx0: usize = vtx2jdx[i_vtx].as_();
        let idx1: usize = vtx2jdx[i_vtx + 1].as_();
        for &i_elem0 in &jdx2elem[idx0..idx1] {
            let i_elem0: usize = i_elem0.as_();
            let mut update = |i_node0: usize, i_node1: usize| {
                let j_vtx0: usize = idx2vtx[elem2idx[i_elem0].as_() + i_node0].as_();
                let j_vtx1: usize = idx2vtx[elem2idx[i_elem0].as_() + i_node1].as_();
                if j_vtx0 != i_vtx && j_vtx1 != i_vtx {
                    return;
                }
                if j_vtx0 == i_vtx {
                    if is_bidirectional || j_vtx1 > i_vtx {
                        set_vtx_idx.insert(j_vtx1);
                    }
                } else if is_bidirectional || j_vtx0 > i_vtx {
                    set_vtx_idx.insert(j_vtx0);
                }
            };
            let num_node: usize = (elem2idx[i_elem0 + 1] - elem2idx[i_elem0]).as_();
            match num_node {
                4 => {
                    del_geo_core::tet::EDGE2NODE
                        .iter()
                        .for_each(|&[i_node0, i_node1]| update(i_node0, i_node1));
                }
                5 => {
                    del_geo_core::pyramid::EDGE2NODE
                        .iter()
                        .for_each(|&[i_node0, i_node1]| update(i_node0, i_node1));
                }
                6 => {
                    del_geo_core::prism::EDGE2NODE
                        .iter()
                        .for_each(|&[i_node0, i_node1]| update(i_node0, i_node1));
                }
                8 => {
                    del_geo_core::hex::EDGE2NODE
                        .iter()
                        .for_each(|&[i_node0, i_node1]| update(i_node0, i_node1));
                }
                _ => unreachable!(),
            }
        }
        for &itr in &set_vtx_idx {
            kdx2vtx.push(itr.as_());
        }
        vtx2kdx[i_vtx + 1] = vtx2kdx[i_vtx] + set_vtx_idx.len().as_();
    }
    (vtx2kdx, kdx2vtx)
}

pub fn edge2vtx_with_vtx2elem<INDEX>(
    elem2idx_offset: &[INDEX],
    idx2vtx: &[INDEX],
    vtx2jdx: &[INDEX],
    jdx2elem: &[INDEX],
) -> Vec<INDEX>
where
    INDEX: num_traits::PrimInt + num_traits::AsPrimitive<usize>,
    usize: AsPrimitive<INDEX>,
{
    // Extract vertex-to-vertex connectivity from polygon edges
    let vtx2vtx = vtx2vtx_with_vtx2elem(
        elem2idx_offset,
        idx2vtx,
        vtx2jdx,
        jdx2elem,
        false, // Don't include duplicate edges
    );
    // Convert to edge list
    let mut edge2vtx = vec![INDEX::zero(); vtx2vtx.1.len() * 2];
    crate::edge2vtx::from_vtx2vtx(&vtx2vtx.0, &vtx2vtx.1, &mut edge2vtx);
    edge2vtx
}
