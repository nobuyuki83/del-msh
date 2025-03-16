//! 2D triangle mesh with dynamic topology change for meshing, re-meshing

use num_traits::AsPrimitive;

/// make a large triangle that surrounds input 2d points
///
/// # Returns
/// * `tri2vtx` -
/// * `tri2tri` - index of adjacent triangle for each triangle edge
/// * `vtx2tri` -
pub fn make_super_triangle<T>(
    vtx2xy: &mut Vec<[T; 2]>,
    min_xy: &[T; 2],
    max_xy: &[T; 2],
) -> (Vec<usize>, Vec<usize>, Vec<usize>)
where
    T: num_traits::Float,
{
    let one = T::one();
    let two = one + one;
    let three = two + one;
    let four = two + two;
    let six = three + three;
    let half = one / two;
    // super triangle
    let mut vtx2tri = vec![usize::MAX; vtx2xy.len()];
    //
    assert_eq!(vtx2tri.len(), vtx2xy.len());
    let (max_len, center) = {
        let size_xy = [max_xy[0] - min_xy[0], max_xy[1] - min_xy[1]];
        let max_len = if size_xy[0] > size_xy[1] {
            size_xy[0]
        } else {
            size_xy[1]
        };
        (
            max_len,
            [
                (min_xy[0] + max_xy[0]) * half,
                (min_xy[1] + max_xy[1]) * half,
            ],
        )
    };
    let tri_len: T = max_len * four;
    let tmp_len: T = tri_len * three.sqrt() / six;
    let npo = vtx2xy.len();
    //
    vtx2xy.resize(npo + 3, [T::zero(); 2]);
    vtx2xy[npo] = [center[0], center[1] + two * tmp_len];
    vtx2xy[npo + 1] = [center[0] - half * tri_len, center[1] - tmp_len];
    vtx2xy[npo + 2] = [center[0] + half * tri_len, center[1] - tmp_len];
    //
    vtx2tri.resize(npo + 3, 0);
    //
    let tri2vtx = vec![npo, npo + 1, npo + 2];
    let tri2tri = vec![usize::MAX; 3];
    (tri2vtx, tri2tri, vtx2tri)
}

pub fn add_points_to_mesh<T>(
    tri2vtx: &mut Vec<usize>,
    tri2tri: &mut Vec<usize>,
    vtx2tri: &mut [usize],
    vtx2xy: &[[T; 2]],
    i_vtx: usize,
) where
    T: num_traits::Float + Copy + std::fmt::Debug + 'static + std::fmt::Display,
    f64: AsPrimitive<T>,
{
    assert_eq!(vtx2xy.len(), vtx2tri.len());
    if vtx2tri[i_vtx] != usize::MAX {
        return;
    } // already added
    let po_add = vtx2xy[i_vtx];
    for i_tri in 0..tri2vtx.len() {
        let areas = [
            del_geo_core::tri2::area(
                &po_add,
                &vtx2xy[tri2vtx[i_tri * 3 + 1]],
                &vtx2xy[tri2vtx[i_tri * 3 + 2]],
            ),
            del_geo_core::tri2::area(
                &po_add,
                &vtx2xy[tri2vtx[i_tri * 3 + 2]],
                &vtx2xy[tri2vtx[i_tri * 3]],
            ),
            del_geo_core::tri2::area(
                &po_add,
                &vtx2xy[tri2vtx[i_tri * 3]],
                &vtx2xy[tri2vtx[i_tri * 3 + 1]],
            ),
        ];
        let area_sum: T = areas[0] + areas[1] + areas[2];
        assert!(area_sum > T::zero(), "area_sum={}", area_sum);
        let (&area_min, iedge) = areas
            .iter()
            .zip(0..)
            .min_by(|a, b| a.0.partial_cmp(b.0).expect("NaN area"))
            .unwrap();
        // dbg!(area_sum, areas, area_min, iedge);
        if area_min <= -area_sum * 1.0e-10f64.as_() {
            continue;
        } // the point is out of the triangle
          //
        if area_min > area_sum * 1.0e-3f64.as_() {
            crate::trimesh_topology::insert_a_point_inside_an_element(
                i_vtx, i_tri, tri2vtx, tri2tri, vtx2tri,
            );
        } else {
            crate::trimesh_topology::insert_point_on_elem_edge(
                i_vtx, i_tri, iedge, tri2vtx, tri2tri, vtx2tri,
            );
        }
        return;
    }
    panic!();
}

pub fn should_flip<T>(
    i_tri0: usize,
    i_node0: usize,
    tri2vtx: &[usize],
    tri2tri: &[usize],
    vtx2xy: &[[T; 2]],
) -> bool
where
    T: num_traits::Float + std::fmt::Display + std::fmt::Debug,
{
    if tri2tri[i_tri0 * 3 + i_node0] >= tri2vtx.len() / 3 {
        return false;
    } // there is adjacent triangle
    let j_tri0 = tri2tri[i_tri0 * 3 + i_node0];
    let j_node0 = crate::trimesh_topology::find_adjacent_edge_index(
        arrayref::array_ref!(tri2vtx, i_tri0 * 3, 3),
        arrayref::array_ref!(tri2tri, i_tri0 * 3, 3),
        i_node0,
        tri2vtx,
    );
    assert_eq!(tri2tri[j_tri0 * 3 + j_node0], i_tri0);
    let pj0 = vtx2xy[tri2vtx[j_tri0 * 3 + j_node0]];
    let pi0 = vtx2xy[tri2vtx[i_tri0 * 3 + i_node0]];
    let pi1 = vtx2xy[tri2vtx[i_tri0 * 3 + (i_node0 + 1) % 3]];
    let pi2 = vtx2xy[tri2vtx[i_tri0 * 3 + (i_node0 + 2) % 3]];
    let a_i0_i1_i2 = del_geo_core::tri2::area(&pi0, &pi1, &pi2);
    let a_j0_i2_i1 = del_geo_core::tri2::area(&pj0, &pi2, &pi1);
    assert!(a_i0_i1_i2 > T::zero(), "{} {}", a_i0_i1_i2, a_j0_i2_i1);
    assert!(a_j0_i2_i1 > T::zero(), "{} {}", a_i0_i1_i2, a_j0_i2_i1);
    let area_diamond = a_i0_i1_i2 + a_j0_i2_i1;
    let a_i0_i1_j0 = del_geo_core::tri2::area(&pi0, &pi1, &pj0);
    let a_i0_j0_i2 = del_geo_core::tri2::area(&pi0, &pj0, &pi2);
    if a_i0_i1_j0 < area_diamond * T::epsilon() {
        return false;
    }
    if a_i0_j0_i2 < area_diamond * T::epsilon() {
        return false;
    }
    let cc = del_geo_core::tri2::circumcenter(&pi0, &pi1, &pi2);
    let rad = del_geo_core::edge2::squared_length(&cc, &pi0);
    let dist = del_geo_core::edge2::squared_length(&cc, &pj0);
    if dist >= rad {
        return false;
    }
    true
}

pub fn delaunay_around_point<T>(
    i_vtx0: usize,
    tri2vtx: &mut [usize],
    tri2tri: &mut [usize],
    vtx2tri: &mut [usize],
    vtx2xy: &[[T; 2]],
) where
    T: num_traits::Float + std::fmt::Debug + std::fmt::Display,
{
    assert_eq!(vtx2xy.len(), vtx2tri.len());
    assert!(i_vtx0 < vtx2tri.len());
    if vtx2tri[i_vtx0] == usize::MAX {
        return;
    }

    let mut i_tri0 = vtx2tri[i_vtx0];
    let mut i_node0 = crate::trimesh_topology::find_node(i_vtx0, tri2vtx, i_tri0);
    assert_eq!(i_vtx0, tri2vtx[i_tri0 * 3 + i_node0]);

    // ---------------------------
    // go counter-clock-wise
    let mut flag_is_wall = false;
    loop {
        assert_eq!(tri2vtx[i_tri0 * 3 + i_node0], i_vtx0);
        if should_flip(i_tri0, i_node0, tri2vtx, tri2tri, vtx2xy) {
            // there is adjacent triangle
            crate::trimesh_topology::flip_edge(i_tri0, i_node0, tri2vtx, tri2tri, vtx2tri); // this edge is not on the edge and should be successful
            i_node0 = 2;
            assert_eq!(tri2vtx[i_tri0 * 3 + i_node0], i_vtx0); // this is the rule from FlipEdge function
            continue; // need to check the flipped element
        }
        if !crate::trimesh_topology::move_ccw(
            &mut i_tri0,
            &mut i_node0,
            usize::MAX,
            tri2vtx,
            tri2tri,
        ) {
            flag_is_wall = true;
            break;
        }
        if i_tri0 == vtx2tri[i_vtx0] {
            break;
        }
    }
    if !flag_is_wall {
        return;
    }

    // ----------------------------
    // go clock-wise
    loop {
        assert_eq!(tri2vtx[i_tri0 * 3 + i_node0], i_vtx0);
        if should_flip(i_tri0, i_node0, tri2vtx, tri2tri, vtx2xy) {
            let j_tri0 = tri2tri[i_tri0 * 3 + i_node0];
            crate::trimesh_topology::flip_edge(i_tri0, i_node0, tri2vtx, tri2tri, vtx2tri);
            i_tri0 = j_tri0;
            i_node0 = 1;
            assert_eq!(tri2vtx[i_tri0 * 3 + i_node0], i_vtx0);
            continue;
        }
        if !crate::trimesh_topology::move_cw(
            &mut i_tri0,
            &mut i_node0,
            usize::MAX,
            tri2vtx,
            tri2tri,
        ) {
            return;
        }
    }
}

fn find_edge_point_across_edge<T>(
    ipo0: usize,
    ipo1: usize,
    tri2vtx: &[usize],
    tri2tri: &[usize],
    vtx2tri: &[usize],
    vtx2xy: &[[T; 2]],
) -> Option<(usize, usize, usize, T)>
where
    T: num_traits::Float + std::fmt::Debug,
{
    let i_tri_ini = vtx2tri[ipo0];
    let i_node_ini = crate::trimesh_topology::find_node(ipo0, tri2vtx, i_tri_ini);
    let mut i_tri_cur = i_tri_ini;
    let mut i_node_cur = i_node_ini;
    loop {
        assert_eq!(tri2vtx[i_tri_cur * 3 + i_node_cur], ipo0);
        {
            let i2_node = (i_node_cur + 1) % 3;
            let i3_node = (i_node_cur + 2) % 3;
            let area0 = del_geo_core::tri2::area(
                &vtx2xy[ipo0],
                &vtx2xy[tri2vtx[i_tri_cur * 3 + i2_node]],
                &vtx2xy[ipo1],
            );
            if area0 > -T::epsilon() {
                let area1 = del_geo_core::tri2::area(
                    &vtx2xy[ipo0],
                    &vtx2xy[ipo1],
                    &vtx2xy[tri2vtx[i_tri_cur * 3 + i3_node]],
                );
                if area1 > -T::epsilon() {
                    assert!(area0 + area1 > T::epsilon());
                    let ratio = area0 / (area0 + area1);
                    return Some((i_tri_cur, i2_node, i3_node, ratio));
                }
            }
        }
        {
            let i2_node = (i_node_cur + 1) % 3;
            let i_tri_nex = tri2tri[i_tri_cur * 3 + i2_node];
            if i_tri_nex == usize::MAX {
                break;
            }
            let j_node = crate::trimesh_topology::find_adjacent_edge_index(
                tri2vtx[i_tri_nex * 3..i_tri_nex * 3 + 3]
                    .try_into()
                    .unwrap(),
                tri2tri[i_tri_nex * 3..i_tri_nex * 3 + 3]
                    .try_into()
                    .unwrap(),
                i2_node,
                tri2vtx,
            );
            let i3_node = (j_node + 1) % 3;
            assert!(i_tri_nex < tri2vtx.len());
            assert_eq!(tri2vtx[i_tri_nex * 3 + i3_node], ipo0);
            if i_tri_nex == i_tri_ini {
                return None;
            }
            i_tri_cur = i_tri_nex;
            i_node_cur = i3_node;
        }
    }

    i_node_cur = i_node_ini;
    i_tri_cur = i_tri_ini;
    loop {
        assert_eq!(tri2vtx[i_tri_cur * 3 + i_node_cur], ipo0);
        {
            let i2_node = (i_node_cur + 1) % 3;
            let i3_node = (i_node_cur + 2) % 3;
            let area0 = del_geo_core::tri2::area(
                &vtx2xy[ipo0],
                &vtx2xy[tri2vtx[i_tri_cur * 3 + i2_node]],
                &vtx2xy[ipo1],
            );
            if area0 > -T::epsilon() {
                let area1 = del_geo_core::tri2::area(
                    &vtx2xy[ipo0],
                    &vtx2xy[ipo1],
                    &vtx2xy[tri2vtx[i_tri_cur * 3 + i3_node]],
                );
                if area1 > -T::epsilon() {
                    assert!(area0 + area1 > T::epsilon());
                    let ratio = area0 / (area0 + area1);
                    return Some((i_tri_cur, i2_node, i3_node, ratio));
                }
            }
        }
        {
            let i2_node = (i_node_cur + 2) % 3;
            let i_tri_nex = tri2tri[i_tri_cur * 3 + i2_node];
            let j_node = crate::trimesh_topology::find_adjacent_edge_index(
                &tri2vtx[i_tri_cur * 3..i_tri_cur * 3 + 3]
                    .try_into()
                    .unwrap(),
                &tri2tri[i_tri_cur * 3..i_tri_cur * 3 + 3]
                    .try_into()
                    .unwrap(),
                i2_node,
                tri2vtx,
            );
            let i3_node = (j_node + 1) % 3;
            assert_eq!(tri2vtx[i_tri_nex * 3 + i3_node], ipo0);
            if i_tri_nex == i_tri_ini {
                panic!();
            }
            i_tri_cur = i_tri_nex;
            i_node_cur = i3_node;
        }
    }
}

pub fn enforce_edge<T>(
    tri2vtx: &mut [usize],
    tri2tri: &mut [usize],
    vtx2tri: &mut [usize],
    i0_vtx: usize,
    i1_vtx: usize,
    vtx2xy: &[[T; 2]],
) where
    T: num_traits::Float + std::fmt::Debug,
{
    assert_eq!(vtx2xy.len(), vtx2tri.len());
    assert!(i0_vtx < vtx2tri.len());
    assert!(i1_vtx < vtx2tri.len());
    loop {
        if let Some((i0_tri, i0_node, i1_node)) =
            crate::trimesh_topology::find_edge_by_looking_around_point(
                i0_vtx, i1_vtx, tri2vtx, tri2tri, vtx2tri,
            )
        {
            // this edge divides outside and inside
            assert_ne!(i0_node, i1_node);
            assert!(i0_node < 3);
            assert!(i1_node < 3);
            assert_eq!(tri2vtx[i0_tri * 3 + i0_node], i0_vtx);
            assert_eq!(tri2vtx[i0_tri * 3 + i1_node], i1_vtx);
            let ied0 = 3 - i0_node - i1_node;
            {
                let itri1 = tri2tri[i0_tri * 3 + ied0];
                let ied1 = crate::trimesh_topology::find_adjacent_edge_index(
                    arrayref::array_ref![tri2vtx, i0_tri * 3, 3],
                    arrayref::array_ref![tri2tri, i0_tri * 3, 3],
                    ied0,
                    tri2vtx,
                );
                assert_eq!(tri2tri[itri1 * 3 + ied1], i0_tri);
                tri2tri[itri1 * 3 + ied1] = usize::MAX;
                tri2tri[i0_tri * 3 + ied0] = usize::MAX;
            }
            break;
        } else {
            // this edge is divided from connection outer triangle
            let Some((i0_tri, i0_node, i1_node, ratio)) =
                find_edge_point_across_edge(i0_vtx, i1_vtx, tri2vtx, tri2tri, vtx2tri, vtx2xy)
            else {
                panic!();
            };
            assert!(ratio > -T::epsilon());
            // assert!( ratio < 1_f64.as_() + 1.0e-20_f64.as_());
            assert!(
                del_geo_core::tri2::area(
                    &vtx2xy[i0_vtx],
                    &vtx2xy[tri2vtx[i0_tri * 3 + i0_node]],
                    &vtx2xy[i1_vtx]
                ) > T::epsilon()
            );
            assert!(
                del_geo_core::tri2::area(
                    &vtx2xy[i0_vtx],
                    &vtx2xy[i1_vtx],
                    &vtx2xy[tri2vtx[i0_tri * 3 + i1_node]]
                ) > T::epsilon()
            );
            if ratio < T::epsilon() {
                panic!();
            } else if ratio > T::one() - T::epsilon() {
                panic!();
            } else {
                let ied0 = 3 - i0_node - i1_node;
                assert!(tri2tri[i0_tri * 3 + ied0] < tri2vtx.len());
                let res =
                    crate::trimesh_topology::flip_edge(i0_tri, ied0, tri2vtx, tri2tri, vtx2tri);
                if !res {
                    break;
                }
            }
        }
    }
}

/// Returns (vtx2tri, vtx2xy)
pub fn delete_unreferenced_points<Real>(
    tri2vtx: &mut [usize],
    vtx2tri_tmp: &[usize],
    vtx2xy_tmp: &[[Real; 2]],
    point_idxs_to_delete: &Vec<usize>,
) -> (Vec<usize>, Vec<[Real; 2]>)
where
    Real: num_traits::Float + Copy,
{
    assert_eq!(vtx2tri_tmp.len(), vtx2xy_tmp.len());
    let (map_po_del, npo_pos) = {
        let mut map_po_del = vec![usize::MAX - 1; vtx2tri_tmp.len()];
        let mut npo_pos;
        for ipo in point_idxs_to_delete {
            map_po_del[*ipo] = usize::MAX;
        }
        npo_pos = 0;
        for po_del in map_po_del.iter_mut() {
            if *po_del == usize::MAX {
                continue;
            }
            *po_del = npo_pos;
            npo_pos += 1;
        }
        (map_po_del, npo_pos)
    };
    let mut vtx2tri = vec![0; npo_pos];
    let mut vtx2xy = vec![[Real::zero(); 2]; npo_pos];
    {
        for ipo in 0..map_po_del.len() {
            if map_po_del[ipo] == usize::MAX {
                continue;
            }
            let ipo1 = map_po_del[ipo];
            vtx2tri[ipo1] = vtx2tri_tmp[ipo];
            vtx2xy[ipo1] = vtx2xy_tmp[ipo];
        }
    }
    for (i_tri, tri) in tri2vtx.chunks_mut(3).enumerate() {
        for ipo in tri.iter_mut() {
            assert_ne!(map_po_del[*ipo], usize::MAX);
            *ipo = map_po_del[*ipo];
            vtx2tri[*ipo] = i_tri;
        }
    }
    (vtx2tri, vtx2xy)
}

///
/// Returns tri2vtx, tri2tri, vtx2tri
pub fn triangulate_single_connected_shape<Real>(
    vtx2xy: &mut Vec<[Real; 2]>,
    loop2idx: &[usize],
    idx2vtx: &[usize],
) -> (Vec<usize>, Vec<usize>, Vec<usize>)
where
    Real: num_traits::Float + std::fmt::Display + std::fmt::Debug + 'static,
    f64: AsPrimitive<Real>,
{
    let point_idx_to_delete = {
        // vertices of the super triangle are to be deleted
        let npo = vtx2xy.len();
        vec![npo, npo + 1, npo + 2]
    };
    let (mut tri2vtx, mut tri2tri, mut vtx2tri) = {
        use slice_of_array::SliceFlatExt;
        let aabb = crate::vtx2xy::aabb2(vtx2xy.flat());
        make_super_triangle(
            vtx2xy,
            aabb[0..2].try_into().unwrap(),
            aabb[2..4].try_into().unwrap(),
        )
    };
    assert_eq!(vtx2xy.len(), vtx2tri.len());
    // crate::io_obj::save_tri_mesh("target/a.obj", &tri2vtx, vtx2xy);
    for i_vtx in 0..vtx2tri.len() - 3 {
        add_points_to_mesh(&mut tri2vtx, &mut tri2tri, &mut vtx2tri, vtx2xy, i_vtx);
        delaunay_around_point(i_vtx, &mut tri2vtx, &mut tri2tri, &mut vtx2tri, vtx2xy);
    }
    // crate::io_obj::save_tri_mesh("target/b.obj", &tri2vtx, vtx2xy);
    for i_loop in 0..loop2idx.len() - 1 {
        let num_vtx_in_loop = loop2idx[i_loop + 1] - loop2idx[i_loop];
        for idx in loop2idx[i_loop]..loop2idx[i_loop + 1] {
            let i0_vtx = idx2vtx[loop2idx[i_loop] + idx % num_vtx_in_loop];
            let i1_vtx = idx2vtx[loop2idx[i_loop] + (idx + 1) % num_vtx_in_loop];
            enforce_edge(
                &mut tri2vtx,
                &mut tri2tri,
                &mut vtx2tri,
                i0_vtx,
                i1_vtx,
                vtx2xy,
            );
        }
    }
    {
        // delete triangles outside
        let Some((itri0_ker, _iedtri)) =
            crate::trimesh_topology::find_edge_by_looking_all_triangles(
                idx2vtx[0], idx2vtx[1], &tri2vtx,
            )
        else {
            panic!()
        };
        assert!(itri0_ker < tri2vtx.len() / 3);
        let mut _tri2flg = crate::trimesh_topology::flag_connected(&tri2tri, itri0_ker, 1);
        (tri2vtx, tri2tri, _tri2flg) =
            crate::trimesh_topology::delete_tri_flag(&tri2vtx, &tri2tri, &_tri2flg, 0);
    }
    assert_eq!(vtx2tri.len(), vtx2xy.len());
    // crate::io_obj::save_tri_mesh("target/c.obj", &tri2vtx, vtx2xy);
    (vtx2tri, *vtx2xy) =
        delete_unreferenced_points(&mut tri2vtx, &vtx2tri, vtx2xy, &point_idx_to_delete);
    assert_eq!(vtx2tri.len(), vtx2xy.len());
    (tri2vtx, tri2tri, vtx2tri)
}

pub fn laplacian_mesh_smoothing_around_point<T>(
    vtx2xy: &mut [[T; 2]],
    i_vtx0: usize,
    tri2vtx: &[usize],
    tri2tri: &[usize],
    vtx2tri: &[usize],
) -> bool
where
    T: num_traits::Float + 'static + Copy,
    usize: AsPrimitive<T>,
{
    use del_geo_core::vec2::Vec2;
    assert_eq!(vtx2xy.len(), vtx2tri.len());
    let mut i_tri0 = vtx2tri[i_vtx0];
    let mut i_node0 = crate::trimesh_topology::find_node(i_vtx0, tri2vtx, i_tri0);
    let pos_before = vtx2xy[i_vtx0];
    let mut pos_new = vtx2xy[i_vtx0];
    let mut num_tri_around: usize = 1;
    loop {
        // counter-clock wise
        assert!(i_tri0 < tri2vtx.len() && i_node0 < 3);
        assert_eq!(tri2vtx[i_tri0 * 3 + i_node0], i_vtx0);
        pos_new = pos_new.add(&vtx2xy[tri2vtx[i_tri0 * 3 + (i_node0 + 1) % 3]]);
        num_tri_around += 1;
        if !crate::trimesh_topology::move_ccw(
            &mut i_tri0,
            &mut i_node0,
            usize::MAX,
            tri2vtx,
            tri2tri,
        ) {
            return false;
        }
        if i_tri0 == vtx2tri[i_vtx0] {
            break;
        }
    }
    vtx2xy[i_vtx0] = pos_new.scale(T::one() / num_tri_around.as_());
    //
    let mut flipped = false;
    i_tri0 = vtx2tri[i_vtx0];
    i_node0 = crate::trimesh_topology::find_node(i_vtx0, tri2vtx, i_tri0);
    loop {
        // counter-clock wise
        let area = crate::trimesh2::area_of_a_triangle(tri2vtx, vtx2xy, i_tri0);
        if area < T::zero() {
            flipped = true;
            break;
        }
        assert!(i_tri0 < tri2vtx.len() && i_node0 < 3);
        assert_eq!(tri2vtx[i_tri0 * 3 + i_node0], i_vtx0);
        if !crate::trimesh_topology::move_ccw(
            &mut i_tri0,
            &mut i_node0,
            usize::MAX,
            tri2vtx,
            tri2tri,
        ) {
            return false;
        }
        if i_tri0 == vtx2tri[i_vtx0] {
            break;
        }
    }
    if flipped {
        vtx2xy[i_vtx0] = pos_before;
    }
    true
}

pub struct MeshForTopologicalChange<'a, T> {
    pub tri2vtx: &'a mut Vec<usize>,
    pub tri2tri: &'a mut Vec<usize>,
    pub vtx2tri: &'a mut Vec<usize>,
    pub vtx2xy: &'a mut Vec<[T; 2]>,
}

pub fn add_points_uniformly<T>(
    dm: MeshForTopologicalChange<T>,
    vtx2flag: &mut Vec<usize>,
    tri2flag: &mut Vec<usize>,
    num_vtx_fix: usize,
    nflgpnt_offset: usize,
    target_len: T,
) where
    T: Copy + 'static + num_traits::Float + std::fmt::Debug + std::fmt::Display,
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    use del_geo_core::vec2::Vec2;
    assert_eq!(dm.vtx2xy.len(), dm.vtx2tri.len());
    assert_eq!(vtx2flag.len(), dm.vtx2tri.len());
    assert_eq!(tri2flag.len(), dm.tri2vtx.len() / 3);
    let mut ratio: T = 3_f64.as_();
    loop {
        let mut nadd = 0;
        for i_tri in 0..dm.tri2vtx.len() / 3 {
            let area = crate::trimesh2::area_of_a_triangle(dm.tri2vtx, dm.vtx2xy, i_tri);
            let len2 = target_len; // len * mesh_density.edgeLengthRatio(pcnt[0], pcnt[1]); //
            if area < len2 * len2 * ratio {
                continue;
            }
            let ipo0 = dm.vtx2tri.len();
            dm.vtx2tri.resize(dm.vtx2tri.len() + 1, usize::MAX);
            dm.vtx2xy.resize(dm.vtx2xy.len() + 1, [T::zero(); 2]);
            dm.vtx2xy[ipo0] = del_geo_core::vec2::add_three(
                &dm.vtx2xy[dm.tri2vtx[i_tri * 3]],
                &dm.vtx2xy[dm.tri2vtx[i_tri * 3 + 1]],
                &dm.vtx2xy[dm.tri2vtx[i_tri * 3 + 2]],
            )
            .scale(T::one() / 3_f64.as_());
            crate::trimesh_topology::insert_a_point_inside_an_element(
                ipo0, i_tri, dm.tri2vtx, dm.tri2tri, dm.vtx2tri,
            );
            let flag_i_tri = tri2flag[i_tri];
            tri2flag.push(flag_i_tri);
            tri2flag.push(flag_i_tri);
            vtx2flag.push(flag_i_tri + nflgpnt_offset);
            delaunay_around_point(ipo0, dm.tri2vtx, dm.tri2tri, dm.vtx2tri, dm.vtx2xy);
            nadd += 1;
        }
        for i_vtx in num_vtx_fix..dm.vtx2xy.len() {
            laplacian_mesh_smoothing_around_point(
                dm.vtx2xy, i_vtx, dm.tri2vtx, dm.tri2tri, dm.vtx2tri,
            );
        }
        if nadd != 0 {
            ratio = ratio * 0.8_f64.as_();
        } else {
            ratio = ratio * 0.5_f64.as_();
        }
        if ratio < 0.65.as_() {
            break;
        }
    }

    for i_vtx in num_vtx_fix..dm.vtx2xy.len() {
        laplacian_mesh_smoothing_around_point(dm.vtx2xy, i_vtx, dm.tri2vtx, dm.tri2tri, dm.vtx2tri);
        delaunay_around_point(i_vtx, dm.tri2vtx, dm.tri2tri, dm.vtx2tri, dm.vtx2xy);
    }
}

/// generate 2D triangle mesh from 2D polyloop
/// * `edge_leength` - length of the edge of triangles
/// * `vtxl2xy` contiguous array of coordinates of the vertex of the polyloop. counter-clock wise order.
pub fn meshing_from_polyloop2<Index, Real>(
    vtxl2xy: &[Real],
    edge_length_boundary: Real,
    edge_length_internal: Real,
) -> (Vec<Index>, Vec<Real>)
where
    Real: Copy
        + 'static
        + num_traits::Float
        + AsPrimitive<usize>
        + std::fmt::Display
        + std::fmt::Debug,
    Index: Copy + 'static,
    f64: AsPrimitive<Real>,
    usize: AsPrimitive<Real> + AsPrimitive<Index>,
{
    let mut vtx2xy: Vec<[Real; 2]> = vtxl2xy.chunks(2).map(|v| [v[0], v[1]]).collect();
    let mut loop2idx = vec![0, vtx2xy.len()];
    let mut idx2vtx: Vec<usize> = (0..vtx2xy.len()).collect();
    if edge_length_boundary > Real::zero() {
        // resample edge edge
        crate::polyloop::resample_multiple_loops_remain_original_vtxs(
            &mut loop2idx,
            &mut idx2vtx,
            &mut vtx2xy,
            edge_length_boundary,
        );
    }
    let (mut tri2vtx, mut tri2tri, mut vtx2tri) =
        triangulate_single_connected_shape(&mut vtx2xy, &loop2idx, &idx2vtx);
    let mut vtx2flag = vec![0; vtx2xy.len()];
    let mut tri2flag = vec![0; tri2vtx.len() / 3];
    let num_vtx_fix = vtx2xy.len();
    add_points_uniformly(
        MeshForTopologicalChange {
            tri2vtx: &mut tri2vtx,
            tri2tri: &mut tri2tri,
            vtx2tri: &mut vtx2tri,
            vtx2xy: &mut vtx2xy,
        },
        &mut vtx2flag,
        &mut tri2flag,
        num_vtx_fix,
        0,
        edge_length_internal,
    );
    let vtx2xy: Vec<Real> = vtx2xy.into_iter().flat_map(|v| [v[0], v[1]]).collect();
    let tri2vtx: Vec<Index> = tri2vtx.iter().map(|&v| v.as_()).collect();
    (tri2vtx, vtx2xy)
}

#[test]
fn test_square() {
    let vtx2xy0 = vec![[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
    // test non-convex shape
    let vtx2xy1 = vec![
        [-1.0, -1.0],
        [1.0, -1.0],
        [1.0, 0.5],
        [0.5, 0.5],
        [0.5, 1.0],
        [-1.0, 1.0],
    ];
    // test add_point_on_edge
    let vtx2xy2 = vec![
        [-1.0, -1.0],
        [-0.5, -1.0],
        [0.0, -1.0],
        [0.5, -1.0],
        [1.0, -1.0],
        [1.0, -0.5],
        [1.0, 0.0],
        [1.0, 0.5],
        [0.5, 0.5],
        [0.5, 1.0],
        [0.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 0.0],
    ];
    // comb model
    let vtx2xy3 = vec![
        [-1.0, -1.0],
        [1.0, -1.0],
        [1.0, -0.0],
        [1.0, 1.0],
        //
        [0.5, 1.0],
        [0.5, -0.9],
        [0.4, -0.9],
        [0.4, 1.0],
        //
        [0.0, 1.0],
        [0.0, -0.9],
        [-0.1, -0.9],
        [-0.1, 1.0],
        //
        [-1.0, 1.0],
    ];
    let vtx2xy4 = vec![
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.2],
        [0.1, 0.2],
        [0.2, 0.5],
        // todo!(0.1, 0.5)
        [1.0, 0.5],
        [1.0, 1.0],
        [0.0, 1.0],
    ];
    for (i_loop, vtx2xy) in [vtx2xy0, vtx2xy1, vtx2xy2, vtx2xy3, vtx2xy4]
        .iter()
        .enumerate()
    {
        let vtx2xy: Vec<f32> = vtx2xy.iter().flat_map(|v| [v[0], v[1]]).collect();
        {
            let (tri2vtx, vtx2xy) = meshing_from_polyloop2::<usize, _>(&vtx2xy, -1., -1.);
            let res = crate::io_obj::save_tri2vtx_vtx2xyz(
                format!("../target/a{}.obj", i_loop),
                &tri2vtx,
                &vtx2xy,
                2,
            );
            assert!(res.is_ok());
        }
        {
            let (tri2vtx, vtx2xy) = meshing_from_polyloop2::<usize, _>(&vtx2xy, 0.1, 0.1);
            let res = crate::io_obj::save_tri2vtx_vtx2xyz(
                format!("../target/b{}.obj", i_loop),
                &tri2vtx,
                &vtx2xy,
                2,
            );
            assert!(res.is_ok());
        }
    }
}

#[test]
fn test_shape_with_hole() {
    {
        // test shape with a hole
        let mut vtx2xy0 = vec![
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [0.0, 0.0],
            [0.5, 0.0],
            [0.0, 0.5],
        ];
        let loop2idx = [0, 4, 7];
        let idx2vtx: Vec<usize> = (0..vtx2xy0.len()).collect();
        let (tri2vtx, _tri2tri, _vtx2tri) =
            triangulate_single_connected_shape(&mut vtx2xy0, &loop2idx, &idx2vtx);
        crate::io_obj::save_tri2vtx_vtx2vecn("../target/d.obj", &tri2vtx, &vtx2xy0).unwrap();
    }
}
