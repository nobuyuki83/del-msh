//! methods for 2D Kd-tree

use num_traits::AsPrimitive;

/// construct Kd-tree recursively
/// * `nodes`
/// * `idx_node`
/// * `points`
/// * `idx_point_begin`
/// * `idx_point_end`
/// * `i_depth`
#[allow(clippy::identity_op)]
pub fn construct_kdtree<Real>(
    tree: &mut Vec<usize>,
    idx_node: usize,
    points: &mut Vec<(nalgebra::Vector2::<Real>, usize)>,
    idx_point_begin: usize,
    idx_point_end: usize,
    i_depth: i32)
    where Real: nalgebra::RealField + Copy
{
    if points.is_empty() {
        tree.clear();
        return;
    }
    if idx_node == 0 {
        tree.resize(3, usize::MAX);
    }

    if idx_point_end - idx_point_begin == 1 { // leaf node
        tree[idx_node * 3 + 0] = points[idx_point_begin].1;
        return;
    }

    if i_depth % 2 == 0 { // sort by x-coordinate
        points[idx_point_begin..idx_point_end].sort_by(
            |a, b| a.0.x.partial_cmp(&b.0.x).unwrap());
    } else { // sort by y-coordinate
        points[idx_point_begin..idx_point_end].sort_by(
            |a, b| a.0.y.partial_cmp(&b.0.y).unwrap());
    }

    let idx_point_mid = (idx_point_end - idx_point_begin) / 2 + idx_point_begin; // median point
    tree[idx_node * 3 + 0] = points[idx_point_mid].1;

    if idx_point_begin != idx_point_mid { // there is at least one point smaller than median
        let idx_node_left = tree.len() / 3;
        tree.resize(tree.len() + 3, usize::MAX);
        tree[idx_node * 3 + 1] = idx_node_left;
        construct_kdtree(
            tree, idx_node_left,
            points, idx_point_begin, idx_point_mid,
            i_depth + 1);
    }
    if idx_point_mid + 1 != idx_point_end { // there is at least one point larger than median
        let idx_node_right = tree.len() / 3;
        tree.resize(tree.len() + 3, usize::MAX);
        tree[idx_node * 3 + 2] = idx_node_right;
        construct_kdtree(
            tree, idx_node_right,
            points, idx_point_mid + 1, idx_point_end,
            i_depth + 1);
    }
}

#[allow(clippy::identity_op)]
pub fn find_edges<Real>(
    edge2xy: &mut Vec<Real>,
    vtx2xy: &[Real],
    nodes: &[usize],
    idx_node: usize,
    min: nalgebra::Vector2::<Real>,
    max: nalgebra::Vector2::<Real>,
    i_depth: i32)
    where Real: Copy
{
    if idx_node >= nodes.len() { return; }
    let ivtx = nodes[idx_node * 3 + 0];
    let pos = &vtx2xy[ivtx * 2..(ivtx + 1) * 2];
    if i_depth % 2 == 0 {
        edge2xy.push(pos[0]);
        edge2xy.push(min[1]);
        edge2xy.push(pos[0]);
        edge2xy.push(max[1]);
        find_edges(edge2xy, vtx2xy, nodes, nodes[idx_node * 3 + 1],
                   min,
                   nalgebra::Vector2::new(pos[0], max[1]),
                   i_depth + 1);
        find_edges(edge2xy, vtx2xy, nodes, nodes[idx_node * 3 + 2],
                   nalgebra::Vector2::new(pos[0], min[1]),
                   max,
                   i_depth + 1);
    } else {
        edge2xy.push(min[0]);
        edge2xy.push(pos[1]);
        edge2xy.push(max[0]);
        edge2xy.push(pos[1]);
        find_edges(edge2xy, vtx2xy, nodes, nodes[idx_node * 3 + 1],
                   min,
                   nalgebra::Vector2::new(max[0], pos[1]),
                   i_depth + 1);
        find_edges(edge2xy, vtx2xy, nodes, nodes[idx_node * 3 + 2],
                   nalgebra::Vector2::new(min[0], pos[1]),
                   max,
                   i_depth + 1);
    }
}

#[allow(clippy::identity_op)]
pub fn nearest<Real>(
    pos_near: &mut (nalgebra::Vector2<Real>, usize),
    pos_in: nalgebra::Vector2<Real>,
    vtx2xy: &[Real],
    nodes: &Vec<usize>,
    idx_node: usize,
    min: nalgebra::Vector2<Real>,
    max: nalgebra::Vector2<Real>,
    i_depth: usize)
    where Real: nalgebra::RealField + Copy,
          f64: AsPrimitive<Real>
{
    if idx_node >= nodes.len() { return; } // this node does not exist

    let cur_dist = (pos_near.0 - pos_in).norm();
    if cur_dist < del_geo::aabb2::signed_distance_aabb(pos_in, min, max) { return; }

    let ivtx = nodes[idx_node * 3 + 0];
    let pos = nalgebra::Vector2::<Real>::new(vtx2xy[ivtx * 2 + 0], vtx2xy[ivtx * 2 + 1]);
    if (pos - pos_in).norm() < cur_dist {
        *pos_near = (pos, ivtx); // update the nearest position
    }

    if i_depth % 2 == 0 { // division in x direction
        if pos_in.x < pos.x {
            nearest(pos_near, pos_in, vtx2xy, nodes, nodes[idx_node * 3 + 1],
                    min, nalgebra::Vector2::<Real>::new(pos.x, max.y), i_depth + 1);
            nearest(pos_near, pos_in, vtx2xy, nodes, nodes[idx_node * 3 + 2],
                    nalgebra::Vector2::<Real>::new(pos.x, min.y), max, i_depth + 1);
        } else {
            nearest(pos_near, pos_in, vtx2xy, nodes, nodes[idx_node * 3 + 2],
                    nalgebra::Vector2::<Real>::new(pos.x, min.y), max, i_depth + 1);
            nearest(pos_near, pos_in, vtx2xy, nodes, nodes[idx_node * 3 + 1],
                    min, nalgebra::Vector2::<Real>::new(pos.x, max.y), i_depth + 1);
        }
    } else { // division in y-direction
        if pos_in.y < pos.y {
            nearest(pos_near, pos_in, vtx2xy, nodes, nodes[idx_node * 3 + 1],
                    min, nalgebra::Vector2::<Real>::new(max.x, pos.y), i_depth + 1);
            nearest(pos_near, pos_in, vtx2xy, nodes, nodes[idx_node * 3 + 2],
                    nalgebra::Vector2::<Real>::new(min.x, pos.y), max, i_depth + 1);
        } else {
            nearest(pos_near, pos_in, vtx2xy, nodes, nodes[idx_node * 3 + 2],
                    nalgebra::Vector2::<Real>::new(min.x, pos.y), max, i_depth + 1);
            nearest(pos_near, pos_in, vtx2xy, nodes, nodes[idx_node * 3 + 1],
                    min, nalgebra::Vector2::<Real>::new(max.x, pos.y), i_depth + 1);
        }
    }
}

#[allow(clippy::identity_op)]
pub fn inside_square<Real>(
    pos_near: &mut Vec<usize>,
    pos_in: nalgebra::Vector2<Real>,
    rad: Real,
    vtx2xy: &[Real],
    nodes: &Vec<usize>,
    idx_node: usize,
    min: nalgebra::Vector2<Real>,
    max: nalgebra::Vector2<Real>,
    i_depth: usize)
    where Real: nalgebra::RealField + Copy,
          f64: AsPrimitive<Real>
{
    if idx_node >= nodes.len() { return; } // this node does not exist

    if rad < del_geo::aabb2::signed_distance_aabb(pos_in, min, max) { return; }

    let ivtx = nodes[idx_node * 3 + 0];
    let pos = nalgebra::Vector2::<Real>::new(vtx2xy[ivtx * 2 + 0], vtx2xy[ivtx * 2 + 1]);
    if (pos.x - pos_in.x).abs() < rad && (pos.y - pos_in.y).abs() < rad {
        pos_near.push(ivtx); // update the nearest position
    }

    if i_depth % 2 == 0 { // division in x direction
        inside_square(pos_near, pos_in, rad, vtx2xy, nodes, nodes[idx_node * 3 + 2],
                      nalgebra::Vector2::<Real>::new(pos.x, min.y), max, i_depth + 1);
        inside_square(pos_near, pos_in, rad, vtx2xy, nodes, nodes[idx_node * 3 + 1],
                      min, nalgebra::Vector2::<Real>::new(pos.x, max.y), i_depth + 1);
    } else { // division in y-direction
        inside_square(pos_near, pos_in, rad, vtx2xy, nodes, nodes[idx_node * 3 + 1],
                      min, nalgebra::Vector2::<Real>::new(max.x, pos.y), i_depth + 1);
        inside_square(pos_near, pos_in, rad, vtx2xy, nodes, nodes[idx_node * 3 + 2],
                      nalgebra::Vector2::<Real>::new(min.x, pos.y), max, i_depth + 1);
    }
}

#[cfg(test)]
mod tests {
    use num_traits::AsPrimitive;
    use rand::distributions::Standard;
    use rand::Rng;

    fn test_data<Real>(num_xy: usize) -> (Vec<Real>, Vec<usize>)
        where Real: nalgebra::RealField + 'static + Copy,
              f64: AsPrimitive<Real>,
              Standard: rand::prelude::Distribution<Real>
    {
        let xys = {
            let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_seed([13_u8; 32]);
            let rad: Real = 0.4_f64.as_();
            let half: Real = 0.4_f64.as_();
            let mut ps = Vec::<Real>::with_capacity(num_xy * 2);
            for _i in 0..num_xy {
                let x: Real = (rng.gen::<Real>() * 2_f64.as_() - Real::one()) * rad + half;
                let y: Real = (rng.gen::<Real>() * 2_f64.as_() - Real::one()) * rad + half;
                ps.push(x);
                ps.push(y);
            }
            ps
        };
        let tree = {
            let mut pairs_xy_idx = xys.chunks(2).enumerate().map(
                |(ivtx, xy)|
                    (nalgebra::Vector2::<Real>::new(xy[0], xy[1]), ivtx))
                .collect();
            let mut tree = Vec::<usize>::new();
            crate::kdtree2::construct_kdtree(
                &mut tree, 0,
                &mut pairs_xy_idx, 0, xys.len() / 2,
                0);
            tree
        };
        (xys, tree)
    }

    #[test]
    fn check_nearest_raw() {
        use crate::kdtree2::nearest;
        use std::time;
        type Real = f64;
        type Vector = nalgebra::Vector2::<Real>;
        let (vtx2xy, nodes) = test_data::<Real>(10000);
        let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_seed([13_u8; 32]);
        let time_nearest = time::Instant::now();
        for _ in 0..10000 {
            let p0 = Vector::new(rng.gen::<Real>(), rng.gen::<Real>());
            let mut pos_near = (Vector::new(Real::MAX, Real::MAX), usize::MAX);
            nearest(&mut pos_near, p0, &vtx2xy, &nodes, 0,
                    Vector::new(0., 0.), Vector::new(1., 1.),
                    0);
        }
        dbg!(time_nearest.elapsed());
        for _ in 0..10000 {
            let p0 = Vector::new(rng.gen::<Real>(), rng.gen::<Real>());
            let mut pos_near = (Vector::new(Real::MAX, Real::MAX), usize::MAX);
            nearest(&mut pos_near, p0, &vtx2xy, &nodes, 0,
                    Vector::new(0., 0.), Vector::new(1., 1.),
                    0);
            let dist_min = (pos_near.0 - p0).norm();
            for xy in vtx2xy.chunks(2) {
                assert!((nalgebra::Vector2::<Real>::from_row_slice(xy) - p0).norm() >= dist_min);
            }
        }
    }

    #[test]
    fn check_inside_square_raw() {
        use std::time;
        use crate::kdtree2::inside_square;
        type Real = f64;
        type Vector = nalgebra::Vector2::<Real>;
        let (vtx2xy, nodes) = test_data::<Real>(10000);
        let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_seed([13_u8; 32]);
        let rad: Real = 0.03;
        let time_inside_square = time::Instant::now();
        for _ in 0..10000 {
            let p0 = Vector::new(rng.gen::<Real>(), rng.gen::<Real>());
            let mut pos_near = Vec::<usize>::new();
            inside_square(&mut pos_near, p0, rad, &vtx2xy, &nodes, 0,
                          Vector::new(0., 0.), Vector::new(1., 1.),
                          0);
        }
        dbg!(time_inside_square.elapsed());
        //
        for _ in 0..10000 {
            let p0 = Vector::new(rng.gen::<Real>(), rng.gen::<Real>());
            let mut idxs0 = Vec::<usize>::new();
            inside_square(&mut idxs0, p0, rad, &vtx2xy, &nodes, 0,
                          Vector::new(0., 0.), Vector::new(1., 1.),
                          0);
            let idxs1: Vec<usize> = vtx2xy.chunks(2).enumerate()
                .filter(|(_, xy)| (xy[0] - p0.x).abs() < rad && (xy[1] - p0.y).abs() < rad)
                .map(|v| v.0)
                .collect();
            let idxs1 = std::collections::BTreeSet::from_iter(idxs1.iter());
            let idxs0 = std::collections::BTreeSet::from_iter(idxs0.iter());
            assert_eq!(idxs1, idxs0);
        }
    }
}