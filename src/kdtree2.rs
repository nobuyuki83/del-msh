//! class and methods for Kd-Tree

use num_traits::AsPrimitive;

#[derive(Clone)]
pub struct Node<Real> {
    pub pos: (nalgebra::Vector2::<Real>, usize),
    pub idx_node_left: usize,
    pub idx_node_right: usize,
}

impl<Real> Node<Real>
    where Real: num_traits::Zero
{
    pub fn new() -> Self {
        Node {
            pos: (nalgebra::Vector2::<Real>::new(Real::zero(), Real::zero()), usize::MAX),
            idx_node_left: usize::MAX,
            idx_node_right: usize::MAX,
        }
    }
}

impl<Real> Default for Node<Real>
    where Real: num_traits::Zero
{
    fn default() -> Self {
        Self::new()
    }
}


/// construct Kd-tree recursively
/// * `nodes`
/// * `idx_node`
/// * `points`
/// * `idx_point_begin`
/// * `idx_point_end`
/// * `i_depth`
pub fn construct_kdtree<Real>(
    nodes: &mut Vec<Node<Real>>,
    idx_node: usize,
    points: &mut Vec<(nalgebra::Vector2::<Real>, usize)>,
    idx_point_begin: usize,
    idx_point_end: usize,
    i_depth: i32)
    where Real: nalgebra::RealField + Copy
{
    if idx_point_end - idx_point_begin == 1 { // leaf node
        nodes[idx_node].pos = points[idx_point_begin];
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
    nodes[idx_node].pos = points[idx_point_mid];

    if idx_point_begin != idx_point_mid { // there is at least one point smaller than median
        let idx_node_left = nodes.len();
        nodes.resize(nodes.len() + 1, Node::new());
        nodes[idx_node].idx_node_left = idx_node_left;
        construct_kdtree(
            nodes, idx_node_left,
            points, idx_point_begin, idx_point_mid,
            i_depth + 1);
    }
    if idx_point_mid + 1 != idx_point_end { // there is at least one point larger than median
        let idx_node_right = nodes.len();
        nodes.resize(nodes.len() + 1, Node::new());
        nodes[idx_node].idx_node_right = idx_node_right;
        construct_kdtree(
            nodes, idx_node_right,
            points, idx_point_mid + 1, idx_point_end,
            i_depth + 1);
    }
}

pub fn find_edges<Real>(
    xyz: &mut Vec<Real>,
    nodes: &Vec<Node<Real>>,
    idx_node: usize,
    min: nalgebra::Vector2::<Real>,
    max: nalgebra::Vector2::<Real>,
    i_depth: i32)
    where Real: Copy
{
    if idx_node >= nodes.len() { return; }
    let pos = &nodes[idx_node].pos.0;
    if i_depth % 2 == 0 {
        xyz.push(pos[0]);
        xyz.push(min[1]);
        xyz.push(pos[0]);
        xyz.push(max[1]);
        find_edges(xyz, nodes, nodes[idx_node].idx_node_left,
                   min,
                   nalgebra::Vector2::new(pos[0], max[1]),
                   i_depth + 1);
        find_edges(xyz, nodes, nodes[idx_node].idx_node_right,
                   nalgebra::Vector2::new(pos[0], min[1]),
                   max,
                   i_depth + 1);
    } else {
        xyz.push(min[0]);
        xyz.push(pos[1]);
        xyz.push(max[0]);
        xyz.push(pos[1]);
        find_edges(xyz, nodes, nodes[idx_node].idx_node_left,
                   min,
                   nalgebra::Vector2::new(max[0], pos[1]),
                   i_depth + 1);
        find_edges(xyz, nodes, nodes[idx_node].idx_node_right,
                   nalgebra::Vector2::new(min[0], pos[1]),
                   max,
                   i_depth + 1);
    }
}

/// signed distance from axis-aligned bounding box
/// * `pos_in` - where the signed distance is evaluated
/// * `x_min` - bounding box's x-coordinate minimum
/// * `x_max` - bounding box's x-coordinate maximum
/// * `y_min` - bounding box's y-coordinate minimum
/// * `y_max` - bounding box's y-coordinate maximum
/// * signed distance (inside is negative)
fn signed_distance_aabb<Real>(
    pos_in: nalgebra::Vector2::<Real>,
    min0: nalgebra::Vector2::<Real>,
    max0: nalgebra::Vector2::<Real>) -> Real
    where Real: nalgebra::RealField + Copy,
          f64: AsPrimitive<Real>
{
    let half = 0.5_f64.as_();
    let x_center = (max0.x + min0.x) * half;
    let y_center = (max0.y + min0.y) * half;
    let x_dist = (pos_in.x - x_center).abs() - (max0.x - min0.x) * half;
    let y_dist = (pos_in.y - y_center).abs() - (max0.y - min0.y) * half;
    x_dist.max(y_dist)
}

pub fn nearest<Real>(
    pos_near: &mut (nalgebra::Vector2<Real>, usize),
    pos_in: nalgebra::Vector2<Real>,
    nodes: &Vec<Node<Real>>,
    idx_node: usize,
    min: nalgebra::Vector2<Real>,
    max: nalgebra::Vector2<Real>,
    i_depth: usize)
    where Real: nalgebra::RealField + Copy,
          f64: AsPrimitive<Real>
{
    if idx_node >= nodes.len() { return; } // this node does not exist

    let cur_dist = (pos_near.0 - pos_in).norm();
    if cur_dist < signed_distance_aabb(pos_in, min, max) { return; }

    let pos = nodes[idx_node].pos.0;
    if (pos - pos_in).norm() < cur_dist {
        *pos_near = nodes[idx_node].pos; // update the nearest position
    }

    if i_depth % 2 == 0 { // division in x direction
        if pos_in.x < pos.x {
            nearest(pos_near, pos_in, nodes, nodes[idx_node].idx_node_left,
                    min, nalgebra::Vector2::<Real>::new(pos.x, max.y), i_depth + 1);
            nearest(pos_near, pos_in, nodes, nodes[idx_node].idx_node_right,
                    nalgebra::Vector2::<Real>::new(pos.x, min.y), max, i_depth + 1);
        } else {
            nearest(pos_near, pos_in, nodes, nodes[idx_node].idx_node_right,
                    nalgebra::Vector2::<Real>::new(pos.x, min.y), max, i_depth + 1);
            nearest(pos_near, pos_in, nodes, nodes[idx_node].idx_node_left,
                    min, nalgebra::Vector2::<Real>::new(pos.x, max.y), i_depth + 1);
        }
    } else { // division in y-direction
        if pos_in.y < pos.y {
            nearest(pos_near, pos_in, nodes, nodes[idx_node].idx_node_left,
                    min, nalgebra::Vector2::<Real>::new(max.x, pos.y), i_depth + 1);
            nearest(pos_near, pos_in, nodes, nodes[idx_node].idx_node_right,
                    nalgebra::Vector2::<Real>::new(min.x, pos.y), max, i_depth + 1);
        } else {
            nearest(pos_near, pos_in, nodes, nodes[idx_node].idx_node_right,
                    nalgebra::Vector2::<Real>::new(min.x, pos.y), max, i_depth + 1);
            nearest(pos_near, pos_in, nodes, nodes[idx_node].idx_node_left,
                    min, nalgebra::Vector2::<Real>::new(max.x, pos.y), i_depth + 1);
        }
    }
}

pub fn inside_square<Real>(
    pos_near: &mut Vec<usize>,
    pos_in: nalgebra::Vector2<Real>,
    rad: Real,
    nodes: &Vec<Node<Real>>,
    idx_node: usize,
    min: nalgebra::Vector2<Real>,
    max: nalgebra::Vector2<Real>,
    i_depth: usize)
    where Real: nalgebra::RealField + Copy,
          f64: AsPrimitive<Real>
{
    if idx_node >= nodes.len() { return; } // this node does not exist

    if rad < signed_distance_aabb(pos_in, min, max) { return; }

    let pos = nodes[idx_node].pos.0;
    if (pos.x - pos_in.x).abs() < rad && (pos.y - pos_in.y).abs() < rad {
        pos_near.push(nodes[idx_node].pos.1); // update the nearest position
    }

    if i_depth % 2 == 0 { // division in x direction
        inside_square(pos_near, pos_in, rad, nodes, nodes[idx_node].idx_node_right,
                      nalgebra::Vector2::<Real>::new(pos.x, min.y), max, i_depth + 1);
        inside_square(pos_near, pos_in, rad, nodes, nodes[idx_node].idx_node_left,
                      min, nalgebra::Vector2::<Real>::new(pos.x, max.y), i_depth + 1);
    } else { // division in y-direction
        inside_square(pos_near, pos_in, rad, nodes, nodes[idx_node].idx_node_left, min,
                      nalgebra::Vector2::<Real>::new(max.x, pos.y), i_depth + 1);
        inside_square(pos_near, pos_in, rad, nodes, nodes[idx_node].idx_node_right,
                      nalgebra::Vector2::<Real>::new(min.x, pos.y), max, i_depth + 1);
    }
}

pub struct KdTree2<Real> {
    min: nalgebra::Vector2::<Real>,
    max: nalgebra::Vector2::<Real>,
    nodes: Vec<Node<Real>>,
}

impl<Real> KdTree2<Real>
    where Real: num_traits::Float + nalgebra::RealField,
          f64: AsPrimitive<Real>
{
    pub fn from_matrix(points_: &nalgebra::Matrix2xX::<Real>) -> Self {
        let mut ps = points_
            .column_iter().enumerate()
            .map(|v| (v.1.into_owned(), v.0))
            .collect();
        let mut nodes = Vec::<Node<Real>>::new();
        nodes.resize(1, Node::new());
        construct_kdtree(&mut nodes, 0,
                         &mut ps, 0, points_.shape().1,
                         0);
        use num_traits::Float;
        let min_x = points_.column_iter().fold(Real::nan(), |m, v| Float::min(v.x, m));
        let max_x = points_.column_iter().fold(Real::nan(), |m, v| Float::max(v.x, m));
        let min_y = points_.column_iter().fold(Real::nan(), |m, v| Float::min(v.y, m));
        let max_y = points_.column_iter().fold(Real::nan(), |m, v| Float::max(v.y, m));
        KdTree2 {
            min: nalgebra::Vector2::<Real>::new(min_x, min_y),
            max: nalgebra::Vector2::<Real>::new(max_x, max_y),
            nodes,
        }
    }

    pub fn from_vec(points_: &Vec<nalgebra::Vector2::<Real>>) -> Self {
        let mut nodes = Vec::<Node<Real>>::new();
        let mut ps = points_.iter().enumerate().map(|v| (*v.1, v.0)).collect();
        nodes.resize(1, Node::new());
        construct_kdtree(&mut nodes, 0,
                         &mut ps, 0, points_.len(),
                         0);
        use num_traits::Float;
        let min_x = points_.iter().fold(Real::nan(), |m, v| Float::min(v.x, m));
        let max_x = points_.iter().fold(Real::nan(), |m, v| Float::max(v.x, m));
        let min_y = points_.iter().fold(Real::nan(), |m, v| Float::min(v.y, m));
        let max_y = points_.iter().fold(Real::nan(), |m, v| Float::max(v.y, m));
        KdTree2 {
            min: nalgebra::Vector2::<Real>::new(min_x, min_y),
            max: nalgebra::Vector2::<Real>::new(max_x, max_y),
            nodes,
        }
    }

    pub fn edges(&self) -> Vec<Real> {
        let mut xys = Vec::<Real>::new();
        find_edges(
            &mut xys,
            &self.nodes,
            0,
            self.min, self.max,
            0);
        xys.push(self.min.x);
        xys.push(self.min.y);
        xys.push(self.max.x);
        xys.push(self.min.y);
        //
        xys.push(self.max.x);
        xys.push(self.min.y);
        xys.push(self.max.x);
        xys.push(self.max.y);
        //
        xys.push(self.max.x);
        xys.push(self.max.y);
        xys.push(self.min.x);
        xys.push(self.max.y);
        //
        xys.push(self.min.x);
        xys.push(self.max.y);
        xys.push(self.min.x);
        xys.push(self.min.y);
        xys
    }

    pub fn near(&self, pos_in: nalgebra::Vector2::<Real>) -> usize {
        let vmax = <Real as nalgebra::RealField>::max_value().unwrap();
        let mut pos_near = (nalgebra::Vector2::<Real>::new(vmax, vmax), usize::MAX);
        nearest(&mut pos_near, pos_in, &self.nodes, 0,
                self.min, self.max, 0);
        pos_near.1
    }

    pub fn inside_square(&self, pos_in: nalgebra::Vector2::<Real>, rad: Real) -> Vec<usize>{
        let mut idxs0 = Vec::<usize>::new();
        inside_square(&mut idxs0, pos_in, rad, &self.nodes, 0,
                      self.min, self.max, 0);
        idxs0
    }
}

#[cfg(test)]
mod tests {
    use crate::kdtree2::{KdTree2, Node};
    use num_traits::AsPrimitive;
    use rand::distributions::Standard;
    use rand::Rng;

    fn test_data<Real>(num_xy: usize) -> (Vec<nalgebra::Vector2<Real>>, Vec<Node<Real>>)
        where Real: nalgebra::RealField + 'static + Copy,
              f64: AsPrimitive<Real>,
              Standard: rand::prelude::Distribution<Real>
    {
        let xys = {
            let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_seed([13_u8; 32]);
            let rad: Real = 0.4_f64.as_();
            let half: Real = 0.4_f64.as_();
            let mut ps = Vec::<nalgebra::Vector2::<Real>>::new();
            for _i in 0..num_xy {
                let x: Real = (rng.gen::<Real>() * 2_f64.as_() - Real::one()) * rad + half;
                let y: Real = (rng.gen::<Real>() * 2_f64.as_() - Real::one()) * rad + half;
                ps.push(nalgebra::Vector2::<Real>::new(x, y));
            }
            ps
        };
        let nodes = {
            let mut nodes = Vec::<crate::kdtree2::Node<Real>>::new();
            let mut ps = xys.iter().enumerate().map(|v| (*v.1, v.0)).collect();
            nodes.resize(1, crate::kdtree2::Node::new());
            crate::kdtree2::construct_kdtree(
                &mut nodes, 0,
                &mut ps, 0, xys.len(),
                0);
            nodes
        };
        (xys, nodes)
    }

    #[test]
    fn check_nearest_raw() {
        use crate::kdtree2::nearest;
        use std::time;
        type Real = f64;
        type Vector = nalgebra::Vector2::<Real>;
        let (xys, nodes) = test_data(10000);
        let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_seed([13_u8; 32]);
        let time_nearest = time::Instant::now();
        for _ in 0..10000 {
            let p0 = Vector::new(rng.gen::<Real>(), rng.gen::<Real>());
            let mut pos_near = (Vector::new(Real::MAX, Real::MAX), usize::MAX);
            nearest(&mut pos_near, p0, &nodes, 0,
                    Vector::new(0., 0.), Vector::new(1., 1.),
                    0);
        }
        dbg!(time_nearest.elapsed());
        for _ in 0..10000 {
            let p0 = Vector::new(rng.gen::<Real>(), rng.gen::<Real>());
            let mut pos_near = (Vector::new(Real::MAX, Real::MAX), usize::MAX);
            nearest(&mut pos_near, p0, &nodes, 0,
                    Vector::new(0., 0.), Vector::new(1., 1.),
                    0);
            let dist_min = (xys[pos_near.1] - p0).norm();
            for i in 0..xys.len() {
                assert!((xys[i] - p0).norm() >= dist_min);
            }
        }
    }

    #[test]
    fn check_inside_square_raw() {
        use std::time;
        use crate::kdtree2::inside_square;
        type Real = f64;
        type Vector = nalgebra::Vector2::<Real>;
        let (xys, nodes) = test_data(10000);
        let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_seed([13_u8; 32]);
        let rad: Real = 0.03;
        let time_inside_square = time::Instant::now();
        for _ in 0..10000 {
            let p0 = Vector::new(rng.gen::<Real>(), rng.gen::<Real>());
            let mut pos_near = Vec::<usize>::new();
            inside_square(&mut pos_near, p0, rad, &nodes, 0,
                          Vector::new(0., 0.), Vector::new(1., 1.),
                          0);
        }
        dbg!(time_inside_square.elapsed());
        //
        for _ in 0..10000 {
            let p0 = Vector::new(rng.gen::<Real>(), rng.gen::<Real>());
            let mut idxs0 = Vec::<usize>::new();
            inside_square(&mut idxs0, p0, rad, &nodes, 0,
                          Vector::new(0., 0.), Vector::new(1., 1.),
                          0);
            let idxs1: Vec<usize> = xys.iter().enumerate()
                .filter(|&v| (v.1.x - p0.x).abs() < rad && (v.1.y - p0.y).abs() < rad)
                .map(|v| v.0)
                .collect();
            let idxs1 = std::collections::BTreeSet::from_iter(idxs1.iter());
            let idxs0 = std::collections::BTreeSet::from_iter(idxs0.iter());
            assert_eq!(idxs1, idxs0);
        }
    }

    #[test]
    fn check_nearest_matrix() {
        type Real = f64;
        type Vector = nalgebra::Vector2::<Real>;
        let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_seed([13_u8; 32]);
        let xys = {
            let mut xys = nalgebra::Matrix2xX::<Real>::zeros(10000);
            xys.iter_mut().for_each(|v| *v = rng.gen::<Real>());
            xys
        };
        let kdtree2 = KdTree2::from_matrix(&xys);
        for _ in 0..10000 {
            let p0 = Vector::new(rng.gen::<Real>(), rng.gen::<Real>());
            let idx = kdtree2.near(p0);
            let dist_min = (xys.column(idx) - p0).norm();
            for col in xys.column_iter() {
                assert!((col - p0).norm() >= dist_min);
            }
        }
    }

    #[test]
    fn check_inside_square_matrix() {
        type Real = f64;
        type Vector = nalgebra::Vector2::<Real>;
        let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_seed([13_u8; 32]);
        let xys = {
            let mut xys = nalgebra::Matrix2xX::<Real>::zeros(10000);
            xys.iter_mut().for_each(|v| *v = rng.gen::<Real>());
            xys
        };
        let kdtree2 = KdTree2::from_matrix(&xys);
        let rad = 0.01;
        for _ in 0..10000 {
            let p0 = Vector::new(rng.gen::<Real>(), rng.gen::<Real>());
            let idxs0 = kdtree2.inside_square(p0,rad);
            let idxs1: Vec<usize> = xys.column_iter().enumerate()
                .filter(|&v| (v.1.x - p0.x).abs() < rad && (v.1.y - p0.y).abs() < rad)
                .map(|v| v.0)
                .collect();
            let idxs1 = std::collections::BTreeSet::from_iter(idxs1.iter());
            let idxs0 = std::collections::BTreeSet::from_iter(idxs0.iter());
            assert_eq!(idxs1, idxs0);
        }
    }
}