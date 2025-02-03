//! methods for 2D Kd-tree

use num_traits::AsPrimitive;

// TODO: insert point in KD-tree for poisson disk sampling

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
    points: &mut Vec<([Real; 2], usize)>,
    idx_point_begin: usize,
    idx_point_end: usize,
    i_depth: i32,
) where
    Real: num_traits::Float + Copy,
{
    if points.is_empty() {
        tree.clear();
        return;
    }
    if idx_node == 0 {
        tree.resize(3, usize::MAX);
    }

    if idx_point_end - idx_point_begin == 1 {
        // leaf node
        tree[idx_node * 3 + 0] = points[idx_point_begin].1;
        return;
    }

    if i_depth % 2 == 0 {
        // sort by x-coordinate
        points[idx_point_begin..idx_point_end].sort_by(|a, b| a.0[0].partial_cmp(&b.0[0]).unwrap());
    } else {
        // sort by y-coordinate
        points[idx_point_begin..idx_point_end].sort_by(|a, b| a.0[1].partial_cmp(&b.0[1]).unwrap());
    }

    let idx_point_mid = (idx_point_end - idx_point_begin) / 2 + idx_point_begin; // median point
    tree[idx_node * 3 + 0] = points[idx_point_mid].1;

    if idx_point_begin != idx_point_mid {
        // there is at least one point smaller than median
        let idx_node_left = tree.len() / 3;
        tree.resize(tree.len() + 3, usize::MAX);
        tree[idx_node * 3 + 1] = idx_node_left;
        construct_kdtree(
            tree,
            idx_node_left,
            points,
            idx_point_begin,
            idx_point_mid,
            i_depth + 1,
        );
    }
    if idx_point_mid + 1 != idx_point_end {
        // there is at least one point larger than median
        let idx_node_right = tree.len() / 3;
        tree.resize(tree.len() + 3, usize::MAX);
        tree[idx_node * 3 + 2] = idx_node_right;
        construct_kdtree(
            tree,
            idx_node_right,
            points,
            idx_point_mid + 1,
            idx_point_end,
            i_depth + 1,
        );
    }
}

#[allow(clippy::identity_op)]
pub fn find_edges<Real>(
    edge2xy: &mut Vec<Real>,
    vtx2xy: &[Real],
    nodes: &[usize],
    idx_node: usize,
    min: [Real; 2],
    max: [Real; 2],
    i_depth: i32,
) where
    Real: Copy,
{
    if idx_node >= nodes.len() {
        return;
    }
    let ivtx = nodes[idx_node * 3 + 0];
    let pos = &vtx2xy[ivtx * 2..(ivtx + 1) * 2];
    if i_depth % 2 == 0 {
        edge2xy.push(pos[0]);
        edge2xy.push(min[1]);
        edge2xy.push(pos[0]);
        edge2xy.push(max[1]);
        find_edges(
            edge2xy,
            vtx2xy,
            nodes,
            nodes[idx_node * 3 + 1],
            min,
            [pos[0], max[1]],
            i_depth + 1,
        );
        find_edges(
            edge2xy,
            vtx2xy,
            nodes,
            nodes[idx_node * 3 + 2],
            [pos[0], min[1]],
            max,
            i_depth + 1,
        );
    } else {
        edge2xy.push(min[0]);
        edge2xy.push(pos[1]);
        edge2xy.push(max[0]);
        edge2xy.push(pos[1]);
        find_edges(
            edge2xy,
            vtx2xy,
            nodes,
            nodes[idx_node * 3 + 1],
            min,
            [max[0], pos[1]],
            i_depth + 1,
        );
        find_edges(
            edge2xy,
            vtx2xy,
            nodes,
            nodes[idx_node * 3 + 2],
            [min[0], pos[1]],
            max,
            i_depth + 1,
        );
    }
}

pub struct TreeBranch<'a, Real> {
    pub vtx2xy: &'a [Real],
    pub nodes: &'a Vec<usize>,
    pub idx_node: usize,
    pub min: [Real; 2],
    pub max: [Real; 2],
    pub i_depth: usize,
}

#[allow(clippy::identity_op)]
pub fn nearest<Real>(pos_near: &mut ([Real; 2], usize), pos_in: [Real; 2], branch: TreeBranch<Real>)
where
    Real: num_traits::Float + Copy + 'static,
    f64: AsPrimitive<Real>,
{
    use del_geo_core::vec2::Vec2;
    if branch.idx_node >= branch.nodes.len() {
        return;
    } // this node does not exist

    let cur_dist = pos_near.0.sub(&pos_in).norm();
    if cur_dist
        < del_geo_core::aabb2::sdf(
            &[branch.min[0], branch.min[1], branch.max[0], branch.max[1]],
            &pos_in,
        )
    {
        return;
    }

    let ivtx = branch.nodes[branch.idx_node * 3 + 0];
    let pos = [branch.vtx2xy[ivtx * 2], branch.vtx2xy[ivtx * 2 + 1]];
    if pos.sub(&pos_in).norm() < cur_dist {
        *pos_near = (pos, ivtx); // update the nearest position
    }

    if branch.i_depth % 2 == 0 {
        // division in x direction
        if pos_in[0] < pos[0] {
            nearest(
                pos_near,
                pos_in,
                TreeBranch {
                    vtx2xy: branch.vtx2xy,
                    nodes: branch.nodes,
                    idx_node: branch.nodes[branch.idx_node * 3 + 1],
                    min: branch.min,
                    max: [pos[0], branch.max[1]],
                    i_depth: branch.i_depth + 1,
                },
            );
            nearest(
                pos_near,
                pos_in,
                TreeBranch {
                    vtx2xy: branch.vtx2xy,
                    nodes: branch.nodes,
                    idx_node: branch.nodes[branch.idx_node * 3 + 2],
                    min: [pos[0], branch.min[1]],
                    max: branch.max,
                    i_depth: branch.i_depth + 1,
                },
            );
        } else {
            nearest(
                pos_near,
                pos_in,
                TreeBranch {
                    vtx2xy: branch.vtx2xy,
                    nodes: branch.nodes,
                    idx_node: branch.nodes[branch.idx_node * 3 + 2],
                    min: [pos[0], branch.min[1]],
                    max: branch.max,
                    i_depth: branch.i_depth + 1,
                },
            );
            nearest(
                pos_near,
                pos_in,
                TreeBranch {
                    vtx2xy: branch.vtx2xy,
                    nodes: branch.nodes,
                    idx_node: branch.nodes[branch.idx_node * 3 + 1],
                    min: branch.min,
                    max: [pos[0], branch.max[1]],
                    i_depth: branch.i_depth + 1,
                },
            );
        }
    } else {
        // division in y-direction
        if pos_in[1] < pos[1] {
            nearest(
                pos_near,
                pos_in,
                TreeBranch {
                    vtx2xy: branch.vtx2xy,
                    nodes: branch.nodes,
                    idx_node: branch.nodes[branch.idx_node * 3 + 1],
                    min: branch.min,
                    max: [branch.max[0], pos[1]],
                    i_depth: branch.i_depth + 1,
                },
            );
            nearest(
                pos_near,
                pos_in,
                TreeBranch {
                    vtx2xy: branch.vtx2xy,
                    nodes: branch.nodes,
                    idx_node: branch.nodes[branch.idx_node * 3 + 2],
                    min: [branch.min[0], pos[1]],
                    max: branch.max,
                    i_depth: branch.i_depth + 1,
                },
            );
        } else {
            nearest(
                pos_near,
                pos_in,
                TreeBranch {
                    vtx2xy: branch.vtx2xy,
                    nodes: branch.nodes,
                    idx_node: branch.nodes[branch.idx_node * 3 + 2],
                    min: [branch.min[0], pos[1]],
                    max: branch.max,
                    i_depth: branch.i_depth + 1,
                },
            );
            nearest(
                pos_near,
                pos_in,
                TreeBranch {
                    vtx2xy: branch.vtx2xy,
                    nodes: branch.nodes,
                    idx_node: branch.nodes[branch.idx_node * 3 + 1],
                    min: branch.min,
                    max: [branch.max[0], pos[1]],
                    i_depth: branch.i_depth + 1,
                },
            );
        }
    }
}

#[allow(clippy::identity_op)]
pub fn inside_square<Real>(
    pos_near: &mut Vec<usize>,
    pos_in: [Real; 2],
    rad: Real,
    branch: TreeBranch<Real>,
) where
    Real: num_traits::Float + Copy + 'static,
    f64: AsPrimitive<Real>,
{
    if branch.idx_node >= branch.nodes.len() {
        return;
    } // this node does not exist

    if !del_geo_core::aabb2::is_intersect_square(
        &[branch.min[0], branch.min[1], branch.max[0], branch.max[1]],
        &pos_in,
        rad,
    ) {
        return;
    }

    let ivtx = branch.nodes[branch.idx_node * 3 + 0];
    let pos = [branch.vtx2xy[ivtx * 2 + 0], branch.vtx2xy[ivtx * 2 + 1]];
    if (pos[0] - pos_in[0]).abs() < rad && (pos[1] - pos_in[1]).abs() < rad {
        pos_near.push(ivtx); // update the nearest position
    }

    if branch.i_depth % 2 == 0 {
        // division in x direction
        inside_square(
            pos_near,
            pos_in,
            rad,
            TreeBranch {
                vtx2xy: branch.vtx2xy,
                nodes: branch.nodes,
                idx_node: branch.nodes[branch.idx_node * 3 + 2],
                min: [pos[0], branch.min[1]],
                max: branch.max,
                i_depth: branch.i_depth + 1,
            },
        );
        inside_square(
            pos_near,
            pos_in,
            rad,
            TreeBranch {
                vtx2xy: branch.vtx2xy,
                nodes: branch.nodes,
                idx_node: branch.nodes[branch.idx_node * 3 + 1],
                min: branch.min,
                max: [pos[0], branch.max[1]],
                i_depth: branch.i_depth + 1,
            },
        );
    } else {
        // division in y-direction
        inside_square(
            pos_near,
            pos_in,
            rad,
            TreeBranch {
                vtx2xy: branch.vtx2xy,
                nodes: branch.nodes,
                idx_node: branch.nodes[branch.idx_node * 3 + 1],
                min: branch.min,
                max: [branch.max[0], pos[1]],
                i_depth: branch.i_depth + 1,
            },
        );
        inside_square(
            pos_near,
            pos_in,
            rad,
            TreeBranch {
                vtx2xy: branch.vtx2xy,
                nodes: branch.nodes,
                idx_node: branch.nodes[branch.idx_node * 3 + 2],
                min: [branch.min[0], pos[1]],
                max: branch.max,
                i_depth: branch.i_depth + 1,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use crate::kdtree2::TreeBranch;
    use num_traits::AsPrimitive;

    fn test_data<Real>(num_xy: usize) -> (Vec<Real>, Vec<usize>)
    where
        Real: num_traits::Float + 'static + Copy,
        f64: AsPrimitive<Real>,
        rand::distr::StandardUniform: rand::distr::Distribution<Real>,
    {
        let xys = {
            let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_seed([13_u8; 32]);
            let rad: Real = 0.4_f64.as_();
            let half: Real = 0.4_f64.as_();
            let mut ps = Vec::<Real>::with_capacity(num_xy * 2);
            for _i in 0..num_xy {
                use rand::Rng;
                let x: Real = (rng.random::<Real>() * 2_f64.as_() - Real::one()) * rad + half;
                let y: Real = (rng.random::<Real>() * 2_f64.as_() - Real::one()) * rad + half;
                ps.push(x);
                ps.push(y);
            }
            ps
        };
        let tree = {
            let mut pairs_xy_idx = xys
                .chunks(2)
                .enumerate()
                .map(|(ivtx, xy)| ([xy[0], xy[1]], ivtx))
                .collect();
            let mut tree = Vec::<usize>::new();
            crate::kdtree2::construct_kdtree(&mut tree, 0, &mut pairs_xy_idx, 0, xys.len() / 2, 0);
            tree
        };
        (xys, tree)
    }

    #[test]
    fn check_nearest_raw() {
        use crate::kdtree2::nearest;
        use del_geo_core::vec2::Vec2;
        // use std::time;
        type Real = f64;
        let (vtx2xy, nodes) = test_data::<Real>(10000);
        let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_seed([13_u8; 32]);
        // let time_nearest = time::Instant::now();
        for _ in 0..10000 {
            use rand::Rng;
            let p0 = [rng.random::<Real>(), rng.random::<Real>()];
            let mut pos_near = ([Real::MAX, Real::MAX], usize::MAX);
            nearest(
                &mut pos_near,
                p0,
                TreeBranch {
                    vtx2xy: &vtx2xy,
                    nodes: &nodes,
                    idx_node: 0,
                    min: [0., 0.],
                    max: [1., 1.],
                    i_depth: 0,
                },
            );
        }
        // dbg!(time_nearest.elapsed());
        for _ in 0..10000 {
            use rand::Rng;
            let p0 = [rng.random::<Real>(), rng.random::<Real>()];
            let mut pos_near = ([Real::MAX, Real::MAX], usize::MAX);
            nearest(
                &mut pos_near,
                p0,
                TreeBranch {
                    vtx2xy: &vtx2xy,
                    nodes: &nodes,
                    idx_node: 0,
                    min: [0., 0.],
                    max: [1., 1.],
                    i_depth: 0,
                },
            );
            let dist_min = pos_near.0.sub(&p0).norm();
            for xy in vtx2xy.chunks(2) {
                let xy = arrayref::array_ref![xy, 0, 2];
                assert!(xy.sub(&p0).norm() >= dist_min);
            }
        }
    }

    #[test]
    fn check_inside_square_raw() {
        // use std::time;
        type Real = f64;
        let (vtx2xy, nodes) = test_data::<Real>(10000);
        let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_seed([13_u8; 32]);
        let rad: Real = 0.03;
        // let time_inside_square = time::Instant::now();
        for _ in 0..10000 {
            use rand::Rng;
            let p0 = [rng.random::<Real>(), rng.random::<Real>()];
            let mut pos_near = Vec::<usize>::new();
            crate::kdtree2::inside_square(
                &mut pos_near,
                p0,
                rad,
                TreeBranch {
                    vtx2xy: &vtx2xy,
                    nodes: &nodes,
                    idx_node: 0,
                    min: [0., 0.],
                    max: [1., 1.],
                    i_depth: 0,
                },
            );
        }
        // dbg!(time_inside_square.elapsed());
        //
        for _ in 0..10000 {
            use rand::Rng;
            let p0 = [rng.random::<Real>(), rng.random::<Real>()];
            let mut idxs0 = Vec::<usize>::new();
            crate::kdtree2::inside_square(
                &mut idxs0,
                p0,
                rad,
                TreeBranch {
                    vtx2xy: &vtx2xy,
                    nodes: &nodes,
                    idx_node: 0,
                    min: [0., 0.],
                    max: [1., 1.],
                    i_depth: 0,
                },
            );
            let idxs1: Vec<usize> = vtx2xy
                .chunks(2)
                .enumerate()
                .filter(|(_, xy)| (xy[0] - p0[0]).abs() < rad && (xy[1] - p0[1]).abs() < rad)
                .map(|v| v.0)
                .collect();
            let idxs1 = std::collections::BTreeSet::from_iter(idxs1.iter());
            let idxs0 = std::collections::BTreeSet::from_iter(idxs0.iter());
            assert_eq!(idxs1, idxs0);
        }
    }
}
