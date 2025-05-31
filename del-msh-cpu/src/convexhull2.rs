/**
 * This function return the array of indices of the input 2D points on the convex hull
 * in the counterclockwise orientation.
 * @tparam VEC CVec2X or Eigen::Vector2x
 * @param point_idxs
 * @param points
 */
pub fn convex_hull2<Real>(vtx2xy: &[[Real; 2]]) -> Vec<usize>
where
    Real: num_traits::Float + std::fmt::Debug,
{
    use del_geo_core::vec2::Vec2;
    let p0_idx = {
        // find the index with minimum y coordinate
        let mut p0_idx;
        p0_idx = 0;
        let mut p0 = vtx2xy[0];
        for (i, xy) in vtx2xy.iter().enumerate().skip(1) {
            if xy[1] > p0[1] {
                continue;
            }
            if xy[1] == p0[1] && xy[0] > p0[0] {
                continue;
            }
            p0_idx = i;
            p0 = *xy;
        }
        p0_idx
    };
    assert!(p0_idx < vtx2xy.len());
    // dbg!(points[p0_idx]);

    let mut idxcos: Vec<(usize, Real)> = vec![];
    {
        // compute and sort points by cosine value
        let x_axis = [Real::one(), Real::zero()];
        for i in 0..vtx2xy.len() {
            if i == p0_idx {
                continue;
            }
            let dir = vtx2xy[i].sub(&vtx2xy[p0_idx]).normalize();
            idxcos.push((i, -x_axis.dot(&dir))); // use cos for atan2
        }
    }

    {
        // sort idxcos
        idxcos.sort_by(|a, b| {
            if a.1 != b.1 {
                a.1.partial_cmp(&b.1).unwrap()
            } else {
                dbg!("hogehoge");
                let dist_a = del_geo_core::edge2::length(&vtx2xy[a.0], &vtx2xy[p0_idx]);
                let dist_b = del_geo_core::edge2::length(&vtx2xy[b.0], &vtx2xy[p0_idx]);
                dist_a.partial_cmp(&dist_b).unwrap()
            }
        });
        // dbg!(&idxcos);
    }

    // check for collinear points
    /*
        for (auto itr = ++idxcos.begin(); std::next(itr) != idxcos.end(); itr++) {
        if (std::abs((*itr).second - (*std::next(itr)).second) < EPSILON) {
            idxcos.erase(std::next(itr)); // only keep the furthest
        }
    }
    */
    // idxcos.push((p0_idx, Real::zero()));

    let mut point_idxs: Vec<usize> = vec![p0_idx, idxcos[0].0];
    let mut stack_top = 1;
    for itr in idxcos.iter().skip(1) {
        let p3_idx = itr.0;
        loop {
            assert!(stack_top > 0);
            let p1_idx = point_idxs[stack_top - 1];
            let p2_idx = point_idxs[stack_top];
            let p1p2 = vtx2xy[p2_idx].sub(&vtx2xy[p1_idx]);
            let p1p3 = vtx2xy[p3_idx].sub(&vtx2xy[p1_idx]);
            if p1p2[0] * p1p3[1] - p1p2[1] * p1p3[0] <= Real::zero() {
                // right turn or collinear
                point_idxs.pop(); // pop top of the stack
                stack_top -= 1;
            } else {
                break;
            }
        }
        point_idxs.push(p3_idx);
        stack_top += 1;
    }
    point_idxs
}

#[test]
fn test_convex_hull2() {
    use del_geo_core::vec2::Vec2;
    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaChaRng::seed_from_u64(0);
    for _iter in 0..100 {
        let vtx2xy: Vec<_> = (0..10)
            .map(|_| del_geo_core::aabb2::sample::<_, f32>(&[-2., -2., 2., 2.], &mut rng))
            .collect();
        let polyloop2vtx = convex_hull2(&vtx2xy);
        // dbg!(&polyloop2vtx);
        {
            // checks whether all angles are less than \pi
            let num_points_polygon = polyloop2vtx.len();
            for ip1 in 0..num_points_polygon {
                let ip0 = (ip1 + num_points_polygon - 1) % num_points_polygon;
                let ip2 = (ip1 + 1) % num_points_polygon;
                assert_ne!(ip0, ip1);
                assert_ne!(ip1, ip2);
                let p1p2 = vtx2xy[polyloop2vtx[ip1]].sub(&vtx2xy[polyloop2vtx[ip0]]);
                let p2p3 = vtx2xy[polyloop2vtx[ip2]].sub(&vtx2xy[polyloop2vtx[ip1]]);
                let sin_val = p1p2.cross(&p2p3);
                let v0 = p1p2.norm() * p2p3.norm();
                assert!(sin_val > 1.0e-10 * v0);
            }
        }
        {
            // checks winding number for internal points are 2 * Pi
            let boundary_point_idx =
                std::collections::BTreeSet::<usize>::from_iter(polyloop2vtx.clone());
            let polygon: Vec<[_; 2]> = polyloop2vtx.iter().map(|&i_vtx| vtx2xy[i_vtx]).collect();
            for i_vtx in 0..vtx2xy.len() {
                if boundary_point_idx.get(&i_vtx).is_some() {
                    continue;
                }
                let p = &vtx2xy[i_vtx];
                use slice_of_array::SliceFlatExt;
                let polygon = polygon.flat();
                let wn = crate::polyloop2::winding_number(polygon, p);
                assert!((wn - 1.0).abs() < 1.0e-5);
            }
        }
    }
}
