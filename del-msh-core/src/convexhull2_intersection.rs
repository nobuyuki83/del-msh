// finds furthest point from poly in direction dir
fn find_furthest<Real>(poly: &[[Real; 2]], dir: &[Real; 2]) -> Option<[Real; 2]>
where
    Real: num_traits::Float,
{
    use del_geo_core::vec2::Vec2;
    let mut max_dist = Real::neg_infinity();
    let mut ret: Option<[Real; 2]> = None;
    for xy in poly.iter() {
        let dist = dir.dot(xy);
        if dist < max_dist {
            continue;
        }
        ret = Some(xy.to_owned());
        max_dist = dist;
    }
    ret
}

/**
 * The function returns the triangle simplex that GJK finds if there is intersection
 * @tparam VEC
 * @param vtxsA points of the point set A
 * @param vtxsB points of the point set B
 */
fn find_intersecting_simplex_for_gjk2<Real>(
    vtx2xy_a: &[[Real; 2]],
    vtx2xy_b: &[[Real; 2]],
) -> Option<Vec<[Real; 2]>>
where
    Real: num_traits::Float,
{
    use del_geo_core::vec2::Vec2;
    assert!(!vtx2xy_a.is_empty() && !vtx2xy_b.is_empty());
    // compute support on Minkowski difference
    let support = |dir: &[Real; 2]| -> Option<[Real; 2]> {
        let ndir = dir.scale(-Real::one());
        Some(find_furthest(vtx2xy_a, dir)?.sub(&find_furthest(vtx2xy_b, &ndir)?))
    };

    // add first and second points of simplex
    let mut simplex: Vec<[Real; 2]> = Vec::new();
    let init_dir = vtx2xy_b[0].sub(&vtx2xy_a[0]);
    simplex.push(support(&init_dir)?);
    let vo = [Real::zero(); 2];
    let mut vd = vo.sub(&simplex[0]);
    let mut vab;
    let mut vac;
    let mut vao;
    let mut vabperp;
    let mut vacperp;
    loop {
        let vp = support(&vd)?;
        if vp.dot(&vd) < Real::zero() {
            return None;
        } else {
            simplex.push(vp);
        }
        if simplex.len() == 2 {
            // line case
            vab = simplex[0].sub(&simplex[1]);
            vao = vo.sub(&simplex[1]);
            vd = vao.sub(&vab.scale(vab.dot(&vao) / vab.squared_norm())); // ABperp
        } else if simplex.len() == 3 {
            // triangle case
            vab = simplex[1].sub(&simplex[2]);
            vac = simplex[0].sub(&simplex[2]);
            vao = vo.sub(&simplex[2]);
            vabperp = vac
                .sub(&vab.scale(vab.dot(&vac) / vab.squared_norm()))
                .scale(-Real::one());
            vacperp = vab
                .sub(&vac.scale(vac.dot(&vab) / vac.squared_norm()))
                .scale(-Real::one());
            if vabperp.dot(&vao) > Real::zero() {
                simplex.remove(0); // remove C
                vd = vabperp;
            } else if vacperp.dot(&vao) > Real::zero() {
                simplex.remove(1); // remove B
                vd = vacperp;
            } else {
                return Some(simplex);
            }
        }
    }
}

/**
 * The functions returns true if the convex hull of a 2D point set A
 * and the convex hull of a 2D point set B intersects.
 * @tparam VEC
 * @param vtxsA points of the point set A
 * @param vtxsB points of the point set B
 * @return
 */
pub fn is_intersect_two_convexhull2s_using_gjk<Real>(
    vtx2xy_a: &[[Real; 2]],
    vtx2xy_b: &[[Real; 2]],
) -> bool
where
    Real: num_traits::Float,
{
    assert!(!vtx2xy_a.is_empty() && !vtx2xy_b.is_empty());
    find_intersecting_simplex_for_gjk2(vtx2xy_a, vtx2xy_b).is_some()
}

/**
 *AC.dot(&AB) / AC.squaredNorm()
 * @tparam VEC Eigen::Vector2, Eigen::Vector3, dfm2::CVec2
 * @param[in] vtxs points
 * @param[in] a axis of projection
 * @return range of the projection
 */
fn range_projection_points_on_axis<Real>(vtx2xy: &[[Real; 2]], a: &[Real; 2]) -> (Real, Real)
where
    Real: num_traits::Float,
{
    use del_geo_core::vec2::Vec2;
    assert_eq!(vtx2xy.len() % 2, 0);
    let mut min0 = Real::infinity();
    let mut max0 = Real::neg_infinity();
    for xy in vtx2xy.iter() {
        let d = a.dot(xy);
        min0 = if d < min0 { d } else { min0 };
        max0 = if d > max0 { d } else { max0 };
    }
    assert!(min0 <= max0);
    (min0, max0)
}

pub fn intersection_between_two_convexhull2s_using_sat<Real>(
    vtxs_a: &[[Real; 2]],
    vtxs_b: &[[Real; 2]],
) -> bool
where
    Real: num_traits::Float,
{
    // use arrayref::array_ref;
    use del_geo_core::vec2::Vec2;
    for i_vtx_b in 0..vtxs_b.len() {
        for j_vtx_b in i_vtx_b + 1..vtxs_b.len() {
            let a = vtxs_b[i_vtx_b].sub(&vtxs_b[j_vtx_b]);
            let a = del_geo_core::vec2::rotate90(&a);
            let range_a = range_projection_points_on_axis(vtxs_a, &a);
            let range_b = range_projection_points_on_axis(vtxs_b, &a);
            if range_a.1 < range_b.0 {
                return false;
            } // not intersect
            if range_b.1 < range_a.0 {
                return false;
            } // not intersect
        }
    }

    for i_vtx_a in 0..vtxs_a.len() {
        for j_vtx_b in i_vtx_a + 1..vtxs_a.len() {
            let a = vtxs_a[i_vtx_a].sub(&vtxs_a[j_vtx_b]);
            let a = del_geo_core::vec2::rotate90(&a);
            let range_a = range_projection_points_on_axis(vtxs_a, &a);
            let range_b = range_projection_points_on_axis(vtxs_b, &a);
            if range_a.1 < range_b.0 {
                return false;
            } // not intersect
            if range_b.1 < range_a.0 {
                return false;
            } // not intersect
        }
    }
    true
}

#[test]
fn test_gjk_sat2test0() {
    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
    for _itr in 0..100 {
        let vtx2xy_a: Vec<_> = (0..10)
            .map(|_| del_geo_core::aabb2::sample(&[-1f64, -1f64, 1f64, 1f64], &mut rng))
            .collect();
        let vtx2xy_b0: Vec<_> = (0..10)
            .map(|_| del_geo_core::aabb2::sample(&[-1f64, -1f64, 1f64, 1f64], &mut rng))
            .collect();
        for it in 0..30 {
            let t = it as f64 * 0.1;
            let vtx2xy_b: Vec<_> = vtx2xy_b0
                .iter()
                .map(|xy| {
                    let xy = [xy[0] + 2.0 * (3. * t).sin(), xy[1]];
                    del_geo_core::vec2::rotate(&xy, t)
                })
                .collect();
            let is_intersect_gjk = is_intersect_two_convexhull2s_using_gjk(&vtx2xy_a, &vtx2xy_b);
            let is_intersect_sat =
                intersection_between_two_convexhull2s_using_sat(&vtx2xy_a, &vtx2xy_b);
            println!("{} {}", is_intersect_sat, is_intersect_gjk);
            assert_eq!(is_intersect_gjk, is_intersect_sat);
        }
    }
}
