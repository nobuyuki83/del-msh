struct Mesh<'a, T> {
    tri2vtx: &'a [usize],
    edge2vtx: &'a [usize],
    vtx2xyz: &'a [T],
}

#[allow(clippy::identity_op)]
fn wdw_proximity<T>(
    sum_eng: &mut T,
    grad: &mut [T],
    prox_idx: &[usize],
    prox_param: &[T],
    mesh: Mesh<T>,
    dist0: T,
    stiff: T,
) where
    T: Copy + num_traits::Float + 'static + std::fmt::Display,
    f64: num_traits::AsPrimitive<T>,
{
    use num_traits::AsPrimitive;
    let barrier = |x: T| -stiff * (dist0 - x) * (dist0 - x) * (x / dist0).ln();
    let diff_barrier = |x: T| {
        -stiff * (x - dist0) * (x - dist0) / x - stiff * 2f64.as_() * (x - dist0) * (x / dist0).ln()
    };
    use crate::vtx2xyz::to_vec3;
    use del_geo_core::vec3::Vec3;
    for (iprox, idxs) in prox_idx.chunks(3).enumerate() {
        if idxs[2] == 0 {
            // edge edge
            let ie0 = idxs[0];
            let ie1 = idxs[1];
            let ip0 = mesh.edge2vtx[ie0 * 2 + 0];
            let ip1 = mesh.edge2vtx[ie0 * 2 + 1];
            let iq0 = mesh.edge2vtx[ie1 * 2 + 0];
            let iq1 = mesh.edge2vtx[ie1 * 2 + 1];
            let p0 = to_vec3(mesh.vtx2xyz, ip0);
            let p1 = to_vec3(mesh.vtx2xyz, ip1);
            let q0 = to_vec3(mesh.vtx2xyz, iq0);
            let q1 = to_vec3(mesh.vtx2xyz, iq1);
            let pc = del_geo_core::vec3::add(
                &p0.scale(prox_param[iprox * 4 + 0]),
                &p1.scale(prox_param[iprox * 4 + 1]),
            );
            let qc = del_geo_core::vec3::add(
                &q0.scale(prox_param[iprox * 4 + 2]),
                &q1.scale(prox_param[iprox * 4 + 3]),
            );
            let dist1 = pc.sub(&qc).norm();
            println!("  proximity edge {}", dist1);
            assert!(dist1 <= dist0);
            let eng1 = barrier(dist1);
            let deng1 = diff_barrier(dist1);
            let unorm = pc.sub(&qc).normalize();
            for i in 0..3 {
                grad[ip0 * 3 + i] =
                    grad[ip0 * 3 + i] + unorm[i] * deng1 * prox_param[iprox * 4 + 0];
                grad[ip1 * 3 + i] =
                    grad[ip1 * 3 + i] + unorm[i] * deng1 * prox_param[iprox * 4 + 1];
                grad[iq0 * 3 + i] =
                    grad[iq0 * 3 + i] - unorm[i] * deng1 * prox_param[iprox * 4 + 2];
                grad[iq1 * 3 + i] =
                    grad[iq1 * 3 + i] - unorm[i] * deng1 * prox_param[iprox * 4 + 3];
            }
            *sum_eng = *sum_eng + eng1;
        } else {
            let it = idxs[0];
            let iv = idxs[1];
            let ip0 = mesh.tri2vtx[it * 3 + 0];
            let ip1 = mesh.tri2vtx[it * 3 + 1];
            let ip2 = mesh.tri2vtx[it * 3 + 2];
            let p0 = to_vec3(mesh.vtx2xyz, ip0);
            let p1 = to_vec3(mesh.vtx2xyz, ip1);
            let p2 = to_vec3(mesh.vtx2xyz, ip2);
            let q0 = to_vec3(mesh.vtx2xyz, iv);
            let pc = del_geo_core::vec3::add_three(
                &p0.scale(prox_param[iprox * 4 + 0]),
                &p1.scale(prox_param[iprox * 4 + 1]),
                &p2.scale(prox_param[iprox * 4 + 2]),
            );
            let dist1 = pc.sub(q0).norm();
            println!("  proximity face {}", dist1);
            assert!(dist1 <= dist0);
            let eng1 = barrier(dist1);
            let deng1 = diff_barrier(dist1);
            let unorm = pc.sub(q0).normalize();
            for i in 0..3 {
                grad[ip0 * 3 + i] =
                    grad[ip0 * 3 + i] + unorm[i] * deng1 * prox_param[iprox * 4 + 0];
                grad[ip1 * 3 + i] =
                    grad[ip1 * 3 + i] + unorm[i] * deng1 * prox_param[iprox * 4 + 1];
                grad[ip2 * 3 + i] =
                    grad[ip2 * 3 + i] + unorm[i] * deng1 * prox_param[iprox * 4 + 2];
                grad[iv * 3 + i] = grad[iv * 3 + i] - unorm[i] * deng1;
            }
            *sum_eng = *sum_eng + eng1;
        }
    }
}

fn wdw(
    edge2vtx: &[usize],
    tri2vtx: &[usize],
    vtx2xyz: &[f64],
    vtx2xyz_goal: &[f64],
    dist0: f64,
    k_contact: f64,
    k_diff: f64,
) -> (f64, Vec<f64>) {
    let mut sum_eng = 0f64;
    let mut res = vec![0f64; vtx2xyz.len()];
    let (prox_idx, prox_param) =
        crate::trimesh3_proximity::contacting_pair(tri2vtx, vtx2xyz, edge2vtx, dist0);
    wdw_proximity(
        &mut sum_eng,
        &mut res,
        &prox_idx,
        &prox_param,
        Mesh {
            tri2vtx,
            edge2vtx,
            vtx2xyz,
        },
        dist0,
        k_contact,
    );
    for i in 0..vtx2xyz.len() {
        let d: f64 = vtx2xyz[i] - vtx2xyz_goal[i];
        sum_eng += 0.5 * d * d * k_diff;
        res[i] += d * k_diff;
    }
    (sum_eng, res)
}

pub struct Params {
    pub k_diff: f64,
    pub k_contact: f64,
    pub dist0: f64,
    pub alpha: f64,
    pub num_iter: usize,
}

pub fn match_vtx2xyz_while_avoid_collision(
    tri2vtx: &[usize],
    vtx2xyz_start: &[f64],
    vtx2xyz_goal: &[f64],
    params: Params,
) -> Vec<f64> {
    {
        // there should be no self-intersection in the vtx2xyz_start mesh
        let tripairs = crate::trimesh3_intersection::search_brute_force(tri2vtx, vtx2xyz_start);
        for pair in tripairs.iter() {
            dbg!(pair.i_tri, pair.j_tri);
        }
        assert_eq!(tripairs.len(), 0, "there should be no intersections in start mesh but there are {:} intersecting try pairs", tripairs.len());
    }
    let edge2vtx = crate::edge2vtx::from_triangle_mesh(tri2vtx, vtx2xyz_start.len() / 3);
    assert_eq!(vtx2xyz_start.len(), vtx2xyz_goal.len());
    //
    let mut vtx2xyz = Vec::<f64>::from(vtx2xyz_start);
    for _itr_outer in 0..params.num_iter {
        // crate::io_obj::save_tri_mesh(format!("target/frame_{}.obj",_itr_outer), &tri2vtx, &vtx2xyz);
        let (w0, dw0) = wdw(
            &edge2vtx,
            tri2vtx,
            &vtx2xyz,
            vtx2xyz_goal,
            params.dist0,
            params.k_contact,
            params.k_diff,
        );
        let step: Vec<_> = dw0.iter().map(|r| r * -params.alpha).collect();
        let vtx2xyz_dist: Vec<f64> = vtx2xyz
            .iter()
            .zip(step.iter())
            .map(|(v, r)| v + r)
            .collect();
        //
        let time_max = {
            let (intersection_pair, intersection_time) =
                crate::trimesh3_intersection_time::search_brute_force(
                    &edge2vtx,
                    tri2vtx,
                    &vtx2xyz,
                    &vtx2xyz_dist,
                    f64::EPSILON,
                );
            assert_eq!(intersection_pair.len(), intersection_time.len() * 3);
            dbg!(intersection_pair.len(), intersection_time.len());
            let time_max = intersection_time
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b));
            time_max.clamp(0., 1.)
        };

        let mut coeff = 0.9;
        for _itr in 0..10 {
            let vtx2xyz_cand: Vec<_> = vtx2xyz
                .iter()
                .zip(step.iter())
                .map(|(v, r)| v + r * time_max * coeff)
                .collect();
            {
                let tripairs =
                    crate::trimesh3_intersection::search_brute_force(tri2vtx, &vtx2xyz_cand);
                println!("# of intersecting tripairs  {:}", tripairs.len());
                if !tripairs.is_empty() {
                    dbg!("something is wrong");
                    crate::io_off::save_tri_mesh("target/cand0.off", tri2vtx, &vtx2xyz);
                    crate::io_off::save_tri_mesh("target/cand1.off", tri2vtx, &vtx2xyz_cand);
                    //panic!();
                }
                // assert_eq!(tripairs.len(),0);
            }
            let (w, _) = wdw(
                &edge2vtx,
                tri2vtx,
                &vtx2xyz_cand,
                vtx2xyz_goal,
                params.dist0,
                params.k_contact,
                params.k_diff,
            );
            dbg!(w0, w);
            if w < w0 {
                vtx2xyz
                    .iter_mut()
                    .zip(vtx2xyz_cand.iter())
                    .for_each(|(v, &c)| *v = c);
                break;
            }
            coeff *= 0.5;
            println!("divide coeff {}", coeff);
        }
    }
    vtx2xyz
}

// TODO: write some test
