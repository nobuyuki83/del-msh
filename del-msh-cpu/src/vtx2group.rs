//! methods for making cluster

pub fn from_vtx2vtx(vtx2idx: &[usize], idx2vtx: &[usize], num_cluster: usize) -> Vec<usize> {
    let num_vtx = vtx2idx.len() - 1;
    let mut vtx2group = vec![0; num_vtx];
    let (mut vtx2dist0, _) = crate::dijkstra::vtx2dist_for_vtx2vtx(0, vtx2idx, idx2vtx, None);
    for i_cluster in 1..num_cluster {
        let (i_vtx_max_dist, &max_dist) = vtx2dist0
            .iter()
            .enumerate()
            .max_by_key(|(_i_vtx, &i_dist)| i_dist)
            .unwrap();
        assert_eq!(max_dist, vtx2dist0[i_vtx_max_dist]);
        let (vtx2dist1, _) =
            crate::dijkstra::vtx2dist_for_vtx2vtx(i_vtx_max_dist, vtx2idx, idx2vtx, Some(max_dist));
        vtx2dist0
            .iter_mut()
            .enumerate()
            .zip(vtx2dist1.iter())
            .for_each(|((i_vtx, dist0), &dist1)| {
                if dist1 < *dist0 {
                    *dist0 = dist1;
                    vtx2group[i_vtx] = i_cluster
                }
            });
    }
    vtx2group
}

#[test]
fn test_vtx2dist_for_vtx2vtx() {
    let (tri2vtx, vtx2xyz) = crate::trimesh3_primitive::sphere_yup::<usize, f64>(1.0, 64, 64);
    let (vtx2idx, idx2vtx) =
        crate::vtx2vtx::from_uniform_mesh(&tri2vtx, 3, vtx2xyz.len() / 3, false);
    let num_group = 30;
    let vtx2group = from_vtx2vtx(&vtx2idx, &idx2vtx, num_group);
    use rand::Rng;
    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaChaRng::seed_from_u64(0u64);
    let group2rgb: Vec<_> = (0..num_group * 3).map(|_| rng.random::<f32>()).collect();
    let vtx2rgb: Vec<_> = vtx2group
        .iter()
        .flat_map(|&i_group| {
            [
                group2rgb[i_group * 3],
                group2rgb[i_group * 3 + 1],
                group2rgb[i_group * 3 + 2],
            ]
        })
        .collect();
    crate::io_obj::save_tri2vtx_vtx2xyz_vtx2rgb(
        "../target/vtx2dist_from_vtx2vtx.obj",
        &tri2vtx,
        &vtx2xyz,
        &vtx2rgb,
    )
    .unwrap();
}
