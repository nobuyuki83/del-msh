fn main() {
    let (tri2vtx, vtx2xyz) = del_msh_cpu::io_wavefront_obj::load_tri_mesh::<_, u32, f32>(
        "asset/spot/spot_triangulated.obj",
        Some(1.0),
    )
    .unwrap();
    let num_vtx = vtx2xyz.len() / 3;
    let grad0 = {
        let mut grad0 = vec![0f32; vtx2xyz.len()];
        grad0[0] = 0.0;
        grad0[1] = 1.0;
        grad0[2] = 0.0;
        grad0
    };
    let eps: f32 = 1.0e-3;
    let lambda: f32 = 0.01;
    let sqrt_lambda = lambda.sqrt();
    let norm = eps / (-eps / sqrt_lambda).exp(); // normalization factor
    let grad1 = {
        let mut grad1 = vec![0f32; vtx2xyz.len()];
        for i_vtx in 0..num_vtx {
            let p0 = arrayref::array_ref![vtx2xyz, i_vtx * 3, 3];
            for j_vtx in 0..num_vtx {
                // if i_vtx == j_vtx { continue; }
                let p1 = arrayref::array_ref![vtx2xyz, j_vtx * 3, 3];
                let r2 = del_geo_core::edge3::squared_length(p0, p1);
                let r_eps = (r2 + eps * eps).sqrt();
                let k = norm * (-r_eps / sqrt_lambda).exp() / r_eps;
                grad1[i_vtx * 3 + 0] += k * grad0[j_vtx * 3 + 0];
                grad1[i_vtx * 3 + 1] += k * grad0[j_vtx * 3 + 1];
                grad1[i_vtx * 3 + 2] += k * grad0[j_vtx * 3 + 2];
            }
        }
        grad1
    };
    {
        let edge2vtxe = (0..num_vtx)
            .flat_map(|i| [i * 2, i * 2 + 1])
            .collect::<Vec<usize>>();
        let mut vtxe2xyz = vec![0f32; vtx2xyz.len() * 2];
        for i_vtx in 0..num_vtx {
            vtxe2xyz[i_vtx * 6 + 0] = vtx2xyz[i_vtx * 3 + 0];
            vtxe2xyz[i_vtx * 6 + 1] = vtx2xyz[i_vtx * 3 + 1];
            vtxe2xyz[i_vtx * 6 + 2] = vtx2xyz[i_vtx * 3 + 2];
            vtxe2xyz[i_vtx * 6 + 3] = vtx2xyz[i_vtx * 3 + 0] + grad1[i_vtx * 3 + 0];
            vtxe2xyz[i_vtx * 6 + 4] = vtx2xyz[i_vtx * 3 + 1] + grad1[i_vtx * 3 + 1];
            vtxe2xyz[i_vtx * 6 + 5] = vtx2xyz[i_vtx * 3 + 2] + grad1[i_vtx * 3 + 2];
        }
        del_msh_cpu::io_wavefront_obj::save_edge2vtx_vtx2xyz(
            "target/green1.obj",
            &edge2vtxe,
            &vtxe2xyz,
            3,
        )
        .unwrap();
    }
    del_msh_cpu::io_wavefront_obj::save_tri2vtx_vtx2xyz("target/green0.obj", &tri2vtx, &vtx2xyz, 3)
        .unwrap();
    //dbg!(tri2vtx);
}
