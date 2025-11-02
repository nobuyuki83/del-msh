pub fn screened_poisson3(
    wtx2co: &[f32],
    wtx2lhs: &mut [f32],
    lambda: f32,
    eps: f32,
    vtx2co: &[f32],
    vtx2rhs: &[f32],
) {
    let num_wtx = wtx2co.len() / 3;
    let num_vtx = vtx2co.len() / 3;
    let sqrt_lambda = lambda.sqrt();
    let norm = eps / (-eps / sqrt_lambda).exp(); // normalization factor
    for i_wtx in 0..num_wtx {
        let mut result = [0f32; 3];
        let co_i = arrayref::array_ref![wtx2co, i_wtx * 3, 3];
        for j_vtx in 0..num_vtx {
            let co_j = arrayref::array_ref![vtx2co, j_vtx * 3, 3];
            use del_geo_core::vec3;
            let r = vec3::sub(co_i, co_j);
            let r2 = vec3::dot(&r, &r);
            let r_eps = (r2 + eps * eps).sqrt();
            let k = norm * (-r_eps / sqrt_lambda).exp() / r_eps;
            let rhs_j = arrayref::array_ref![vtx2rhs, j_vtx * 3, 3];
            let res = vec3::scale(rhs_j, k);
            vec3::add_in_place(&mut result, &res);
        }
        wtx2lhs[i_wtx * 3] = result[0];
        wtx2lhs[i_wtx * 3 + 1] = result[1];
        wtx2lhs[i_wtx * 3 + 2] = result[2];
    }
}

pub fn elastic3(
    wtx2co: &[f32],
    wtx2lhs: &mut [f32],
    nu: f32,
    eps: f32,
    vtx2co: &[f32],
    vtx2rhs: &[f32],
) {
    use del_geo_core::vec3;
    let num_wtx = wtx2co.len() / 3;
    let num_vtx = vtx2co.len() / 3;

    let a = 1. / (4. * std::f32::consts::PI);
    let b = a / (4. * (1. - nu));
    let norm = eps / (1.5 * a - b);
    for i_wtx in 0..num_wtx {
        let mut lhs_i = [0f32; 3];
        let co_i = arrayref::array_ref![wtx2co, i_wtx * 3, 3];
        for j_vtx in 0..num_vtx {
            let co_j = arrayref::array_ref![vtx2co, j_vtx * 3, 3];
            let rhs_j = arrayref::array_ref![vtx2rhs, j_vtx * 3, 3];
            let r = vec3::sub(co_i, co_j);
            let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
            let r_eps = (r2 + eps * eps).sqrt();
            let r_eps_inv = 1. / r_eps;
            let r_eps3_inv = 1. / (r_eps * r_eps * r_eps);
            let coeff_i = norm * ((a - b) * r_eps_inv + 0.5 * a * eps * eps * r_eps3_inv);
            let coeff_rr_t = norm * b * r_eps3_inv;
            let dot_rg = r[0] * rhs_j[0] + r[1] * rhs_j[1] + r[2] * rhs_j[2];
            lhs_i[0] += coeff_i * rhs_j[0] + coeff_rr_t * dot_rg * r[0];
            lhs_i[1] += coeff_i * rhs_j[1] + coeff_rr_t * dot_rg * r[1];
            lhs_i[2] += coeff_i * rhs_j[2] + coeff_rr_t * dot_rg * r[2];
        }
        wtx2lhs[i_wtx * 3] = lhs_i[0];
        wtx2lhs[i_wtx * 3 + 1] = lhs_i[1];
        wtx2lhs[i_wtx * 3 + 2] = lhs_i[2];
    }
}
