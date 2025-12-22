pub trait Model {
    fn eval(&self, pos_relative: &[f32; 3], rhs: &[f32; 3]) -> [f32; 3];
}

pub struct ScreenedPoison {
    eps: f32,
    norm: f32,
    sqrt_lambda: f32,
}

impl ScreenedPoison {
    pub fn new(lambda: f32, eps: f32) -> Self {
        let sqrt_lambda = lambda.sqrt();
        let norm = eps / (-eps / sqrt_lambda).exp(); // normalization factor
        ScreenedPoison {
            eps,
            norm,
            sqrt_lambda,
        }
    }
}

impl Model for ScreenedPoison {
    fn eval(&self, r: &[f32; 3], rhs_j: &[f32; 3]) -> [f32; 3] {
        let dist_squared = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        let r_eps = (dist_squared + self.eps * self.eps).sqrt();
        let k = self.norm * (-r_eps / self.sqrt_lambda).exp() / r_eps;
        del_geo_core::vec3::scale(rhs_j, k)
    }
}

pub struct Elastic {
    eps: f32,
    norm: f32,
    a: f32,
    b: f32,
}

impl Elastic {
    fn new(nu: f32, eps: f32) -> Self {
        let a = 1. / (4. * std::f32::consts::PI);
        let b = a / (4. * (1. - nu));
        let norm = eps / (1.5 * a - b);
        Elastic { eps, norm, a, b }
    }
}

impl Model for Elastic {
    fn eval(&self, r: &[f32; 3], rhs_j: &[f32; 3]) -> [f32; 3] {
        let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        let r_eps = (r2 + self.eps * self.eps).sqrt();
        let r_eps_inv = 1. / r_eps;
        let r_eps3_inv = 1. / (r_eps * r_eps * r_eps);
        let coeff_i = self.norm
            * ((self.a - self.b) * r_eps_inv + 0.5 * self.a * self.eps * self.eps * r_eps3_inv);
        let coeff_rr_t = self.norm * self.b * r_eps3_inv;
        let dot_rg = r[0] * rhs_j[0] + r[1] * rhs_j[1] + r[2] * rhs_j[2];
        [
            coeff_i * rhs_j[0] + coeff_rr_t * dot_rg * r[0],
            coeff_i * rhs_j[1] + coeff_rr_t * dot_rg * r[1],
            coeff_i * rhs_j[2] + coeff_rr_t * dot_rg * r[2],
        ]
    }
}

pub fn screened_poisson3<M: Model>(
    spoisson: &M,
    wtx2co: &[f32],
    wtx2lhs: &mut [f32],
    vtx2co: &[f32],
    vtx2rhs: &[f32],
) {
    let num_wtx = wtx2co.len() / 3;
    let num_vtx = vtx2co.len() / 3;
    //let spoisson = ScreenedPoison::new(lambda, eps);
    for i_wtx in 0..num_wtx {
        let mut result = [0f32; 3];
        let co_i = arrayref::array_ref![wtx2co, i_wtx * 3, 3];
        for j_vtx in 0..num_vtx {
            let co_j = arrayref::array_ref![vtx2co, j_vtx * 3, 3];
            use del_geo_core::vec3;
            let xyz_diff = vec3::sub(co_i, co_j);
            let rhs_j = arrayref::array_ref![vtx2rhs, j_vtx * 3, 3];
            let res = spoisson.eval(&xyz_diff, rhs_j);
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
    let elastic = Elastic::new(nu, eps);
    for i_wtx in 0..num_wtx {
        let mut lhs_i = [0f32; 3];
        let co_i = arrayref::array_ref![wtx2co, i_wtx * 3, 3];
        for j_vtx in 0..num_vtx {
            let co_j = arrayref::array_ref![vtx2co, j_vtx * 3, 3];
            let rhs_j = arrayref::array_ref![vtx2rhs, j_vtx * 3, 3];
            let r = vec3::sub(co_i, co_j);
            let force_i = elastic.eval(&r, rhs_j);
            /*
            let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
            let r_eps = (r2 + eps * eps).sqrt();
            let r_eps_inv = 1. / r_eps;
            let r_eps3_inv = 1. / (r_eps * r_eps * r_eps);
            let coeff_i = norm * ((a - b) * r_eps_inv + 0.5 * a * eps * eps * r_eps3_inv);
            let coeff_rr_t = norm * b * r_eps3_inv;
            let dot_rg = r[0] * rhs_j[0] + r[1] * rhs_j[1] + r[2] * rhs_j[2];
             */
            lhs_i[0] += force_i[0]; //coeff_i * rhs_j[0] + coeff_rr_t * dot_rg * r[0];
            lhs_i[1] += force_i[1]; //coeff_i * rhs_j[1] + coeff_rr_t * dot_rg * r[1];
            lhs_i[2] += force_i[2]; //coeff_i * rhs_j[2] + coeff_rr_t * dot_rg * r[2];
        }
        wtx2lhs[i_wtx * 3] = lhs_i[0];
        wtx2lhs[i_wtx * 3 + 1] = lhs_i[1];
        wtx2lhs[i_wtx * 3 + 2] = lhs_i[2];
    }
}

pub struct Octree<'a> {
    pub onodes: &'a [u32],
    pub onode2center: &'a [f32],
    pub onode2depth: &'a [u32],
}

#[allow(clippy::too_many_arguments)]
pub fn barnes_hut<M: Model>(
    spoisson: &M,
    vtx2xyz: &[f32],
    vtx2rhs: &[f32],
    vtx2lhs: &mut [f32],
    transform_world2unit: &[f32; 16],
    octree: Octree,
    idx2vtx: &[u32],
    onode2gcunit: &[[f32; 3]],
    onode2rhs: &[[f32; 3]],
) {
    let linear_world2unit =
        del_geo_core::mat4_col_major::to_mat3_col_major_xyz(transform_world2unit);
    let svd = del_geo_core::mat3_col_major::svd(
        &linear_world2unit,
        del_geo_core::mat3_sym::EigenDecompositionModes::JacobiNumIter(10),
    )
    .unwrap();
    let svd_scale = svd.1;
    dbg!(svd_scale);
    let scale_unit2world = 1.0 / svd_scale[0];
    let num_vtx = vtx2xyz.len() / 3;
    assert_eq!(vtx2rhs.len(), num_vtx * 3);
    assert_eq!(vtx2lhs.len(), num_vtx * 3);
    #[allow(clippy::too_many_arguments)]
    fn get_force<M: Model>(
        spoisson: &M,
        lhs_i: &mut [f32; 3],
        pos_i_world: &[f32; 3],
        pos_i_unit: &[f32; 3],
        j_onode: usize,
        onodes: &[u32],
        onode2center: &[f32],
        onode2depth: &[u32],
        onode2gcunit: &[[f32; 3]],
        onode2rhs: &[[f32; 3]],
        theta: f32,
        scale_unit2world: f32,
        idx2vtx: &[u32],
        vtx2xyz: &[f32],
        vtx2rhs: &[f32],
    ) {
        let num_onode = onode2center.len() / 3;
        let num_vtx = idx2vtx.len();
        assert!(j_onode < num_onode);
        let center_unit = arrayref::array_ref![onode2center, j_onode * 3, 3];
        let gc_unit = onode2gcunit[j_onode];
        let celllen_unit = 1.0 / (1 << onode2depth[j_onode]) as f32;
        let dist_unit = del_geo_core::edge3::length(pos_i_unit, &gc_unit);
        let delta_unit = del_geo_core::edge3::length(center_unit, &gc_unit);
        if dist_unit - delta_unit > 0. && celllen_unit < (dist_unit - delta_unit) * theta {
            // cell is enough far
            let pos_relative_unit = del_geo_core::vec3::sub(pos_i_unit, &gc_unit);
            let pos_relative_world =
                del_geo_core::vec3::scale(&pos_relative_unit, scale_unit2world);
            let rhs_j = &onode2rhs[j_onode];
            let force_i = spoisson.eval(&pos_relative_world, rhs_j);
            lhs_i[0] += force_i[0];
            lhs_i[1] += force_i[1];
            lhs_i[2] += force_i[2];
            return;
        }
        for j_child in 0..8 {
            let j_onode_child = onodes[j_onode * 9 + 1 + j_child];
            if j_onode_child == u32::MAX {
                continue;
            }
            let j_onode_child = j_onode_child as usize;
            if j_onode_child >= num_onode {
                let jdx = j_onode_child - num_onode;
                assert!(jdx < num_vtx);
                let j_vtx = idx2vtx[jdx] as usize;
                let pos_j_world = arrayref::array_ref![vtx2xyz, j_vtx * 3, 3];
                let pos_relative_world = del_geo_core::vec3::sub(pos_i_world, pos_j_world);
                let rhs_j = arrayref::array_ref![vtx2rhs, j_vtx * 3, 3];
                let res = spoisson.eval(&pos_relative_world, rhs_j);
                lhs_i[0] += res[0];
                lhs_i[1] += res[1];
                lhs_i[2] += res[2];
            } else {
                get_force(
                    spoisson,
                    lhs_i,
                    pos_i_world,
                    pos_i_unit,
                    j_onode_child,
                    onodes,
                    onode2center,
                    onode2depth,
                    onode2gcunit,
                    onode2rhs,
                    theta,
                    scale_unit2world,
                    idx2vtx,
                    vtx2xyz,
                    vtx2rhs,
                );
            }
        }
    }
    for i_vtx in 0..num_vtx {
        let pos_world = arrayref::array_ref![vtx2xyz, i_vtx * 3, 3];
        let pos_unit =
            del_geo_core::mat4_col_major::transform_homogeneous(transform_world2unit, pos_world)
                .unwrap();
        get_force(
            spoisson,
            arrayref::array_mut_ref![vtx2lhs, i_vtx * 3, 3],
            pos_world,
            &pos_unit,
            0,
            octree.onodes,
            octree.onode2center,
            octree.onode2depth,
            onode2gcunit,
            onode2rhs,
            0.2,
            scale_unit2world,
            idx2vtx,
            vtx2xyz,
            vtx2rhs,
        );
    }
}
