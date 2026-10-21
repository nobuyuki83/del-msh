pub fn eval_kelvinlet_bruteforce<T>(
    tri2vtx: &[[usize; 3]],
    vtx2xyz: &[[T; 3]],
    vtx2rhs: &[[T; 3]],
    wtx2xyz: &[[T; 3]],
    wtx2lhs: &mut [[T; 3]],
    mu: T,
    nu: T,
) where
    T: num_traits::Float + num_traits::FloatConst + Send + Sync,
{
    assert_eq!(vtx2rhs.len(), vtx2xyz.len());
    assert_eq!(wtx2lhs.len(), wtx2xyz.len());
    let eval_one = |xyz_wtx: &[T; 3]| {
        let mut acc = [T::zero(); 3];
        tri2vtx
            .iter()
            .map(|node2vtx| {
                let node2xyz = [
                    vtx2xyz[node2vtx[0]],
                    vtx2xyz[node2vtx[1]],
                    vtx2xyz[node2vtx[2]],
                ];
                let node2rhs = [
                    vtx2rhs[node2vtx[0]],
                    vtx2rhs[node2vtx[1]],
                    vtx2rhs[node2vtx[2]],
                ];
                del_geo_core::tri3_kelvinlet::eval_lhs_triangle_kelvinlet(
                    &node2xyz, &node2rhs, xyz_wtx, mu, nu,
                )
            })
            .for_each(|lhs_wtx| del_geo_core::vec3::add_in_place(&mut acc, &lhs_wtx));
        acc
    };
    use rayon::prelude::*;
    wtx2lhs
        .par_iter_mut()
        .zip(wtx2xyz.par_iter())
        .for_each(|(lhs, xyz_wtx)| {
            *lhs = eval_one(xyz_wtx);
        });
}
