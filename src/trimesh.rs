#[allow(clippy::identity_op)]
pub fn position_from_barycentric_coordinate<Real, const N: usize>(
    tri2vtx: &[usize],
    vtx2xyz: &[Real],
    i_tri: usize,
    r0: Real,
    r1: Real) -> [Real; N]
    where Real: num_traits::Float
{
    assert!(i_tri < tri2vtx.len() / 3);
    let i0 = tri2vtx[i_tri * 3 + 0];
    let i1 = tri2vtx[i_tri * 3 + 1];
    let i2 = tri2vtx[i_tri * 3 + 2];
    let p0 = &vtx2xyz[i0 * N + 0..i0 * N + N];
    let p1 = &vtx2xyz[i1 * N + 0..i1 * N + N];
    let p2 = &vtx2xyz[i2 * N + 0..i2 * N + N];
    let r2 = Real::one() - r0 - r1;
    let mut res =  [Real::zero(); N];
    for i in 0..N {
        res[i] = r0 * p0[i] + r1 * p1[i] + r2 * p2[i];
    }
    res
}