//! method for quad mesh

use num_traits::AsPrimitive;

/// Generate a regular grid quad mesh with nx×ny quads at integer coordinates.
/// Returns (quad2vtx, vtx2xy) with counter-clockwise vertex ordering.
pub fn from_grid<Real>(nx: usize, ny: usize) -> (Vec<[usize; 4]>, Vec<[Real; 2]>)
where
    Real: num_traits::Float + 'static + Default,
    f32: AsPrimitive<Real>,
    usize: AsPrimitive<Real>,
{
    let np = (nx + 1) * (ny + 1);
    let mut vtx2xy: Vec<[Real; 2]> = vec![[Default::default(); 2]; np];
    for iy in 0..ny + 1 {
        #[allow(clippy::identity_op)]
        for ix in 0..nx + 1 {
            let ip = iy * (nx + 1) + ix;
            vtx2xy[ip][0] = ix.as_();
            vtx2xy[ip][1] = iy.as_();
        }
    }
    let mut quad2vtx = vec![[0usize; 4]; nx * ny];
    for iy in 0..ny {
        #[allow(clippy::identity_op)]
        for ix in 0..nx {
            let iq = iy * nx + ix;
            quad2vtx[iq][0] = (iy + 0) * (nx + 1) + (ix + 0);
            quad2vtx[iq][1] = (iy + 0) * (nx + 1) + (ix + 1);
            quad2vtx[iq][2] = (iy + 1) * (nx + 1) + (ix + 1);
            quad2vtx[iq][3] = (iy + 1) * (nx + 1) + (ix + 0);
        }
    }
    (quad2vtx, vtx2xy)
}

#[test]
fn test_grid_quad2() {
    from_grid::<f32>(12, 5);
    from_grid::<f64>(12, 5);
}
