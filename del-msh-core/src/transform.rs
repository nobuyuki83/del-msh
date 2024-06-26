//! methods for coordinate transformations

#[allow(clippy::identity_op)]
pub fn normalize_coords3<Real>(vtx2xyz: &mut [Real], size: Real)
where
    Real: num_traits::Float,
{
    let num_vtx = vtx2xyz.len() / 3;
    let mut mins = [Real::one(); 3];
    let mut maxs = [-Real::one(); 3];
    for ivtx in 0..num_vtx {
        let x0 = vtx2xyz[ivtx * 3 + 0];
        let y0 = vtx2xyz[ivtx * 3 + 1];
        let z0 = vtx2xyz[ivtx * 3 + 2];
        if ivtx == 0 {
            mins[0] = x0;
            maxs[0] = x0;
            mins[1] = y0;
            maxs[1] = y0;
            mins[2] = z0;
            maxs[2] = z0;
        } else {
            mins[0] = if x0 < mins[0] { x0 } else { mins[0] };
            maxs[0] = if x0 > maxs[0] { x0 } else { maxs[0] };
            mins[1] = if y0 < mins[1] { y0 } else { mins[1] };
            maxs[1] = if y0 > maxs[1] { y0 } else { maxs[1] };
            mins[2] = if z0 < mins[2] { z0 } else { mins[2] };
            maxs[2] = if z0 > maxs[2] { z0 } else { maxs[2] };
        }
    }
    let half = Real::one() / (Real::one() + Real::one());
    let cntr = [
        (mins[0] + maxs[0]) * half,
        (mins[1] + maxs[1]) * half,
        (mins[2] + maxs[2]) * half,
    ];
    let scale = {
        let mut size0 = maxs[0] - mins[0];
        if maxs[1] - mins[1] > size0 {
            size0 = maxs[1] - mins[1];
        }
        if maxs[2] - mins[2] > size0 {
            size0 = maxs[2] - mins[2];
        }
        size / size0
    };
    for ivtx in 0..num_vtx {
        let x0 = vtx2xyz[ivtx * 3 + 0];
        let y0 = vtx2xyz[ivtx * 3 + 1];
        let z0 = vtx2xyz[ivtx * 3 + 2];
        vtx2xyz[ivtx * 3 + 0] = (x0 - cntr[0]) * scale;
        vtx2xyz[ivtx * 3 + 1] = (y0 - cntr[1]) * scale;
        vtx2xyz[ivtx * 3 + 2] = (z0 - cntr[2]) * scale;
    }
}
