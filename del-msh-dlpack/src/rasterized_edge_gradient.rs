use del_dlpack::{
    check_1d_tensor as chk1, check_2d_tensor as chk2, dlpack,
    get_managed_tensor_from_pyany as get_tensor, get_shape_tensor as shape, slice, slice_mut,
};
use pyo3::{types::PyModule, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(rasterized_edge_gradient_bwd, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        rasterized_edge_gradient_edge_gradient_and_type,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        rasterized_edge_gradient_smooth_gradient,
        m
    )?)?;
    Ok(())
}

#[pyo3::pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn rasterized_edge_gradient_bwd(
    _py: Python<'_>,
    tri2vtx: &Bound<'_, PyAny>,
    vtx2xyz: &Bound<'_, PyAny>,
    dldw_vtx2xyz: &Bound<'_, PyAny>,
    transform_world2pix: &Bound<'_, PyAny>,
    dldw_pixval: &Bound<'_, PyAny>,
    pix2tri: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let tri2vtx = get_tensor(tri2vtx)?;
    let vtx2xyz = get_tensor(vtx2xyz)?;
    let dldw_vtx2xyz = get_tensor(dldw_vtx2xyz)?;
    let transform_world2pix = get_tensor(transform_world2pix)?;
    let dldw_pixval = get_tensor(dldw_pixval)?;
    let pix2tri = get_tensor(pix2tri)?;
    //
    let device = tri2vtx.ctx.device_type;
    let num_tri = shape(tri2vtx, 0).unwrap();
    let num_vtx = shape(vtx2xyz, 0).unwrap();
    let img_h = shape(pix2tri, 0).unwrap();
    let img_w = shape(pix2tri, 1).unwrap();
    //
    chk2::<u32>(tri2vtx, num_tri, 3, device).unwrap();
    chk2::<f32>(vtx2xyz, num_vtx, 3, device).unwrap();
    chk2::<f32>(dldw_vtx2xyz, num_vtx, 3, device).unwrap();
    chk1::<f32>(transform_world2pix, 16, device).unwrap();
    chk2::<f32>(dldw_pixval, img_h, img_w, device).unwrap();
    chk2::<u32>(pix2tri, img_h, img_w, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            del_msh_cpu::rasterized_edge_gradient::bwd(
                slice!(tri2vtx, u32).unwrap(),
                slice!(vtx2xyz, f32).unwrap(),
                slice_mut!(dldw_vtx2xyz, f32).unwrap(),
                arrayref::array_ref![slice!(transform_world2pix, f32).unwrap(), 0, 16],
                (img_w as usize, img_h as usize),
                slice!(dldw_pixval, f32).unwrap(),
                slice!(pix2tri, u32).unwrap(),
            );
        }
        _ => {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "GPU not supported",
            ));
        }
    }
    Ok(())
}

#[pyo3::pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn rasterized_edge_gradient_edge_gradient_and_type(
    _py: Python<'_>,
    tri2vtx: &Bound<'_, PyAny>,
    vtx2xyz: &Bound<'_, PyAny>,
    transform_world2pix: &Bound<'_, PyAny>,
    dldw_pixval: &Bound<'_, PyAny>,
    pix2tri: &Bound<'_, PyAny>,
    hedge2type: &Bound<'_, PyAny>,
    hedge2dldr: &Bound<'_, PyAny>,
    vedge2type: &Bound<'_, PyAny>,
    vedge2dldr: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let tri2vtx = get_tensor(tri2vtx)?;
    let vtx2xyz = get_tensor(vtx2xyz)?;
    let transform_world2pix = get_tensor(transform_world2pix)?;
    let dldw_pixval = get_tensor(dldw_pixval)?;
    let pix2tri = get_tensor(pix2tri)?;
    let hedge2type = get_tensor(hedge2type)?;
    let hedge2dldr = get_tensor(hedge2dldr)?;
    let vedge2type = get_tensor(vedge2type)?;
    let vedge2dldr = get_tensor(vedge2dldr)?;
    //
    let device = tri2vtx.ctx.device_type;
    let num_tri = shape(tri2vtx, 0).unwrap();
    let num_vtx = shape(vtx2xyz, 0).unwrap();
    let img_h = shape(pix2tri, 0).unwrap();
    let img_w = shape(pix2tri, 1).unwrap();
    //
    chk2::<u32>(tri2vtx, num_tri, 3, device).unwrap();
    chk2::<f32>(vtx2xyz, num_vtx, 3, device).unwrap();
    chk1::<f32>(transform_world2pix, 16, device).unwrap();
    chk2::<f32>(dldw_pixval, img_h, img_w, device).unwrap();
    chk2::<u32>(pix2tri, img_h, img_w, device).unwrap();
    chk2::<u8>(hedge2type, img_h - 1, img_w, device).unwrap();
    chk2::<f32>(hedge2dldr, img_h - 1, img_w, device).unwrap();
    chk2::<u8>(vedge2type, img_h, img_w - 1, device).unwrap();
    chk2::<f32>(vedge2dldr, img_h, img_w - 1, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            del_msh_cpu::rasterized_edge_gradient::edge_gradient_and_type(
                slice!(tri2vtx, u32).unwrap(),
                slice!(vtx2xyz, f32).unwrap(),
                arrayref::array_ref![slice!(transform_world2pix, f32).unwrap(), 0, 16],
                (img_w as usize, img_h as usize),
                slice!(dldw_pixval, f32).unwrap(),
                slice!(pix2tri, u32).unwrap(),
                slice_mut!(hedge2type, u8).unwrap(),
                slice_mut!(hedge2dldr, f32).unwrap(),
                slice_mut!(vedge2type, u8).unwrap(),
                slice_mut!(vedge2dldr, f32).unwrap(),
            );
        }
        _ => {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "GPU not supported",
            ));
        }
    }
    Ok(())
}

#[pyo3::pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn rasterized_edge_gradient_smooth_gradient(
    _py: Python<'_>,
    hedge2type: &Bound<'_, PyAny>,
    hedge2dldr: &Bound<'_, PyAny>,
    vedge2type: &Bound<'_, PyAny>,
    vedge2dldr: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let hedge2type = get_tensor(hedge2type)?;
    let hedge2dldr = get_tensor(hedge2dldr)?;
    let vedge2type = get_tensor(vedge2type)?;
    let vedge2dldr = get_tensor(vedge2dldr)?;
    //
    let device = hedge2type.ctx.device_type;
    let img_h_m1 = shape(hedge2type, 0).unwrap(); // H-1
    let img_w    = shape(hedge2type, 1).unwrap(); // W
    let img_h    = img_h_m1 + 1;
    //
    chk2::<u8> (hedge2type, img_h_m1, img_w,     device).unwrap();
    chk2::<f32>(hedge2dldr, img_h_m1, img_w,     device).unwrap();
    chk2::<u8> (vedge2type, img_h,    img_w - 1, device).unwrap();
    chk2::<f32>(vedge2dldr, img_h,    img_w - 1, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            del_msh_cpu::rasterized_edge_gradient::smooth_gradient(
                (img_w as usize, img_h as usize),
                slice_mut!(hedge2type, u8).unwrap(),
                slice_mut!(hedge2dldr, f32).unwrap(),
                slice_mut!(vedge2type, u8).unwrap(),
                slice_mut!(vedge2dldr, f32).unwrap(),
            );
        }
        _ => {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "GPU not supported",
            ));
        }
    }
    Ok(())
}
