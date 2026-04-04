use del_dlpack::dlpack;
use pyo3::{types::PyModule, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(differential_rasterizer_antialias, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(differential_rasterizer_bwd_antialias, m)?)?;
    Ok(())
}

#[pyo3::pyfunction]
pub fn differential_rasterizer_bwd_antialias(
    _py: Python<'_>,
    edge2vtx_contour: &Bound<'_, PyAny>,
    vtx2xyz: &Bound<'_, PyAny>,
    dldw_vtx2xyz: &Bound<'_, PyAny>,
    transform_world2pix: &Bound<'_, PyAny>,
    dldw_pixval: &Bound<'_, PyAny>,
    pix2tri: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let edge2vtx_contour = del_dlpack::get_managed_tensor_from_pyany(edge2vtx_contour)?;
    let vtx2xyz = del_dlpack::get_managed_tensor_from_pyany(vtx2xyz)?;
    let dldw_vtx2xyz = del_dlpack::get_managed_tensor_from_pyany(dldw_vtx2xyz)?;
    let transform_world2pix = del_dlpack::get_managed_tensor_from_pyany(transform_world2pix)?;
    let dldw_pixval = del_dlpack::get_managed_tensor_from_pyany(dldw_pixval)?;
    let pix2tri = del_dlpack::get_managed_tensor_from_pyany(pix2tri)?;
    //
    let num_contour = del_dlpack::get_shape_tensor(&edge2vtx_contour, 0).unwrap();
    let num_vtx = del_dlpack::get_shape_tensor(&vtx2xyz, 0).unwrap();
    let img_h = del_dlpack::get_shape_tensor(&dldw_pixval, 0).unwrap();
    let img_w = del_dlpack::get_shape_tensor(&dldw_pixval, 1).unwrap();
    let device = edge2vtx_contour.ctx.device_type;
    //
    del_dlpack::check_2d_tensor::<u32>(edge2vtx_contour, num_contour, 2, device).unwrap();
    del_dlpack::check_2d_tensor::<f32>(vtx2xyz, num_vtx, 3, device).unwrap();
    del_dlpack::check_2d_tensor::<f32>(dldw_vtx2xyz, num_vtx, 3, device).unwrap();
    del_dlpack::check_2d_tensor::<f32>(dldw_pixval, img_h, img_w, device).unwrap();
    del_dlpack::check_2d_tensor::<u32>(pix2tri, img_h, img_w, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            let edge2vtx_contour =
                unsafe { del_dlpack::slice_from_tensor::<u32>(edge2vtx_contour).unwrap() };
            let vtx2xyz = unsafe { del_dlpack::slice_from_tensor::<f32>(vtx2xyz).unwrap() };
            let dldw_vtx2xyz =
                unsafe { del_dlpack::slice_from_tensor_mut::<f32>(dldw_vtx2xyz).unwrap() };
            let transform_world2pix =
                unsafe { del_dlpack::slice_from_tensor::<f32>(transform_world2pix).unwrap() };
            let transform_world2pix = arrayref::array_ref![transform_world2pix, 0, 16];
            let dldw_pixval =
                unsafe { del_dlpack::slice_from_tensor::<f32>(dldw_pixval).unwrap() };
            let pix2tri = unsafe { del_dlpack::slice_from_tensor::<u32>(pix2tri).unwrap() };
            //
            del_msh_cpu::differential_rasterizer::bwd_antialias(
                edge2vtx_contour,
                vtx2xyz,
                dldw_vtx2xyz,
                transform_world2pix,
                (img_w as usize, img_h as usize),
                dldw_pixval,
                pix2tri,
            );
            Ok(())
        }
        _ => {
            todo!()
        }
    }
}

#[pyo3::pyfunction]
pub fn differential_rasterizer_antialias(
    _py: Python<'_>,
    edge2vtx_contour: &Bound<'_, PyAny>,
    vtx2xyz: &Bound<'_, PyAny>,
    transform_world2pix: &Bound<'_, PyAny>,
    pix2tri: &Bound<'_, PyAny>,
    img_data: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let edge2vtx_contour = del_dlpack::get_managed_tensor_from_pyany(edge2vtx_contour)?;
    let vtx2xyz = del_dlpack::get_managed_tensor_from_pyany(vtx2xyz)?;
    let transform_world2pix = del_dlpack::get_managed_tensor_from_pyany(transform_world2pix)?;
    let pix2tri = del_dlpack::get_managed_tensor_from_pyany(pix2tri)?;
    let img_data = del_dlpack::get_managed_tensor_from_pyany(img_data)?;
    //
    let num_contour = del_dlpack::get_shape_tensor(&edge2vtx_contour, 0).unwrap();
    let num_vtx = del_dlpack::get_shape_tensor(&vtx2xyz, 0).unwrap();
    let img_h = del_dlpack::get_shape_tensor(&img_data, 0).unwrap();
    let img_w = del_dlpack::get_shape_tensor(&img_data, 1).unwrap();
    let device = edge2vtx_contour.ctx.device_type;
    //
    del_dlpack::check_2d_tensor::<u32>(edge2vtx_contour, num_contour, 2, device).unwrap();
    del_dlpack::check_2d_tensor::<f32>(vtx2xyz, num_vtx, 3, device).unwrap();
    del_dlpack::check_2d_tensor::<f32>(img_data, img_h, img_w, device).unwrap();
    del_dlpack::check_2d_tensor::<u32>(pix2tri, img_h, img_w, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            let edge2vtx_contour =
                unsafe { del_dlpack::slice_from_tensor::<u32>(edge2vtx_contour).unwrap() };
            let vtx2xyz = unsafe { del_dlpack::slice_from_tensor::<f32>(vtx2xyz).unwrap() };
            let transform_world2pix =
                unsafe { del_dlpack::slice_from_tensor::<f32>(transform_world2pix).unwrap() };
            let transform_world2pix = arrayref::array_ref![transform_world2pix, 0, 16];
            let img_data =
                unsafe { del_dlpack::slice_from_tensor_mut::<f32>(img_data).unwrap() };
            let pix2tri = unsafe { del_dlpack::slice_from_tensor::<u32>(pix2tri).unwrap() };
            //
            del_msh_cpu::differential_rasterizer::antialias(
                edge2vtx_contour,
                vtx2xyz,
                transform_world2pix,
                (img_w as usize, img_h as usize),
                img_data,
                pix2tri,
            );
            Ok(())
        }
        _ => {
            todo!()
        }
    }
}
