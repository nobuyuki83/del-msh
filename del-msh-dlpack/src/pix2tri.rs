use del_dlpack::{
    check_2d_tensor as chk2, check_3d_tensor as chk3, dlpack,
    get_managed_tensor_from_pyany as get_tensor, get_shape_tensor as shape, slice, slice_mut,
};
use pyo3::{types::PyModule, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(pix2tri_by_raycast, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(pix2tri_interpolate, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(pix2tri_interpolate_bwd, m)?)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[pyo3::pyfunction]
pub fn pix2tri_by_raycast(
    _py: Python<'_>,
    pix2tri: &Bound<'_, PyAny>,
    tri2vtx: &Bound<'_, PyAny>,
    vtx2xyz: &Bound<'_, PyAny>,
    bvhnodes: &Bound<'_, PyAny>,
    bvhnode2aabb: &Bound<'_, PyAny>,
    transform_ndc2world: &Bound<'_, PyAny>,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let pix2tri = get_tensor(pix2tri)?;
    let tri2vtx = get_tensor(tri2vtx)?;
    let vtx2xyz = get_tensor(vtx2xyz)?;
    let bvhnodes = get_tensor(bvhnodes)?;
    let bvhnode2aabb = get_tensor(bvhnode2aabb)?;
    let transform_ndc2world = get_tensor(transform_ndc2world)?;
    //
    let device = pix2tri.ctx.device_type;
    let img_shape = [shape(pix2tri, 1).unwrap(), shape(pix2tri, 0).unwrap()];
    let num_tri = shape(tri2vtx, 0).unwrap();
    let num_vtx = shape(vtx2xyz, 0).unwrap();
    let num_bvhnode = shape(bvhnodes, 0).unwrap();
    //
    chk2::<u32>(pix2tri, img_shape[1], img_shape[0], device).unwrap();
    chk2::<u32>(tri2vtx, num_tri, 3, device).unwrap();
    chk2::<f32>(vtx2xyz, num_vtx, 3, device).unwrap();
    chk2::<u32>(bvhnodes, num_bvhnode, 3, device).unwrap();
    chk2::<f32>(bvhnode2aabb, num_bvhnode, 6, device).unwrap();
    chk2::<f32>(transform_ndc2world, 4, 4, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            del_msh_cpu::pix2tri::pix2tri_by_raycast(
                slice_mut!(pix2tri, u32).unwrap(),
                slice!(tri2vtx, u32).unwrap(),
                slice!(vtx2xyz, f32).unwrap(),
                slice!(bvhnodes, u32).unwrap(),
                slice!(bvhnode2aabb, f32).unwrap(),
                (img_shape[0] as usize, img_shape[1] as usize),
                arrayref::array_ref![slice!(transform_ndc2world, f32).unwrap(), 0, 16],
            );
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            use del_cudarc_sys::{cu, cuda_check};
            cuda_check!(cu::cuInit(0)).unwrap();
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            //
            let func = del_cudarc_sys::cache_func::get_function_cached(
                "del_msh::pix2tri",
                del_msh_cuda_kernels::get("pix2tri").unwrap(),
                "pix_to_tri",
            )
            .unwrap();
            let num_pix = (img_shape[0] * img_shape[1]) as usize;
            let mut builder = del_cudarc_sys::Builder::new(stream);
            builder.arg_data(&pix2tri.data);
            builder.arg_u32(num_tri as u32);
            builder.arg_data(&tri2vtx.data);
            builder.arg_data(&vtx2xyz.data);
            builder.arg_u32(img_shape[0] as u32);
            builder.arg_u32(img_shape[1] as u32);
            builder.arg_data(&transform_ndc2world.data);
            builder.arg_data(&bvhnodes.data);
            builder.arg_data(&bvhnode2aabb.data);
            builder
                .launch_kernel(
                    func,
                    del_cudarc_sys::LaunchConfig::for_num_elems(num_pix as u32),
                )
                .unwrap();
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}

#[pyo3::pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn pix2tri_interpolate(
    _py: Python<'_>,
    pix2tri: &Bound<'_, PyAny>,
    tri2vtx: &Bound<'_, PyAny>,
    vtx2xyz: &Bound<'_, PyAny>,
    vtx2val: &Bound<'_, PyAny>,
    transform_ndc2world: &Bound<'_, PyAny>,
    pix2val: &Bound<'_, PyAny>,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let pix2tri = get_tensor(pix2tri)?;
    let tri2vtx = get_tensor(tri2vtx)?;
    let vtx2xyz = get_tensor(vtx2xyz)?;
    let vtx2val = get_tensor(vtx2val)?;
    let transform_ndc2world = get_tensor(transform_ndc2world)?;
    let pix2val = get_tensor(pix2val)?;
    //
    let img_h = shape(pix2tri, 0).unwrap();
    let img_w = shape(pix2tri, 1).unwrap();
    let num_tri = shape(tri2vtx, 0).unwrap();
    let num_vtx = shape(vtx2xyz, 0).unwrap();
    let num_vdim = shape(vtx2val, 1).unwrap();
    let device = pix2tri.ctx.device_type;
    //
    chk2::<u32>(pix2tri, img_h, img_w, device).unwrap();
    chk2::<u32>(tri2vtx, num_tri, 3, device).unwrap();
    chk2::<f32>(vtx2xyz, num_vtx, 3, device).unwrap();
    chk2::<f32>(vtx2val, num_vtx, num_vdim, device).unwrap();
    chk2::<f32>(transform_ndc2world, 4, 4, device).unwrap();
    chk3::<f32>(pix2val, img_h, img_w, num_vdim, device).unwrap();

    match device {
        dlpack::device_type_codes::CPU => {
            del_msh_cpu::pix2tri::interpolate(
                (img_w as usize, img_h as usize),
                slice!(pix2tri, u32).unwrap(),
                slice!(tri2vtx, u32).unwrap().as_chunks::<3>().0,
                slice!(vtx2xyz, f32).unwrap().as_chunks::<3>().0,
                num_vdim as usize,
                slice!(vtx2val, f32).unwrap(),
                arrayref::array_ref![slice!(transform_ndc2world, f32).unwrap(), 0, 16],
                slice_mut!(pix2val, f32).unwrap(),
            );
        }
        _ => {
            todo!("unsupported device")
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[pyo3::pyfunction]
pub fn pix2tri_interpolate_bwd(
    _py: Python<'_>,
    pix2tri: &Bound<'_, PyAny>,
    tri2vtx: &Bound<'_, PyAny>,
    vtx2xyz: &Bound<'_, PyAny>,
    vtx2val: &Bound<'_, PyAny>,
    transform_ndc2world: &Bound<'_, PyAny>,
    dldw_pix2val: &Bound<'_, PyAny>,
    dldw_vtx2xyz: &Bound<'_, PyAny>,
    dldw_vtx2val: &Bound<'_, PyAny>,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let pix2tri = get_tensor(pix2tri)?;
    let tri2vtx = get_tensor(tri2vtx)?;
    let vtx2xyz = get_tensor(vtx2xyz)?;
    let vtx2val = get_tensor(vtx2val)?;
    let transform_ndc2world = get_tensor(transform_ndc2world)?;
    let dldw_pix2val = get_tensor(dldw_pix2val)?;
    let dldw_vtx2xyz = get_tensor(dldw_vtx2xyz)?;
    let dldw_vtx2val = get_tensor(dldw_vtx2val)?;
    //
    let img_h = shape(pix2tri, 0).unwrap();
    let img_w = shape(pix2tri, 1).unwrap();
    let num_tri = shape(tri2vtx, 0).unwrap();
    let num_vtx = shape(vtx2xyz, 0).unwrap();
    let num_vdim = shape(vtx2val, 1).unwrap();
    let device = pix2tri.ctx.device_type;
    //
    chk2::<u32>(pix2tri, img_h, img_w, device).unwrap();
    chk2::<u32>(tri2vtx, num_tri, 3, device).unwrap();
    chk2::<f32>(vtx2xyz, num_vtx, 3, device).unwrap();
    chk2::<f32>(vtx2val, num_vtx, num_vdim, device).unwrap();
    chk2::<f32>(transform_ndc2world, 4, 4, device).unwrap();
    chk3::<f32>(dldw_pix2val, img_h, img_w, num_vdim, device).unwrap();
    chk2::<f32>(dldw_vtx2xyz, num_vtx, 3, device).unwrap();
    chk2::<f32>(dldw_vtx2val, num_vtx, num_vdim, device).unwrap();

    match device {
        dlpack::device_type_codes::CPU => {
            del_msh_cpu::pix2tri::interpolate_bwd(
                (img_w as usize, img_h as usize),
                slice!(pix2tri, u32).unwrap(),
                slice!(tri2vtx, u32).unwrap().as_chunks::<3>().0,
                slice!(vtx2xyz, f32).unwrap().as_chunks::<3>().0,
                num_vdim as usize,
                slice!(vtx2val, f32).unwrap(),
                arrayref::array_ref![slice!(transform_ndc2world, f32).unwrap(), 0, 16],
                slice!(dldw_pix2val, f32).unwrap(),
                slice_mut!(dldw_vtx2xyz, f32)
                    .unwrap()
                    .as_chunks_mut::<3>()
                    .0,
                slice_mut!(dldw_vtx2val, f32).unwrap(),
            );
        }
        _ => {
            todo!("unsupported device")
        }
    }
    Ok(())
}
