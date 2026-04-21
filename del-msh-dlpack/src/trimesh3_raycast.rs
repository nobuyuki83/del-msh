use del_dlpack::{
    check_2d_tensor as chk2, dlpack, get_managed_tensor_from_pyany as get_tensor,
    get_shape_tensor as shape, slice, slice_mut,
};
use pyo3::{types::PyModule, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(trimesh3_raycast_pix2tri_by_raycast, m)?)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[pyo3::pyfunction]
pub fn trimesh3_raycast_pix2tri_by_raycast(
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
