#[cfg(feature = "cuda")]
use del_cudarc_sys::{cu, CuVec, LaunchConfig};
#[cfg(feature = "cuda")]
use del_dlpack::del_cudarc_sys;
use del_dlpack::{
    check_2d_tensor as chk2, dlpack, get_managed_tensor_from_pyany as get_tensor,
    get_shape_tensor as shape, make_capsule_from_vec as capsule, pyo3, slice,
};
use pyo3::{pyfunction, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<pyo3::types::PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(vtx2elem_from_uniform_mesh, m)?)?;
    Ok(())
}

#[pyfunction]
fn vtx2elem_from_uniform_mesh(
    py: Python<'_>,
    elem2vtx: &Bound<'_, PyAny>,
    num_vtx: usize,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<(pyo3::Py<PyAny>, pyo3::Py<PyAny>)> {
    let elem2vtx = get_tensor(elem2vtx)?;
    //
    let num_elem = shape(elem2vtx, 0).unwrap();
    let num_node = shape(elem2vtx, 1).unwrap();
    let device = elem2vtx.ctx.device_type;
    chk2::<u32>(elem2vtx, num_elem, num_node, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            let s = slice!(elem2vtx, u32).unwrap();
            let (vtx2idx, idx2elem) = match num_node as usize {
                2 => del_msh_cpu::vtx2elem::from_uniform_mesh(s.as_chunks::<2>().0, num_vtx),
                3 => del_msh_cpu::vtx2elem::from_uniform_mesh(s.as_chunks::<3>().0, num_vtx),
                4 => del_msh_cpu::vtx2elem::from_uniform_mesh(s.as_chunks::<4>().0, num_vtx),
                _ => panic!("unsupported num_node: {num_node}"),
            };
            Ok((
                capsule(py, vec![vtx2idx.len() as i64], vtx2idx),
                capsule(py, vec![idx2elem.len() as i64], idx2elem),
            ))
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            use del_cudarc_sys::{cu, cuda_check, CuVec};
            cuda_check!(cu::cuInit(0)).unwrap();
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            use del_cudarc_sys::cu::CUdeviceptr;
            let elem2vtx = CuVec::<u32>::new(
                elem2vtx.data as CUdeviceptr,
                (num_elem * num_node) as usize,
                false,
            );
            let (vtx2idx, idx2elem) =
                from_uniform_mesh(stream, &elem2vtx, num_elem as usize, num_vtx);
            let vtx2idx_cap =
                del_dlpack::make_capsule_from_cuvec(py, 0, vec![vtx2idx.n as i64], vtx2idx);
            let idx2elem_cap =
                del_dlpack::make_capsule_from_cuvec(py, 0, vec![idx2elem.n as i64], idx2elem);
            Ok((vtx2idx_cap, idx2elem_cap))
        }
        _ => Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "GPU not supported (compile with --features cuda)",
        )),
    }
}

// --------------------------------------------

#[cfg(feature = "cuda")]
pub fn from_uniform_mesh(
    stream: cu::CUstream,
    elem2vtx: &CuVec<u32>,
    num_elem: usize,
    num_vtx: usize,
) -> (CuVec<u32>, CuVec<u32>) {
    let num_node = elem2vtx.n / num_elem;
    assert_eq!(elem2vtx.n, num_elem * num_node);
    let vtx2valence: CuVec<u32> = CuVec::with_capacity(num_vtx + 1).unwrap();
    vtx2valence.set_zeros(stream).unwrap();
    {
        /*
        let (func, _mdl) =
            del_cudarc_sys::load_function_in_module(del_msh_cuda_kernel::UNIFORM_MESH, "vtx2nelem")
                .unwrap();
         */
        // let func = crate::load_get_function("uniform_mesh", "vtx2nelem").unwrap();
        let func = del_cudarc_sys::cache_func::get_function_cached(
            "del_msh::uniform_mesh",
            del_msh_cuda_kernels::get("uniform_mesh").unwrap(),
            "vtx2nelem",
        )
        .unwrap();
        let mut builder = del_cudarc_sys::Builder::new(stream);
        builder.arg_u32(num_elem as u32);
        builder.arg_dptr(elem2vtx.dptr);
        builder.arg_u32(num_node as u32);
        builder.arg_dptr(vtx2valence.dptr);
        builder
            .launch_kernel(
                func,
                LaunchConfig::from_num_thread_for_max_occupancy(num_elem as u32, func, 0, 0)
                    .unwrap(),
            )
            .unwrap();
    }
    let vtx2idx0 = CuVec::<u32>::with_capacity(vtx2valence.n).unwrap();
    del_cudarc_sys::cumsum::exclusive_scan(stream, &vtx2valence, &vtx2idx0);
    // dbg!(vtx2idx0.copy_to_host());
    let num_idx = vtx2idx0.last().unwrap();
    let idx2elem: CuVec<u32> = CuVec::with_capacity(num_idx as usize).unwrap();
    {
        /*
        let (func, _mdl) = del_cudarc_sys::load_function_in_module(
            del_msh_cuda_kernel::UNIFORM_MESH,
            "fill_idx2elem",
        )
        .unwrap();
         */
        // let func = crate::load_get_function("uniform_mesh", "fill_idx2elem").unwrap();
        let func = del_cudarc_sys::cache_func::get_function_cached(
            "del_msh::uniform_mesh",
            del_msh_cuda_kernels::get("uniform_mesh").unwrap(),
            "fill_idx2elem",
        )
        .unwrap();
        let mut builder = del_cudarc_sys::Builder::new(stream);
        builder.arg_u32(num_elem as u32);
        builder.arg_dptr(elem2vtx.dptr);
        builder.arg_u32(num_node as u32);
        builder.arg_dptr(vtx2idx0.dptr);
        builder.arg_dptr(idx2elem.dptr);
        builder
            .launch_kernel(
                func,
                LaunchConfig::from_num_thread_for_max_occupancy(num_elem as u32, func, 0, 0)
                    .unwrap(),
            )
            .unwrap();
    }
    // dbg!(vtx2idx0.copy_to_host());
    let vtx2idx = del_cudarc_sys::array1d::shift_array_right(stream, &vtx2idx0);
    del_cudarc_sys::offset_array::sort(stream, &vtx2idx, &idx2elem);
    (vtx2idx, idx2elem)
}
