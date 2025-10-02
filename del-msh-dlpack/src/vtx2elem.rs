use crate::{check_2d_tensor, get_shape_tensor};
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
) -> PyResult<(pyo3::PyObject, pyo3::PyObject)> {
    let elem2vtx = crate::get_managed_tensor_from_pyany(elem2vtx)?;
    //
    let num_elem = get_shape_tensor(elem2vtx, 0);
    let num_node = get_shape_tensor(elem2vtx, 1);
    let device = elem2vtx.ctx.device_type;
    check_2d_tensor::<i32>(elem2vtx, num_elem, num_node, device);
    //
    match device {
        dlpack::device_type_codes::CPU => {
            let elem2vtx = unsafe { crate::slice_from_tensor::<i32>(elem2vtx).unwrap() };
            let (vtx2idx, idx2elem) =
                del_msh_cpu::vtx2elem::from_uniform_mesh(elem2vtx, num_node as usize, num_vtx);
            let vtx2idx_cap = crate::make_capsule_from_vec(py, vec![vtx2idx.len() as i64], vtx2idx);
            let idx2elem_cap =
                crate::make_capsule_from_vec(py, vec![idx2elem.len() as i64], idx2elem);
            Ok((vtx2idx_cap, idx2elem_cap))
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            use del_cudarc_sys::{cu, cuda_check, CuVec, LaunchConfig};
            cuda_check!(cu::cuInit(0));
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            let vtx2valence: CuVec<u32> = CuVec::with_capacity(num_vtx + 1);
            vtx2valence.set_zeros(stream);
            {
                let (func, _mdl) = del_cudarc_sys::load_function_in_module(
                    del_msh_cuda_kernel::UNIFORM_MESH,
                    "vtx2valence",
                );
                let mut builder = del_cudarc_sys::Builder::new(stream);
                builder.arg_i32(num_elem as i32);
                builder.arg_data(&elem2vtx.data);
                builder.arg_i32(num_node as i32);
                builder.arg_dptr(vtx2valence.dptr);
                builder.launch_kernel(func, LaunchConfig::for_num_elems(num_elem as u32));
            }
            let vtx2idx0 = del_cudarc_sys::cumsum::cumsum(&vtx2valence, stream);
            // dbg!(vtx2idx0.copy_to_host());
            let num_idx = vtx2idx0.last();
            let idx2elem: CuVec<u32> = CuVec::with_capacity(num_idx as usize);
            {
                let (func, _mdl) = del_cudarc_sys::load_function_in_module(
                    del_msh_cuda_kernel::UNIFORM_MESH,
                    "fill_idx2vtx",
                );
                let mut builder = del_cudarc_sys::Builder::new(stream);
                builder.arg_i32(num_elem as i32);
                builder.arg_data(&elem2vtx.data);
                builder.arg_i32(num_node as i32);
                builder.arg_dptr(vtx2idx0.dptr);
                builder.arg_dptr(idx2elem.dptr);
                builder.launch_kernel(func, LaunchConfig::for_num_elems(num_elem as u32));
            }
            // dbg!(vtx2idx0.copy_to_host());
            let vtx2idx = del_cudarc_sys::util::shift_array_right(stream, &vtx2idx0);
            del_cudarc_sys::util::sort_indexed_array(stream, &vtx2idx, &idx2elem);
            // dbg!(vtx2idx.copy_to_host());
            // dbg!(idx2elem.copy_to_host());
            let vtx2idx_cap =
                crate::make_capsule_from_cuvec(py, 0, vec![vtx2idx.n as i64], vtx2idx);
            let idx2elem_cap =
                crate::make_capsule_from_cuvec(py, 0, vec![idx2elem.n as i64], idx2elem);
            Ok((vtx2idx_cap, idx2elem_cap))
        }
        _ => {
            todo!()
        }
    }
}
