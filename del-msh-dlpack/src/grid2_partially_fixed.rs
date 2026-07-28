use del_dlpack::{
    check_2d_tensor as chk2, check_3d_tensor as chk3, dlpack,
    get_managed_tensor_from_pyany as get_tensor, get_shape_tensor as shape, slice, slice_mut,
};
use pyo3::{types::PyModule, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(grid2_nearest_to_fixed_cell, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(grid2_smooth_gauss_seidel, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        grid2_smooth_gauss_seidel_with_radius,
        m
    )?)?;
    Ok(())
}

// cell2isfixed: (h, w) u8; cell2nearest: (h, w) u32; cell2distance: (h, w) f32
// img shape is derived from cell2isfixed; GPU path uses Jump Flooding Algorithm
#[pyo3::pyfunction]
pub fn grid2_nearest_to_fixed_cell(
    _py: Python<'_>,
    cell2isfixed: &Bound<'_, PyAny>,
    cell2nearest: &Bound<'_, PyAny>,
    cell2distance: &Bound<'_, PyAny>,
    is_initial: bool,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let cell2isfixed = get_tensor(cell2isfixed)?;
    let cell2nearest = get_tensor(cell2nearest)?;
    let cell2distance = get_tensor(cell2distance)?;
    //
    let device = cell2isfixed.ctx.device_type;
    let img_w = shape(cell2isfixed, 1).unwrap();
    let img_h = shape(cell2isfixed, 0).unwrap();
    //
    chk2::<u8>(cell2isfixed, img_h, img_w, device).unwrap();
    chk2::<u32>(cell2nearest, img_h, img_w, device).unwrap();
    chk2::<f32>(cell2distance, img_h, img_w, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            del_msh_cpu::grid2_partially_fixed::nearest_to_fixed_cell(
                (img_w as usize, img_h as usize),
                slice!(cell2isfixed, u8).unwrap(),
                slice_mut!(cell2nearest, u32).unwrap(),
                slice_mut!(cell2distance, f32).unwrap(),
                is_initial,
            );
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            use del_cudarc_sys::{cu, cuda_check, CuVec};
            cuda_check!(cu::cuInit(0)).unwrap();
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            let num_cell = (img_w * img_h) as u32;
            let cfg = del_cudarc_sys::LaunchConfig::for_num_elems(num_cell);
            let fatbin = del_msh_cuda_kernels::get("grid2_partially_fixed").unwrap();
            let fn_init = del_cudarc_sys::cache_func::get_function_cached(
                "del_msh__grid2_partially_fixed",
                fatbin,
                "nearest_to_fixed_cell_initialize",
            )
            .unwrap();
            let fn_jfa = del_cudarc_sys::cache_func::get_function_cached(
                "del_msh__grid2_partially_fixed",
                fatbin,
                "nearest_to_fixed_cell_jump_flood",
            )
            .unwrap();
            let fn_dist = del_cudarc_sys::cache_func::get_function_cached(
                "del_msh__grid2_partially_fixed",
                fatbin,
                "nearest_to_fixed_cell_compute_distance",
            )
            .unwrap();
            // Initialize cell2nearest from cell2isfixed
            {
                let mut b = del_cudarc_sys::Builder::new(stream);
                b.arg_u32(img_w as u32);
                b.arg_u32(img_h as u32);
                b.arg_data(&cell2isfixed.data);
                b.arg_data(&cell2nearest.data);
                b.launch_kernel(fn_init, cfg).unwrap();
            }
            // JFA ping-pong: allocate a temp buffer for the second slot
            let tmp = CuVec::<u32>::with_capacity(num_cell as usize).unwrap();
            let cell_dptr = cell2nearest.data as cu::CUdeviceptr;
            let mut src_dptr = cell_dptr;
            let mut dst_dptr = tmp.dptr;
            // Start step = largest power of 2 <= max(w, h), halve until step = 0
            let max_dim = (img_w as u32).max(img_h as u32);
            let mut step = max_dim.next_power_of_two();
            if step > 1 {
                step >>= 1;
            }
            while step >= 1 {
                {
                    let mut b = del_cudarc_sys::Builder::new(stream);
                    b.arg_u32(img_w as u32);
                    b.arg_u32(img_h as u32);
                    b.arg_dptr(src_dptr);
                    b.arg_dptr(dst_dptr);
                    b.arg_u32(step);
                    b.launch_kernel(fn_jfa, cfg).unwrap();
                }
                std::mem::swap(&mut src_dptr, &mut dst_dptr);
                step >>= 1;
            }
            // If the final result ended up in the temp buffer, copy it to cell2nearest
            if src_dptr != cell_dptr {
                del_cudarc_sys::memcpy_d2d_32(cell_dptr, src_dptr, num_cell as usize, stream)
                    .unwrap();
            }
            // Compute Euclidean distances from nearest indices
            {
                let mut b = del_cudarc_sys::Builder::new(stream);
                b.arg_u32(img_w as u32);
                b.arg_u32(img_h as u32);
                b.arg_data(&cell2nearest.data);
                b.arg_data(&cell2distance.data);
                b.launch_kernel(fn_dist, cfg).unwrap();
            }
        }
        _ => {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "nearest_to_fixed_cell is only implemented for CPU and CUDA",
            ));
        }
    }
    Ok(())
}

// cell2isfix: (h, w) u8; cell2val: (h, w, num_vdim) f32
// One full iteration = two red-black passes on GPU.
#[pyo3::pyfunction]
pub fn grid2_smooth_gauss_seidel(
    _py: Python<'_>,
    cell2isfix: &Bound<'_, PyAny>,
    cell2val: &Bound<'_, PyAny>,
    num_iter: usize,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let cell2isfix = get_tensor(cell2isfix)?;
    let cell2val = get_tensor(cell2val)?;
    //
    let device = cell2isfix.ctx.device_type;
    let img_h = shape(cell2isfix, 0).unwrap();
    let img_w = shape(cell2isfix, 1).unwrap();
    let num_vdim = shape(cell2val, 2).unwrap();
    //
    chk2::<u8>(cell2isfix, img_h, img_w, device).unwrap();
    chk3::<f32>(cell2val, img_h, img_w, num_vdim, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            for _iter in 0..num_iter {
                del_msh_cpu::grid2_partially_fixed::smooth_gauss_seidel(
                    (img_w as usize, img_h as usize),
                    slice!(cell2isfix, u8).unwrap(),
                    num_vdim as usize,
                    slice_mut!(cell2val, f32).unwrap(),
                );
            }
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            let function = del_cudarc_sys::cache_func::get_function_cached(
                "del_msh__grid2_partially_fixed",
                del_msh_cuda_kernels::get("grid2_partially_fixed").unwrap(),
                "smooth_red_black_gauss_seidel",
            )
            .unwrap();
            use del_cudarc_sys::{cu, cuda_check};
            cuda_check!(cu::cuInit(0)).unwrap();
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            let num_cell = (img_w * img_h) as u32;
            for i_iter in 0..num_iter * 2 {
                // Two passes per iteration: red (color=0) then black (color=1)
                let mut builder = del_cudarc_sys::Builder::new(stream);
                builder.arg_u32(img_w as u32);
                builder.arg_u32(img_h as u32);
                builder.arg_data(&cell2isfix.data);
                builder.arg_u32(num_vdim as u32);
                builder.arg_data(&cell2val.data);
                builder.arg_u32((i_iter % 2) as u32);
                builder
                    .launch_kernel(
                        function,
                        del_cudarc_sys::LaunchConfig::for_num_elems(num_cell),
                    )
                    .unwrap();
            }
        }
        _ => {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "smooth_gauss_seidel is only implemented for CPU and CUDA",
            ));
        }
    }
    Ok(())
}

// cell2isfixed: (h, w) u8; cell2dist: (h, w) f32; cell2val: (h, w, num_vdim) f32
// GPU path: one Jacobi step with a temp buffer (ping-pong), result copied back to cell2val.
#[pyo3::pyfunction]
pub fn grid2_smooth_gauss_seidel_with_radius(
    _py: Python<'_>,
    cell2isfixed: &Bound<'_, PyAny>,
    cell2dist: &Bound<'_, PyAny>,
    ratio: f32,
    cell2val: &Bound<'_, PyAny>,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let cell2isfixed = get_tensor(cell2isfixed)?;
    let cell2dist = get_tensor(cell2dist)?;
    let cell2val = get_tensor(cell2val)?;
    //
    let device = cell2isfixed.ctx.device_type;
    let img_h = shape(cell2isfixed, 0).unwrap();
    let img_w = shape(cell2isfixed, 1).unwrap();
    let num_vdim = shape(cell2val, 2).unwrap();
    //
    chk2::<u8>(cell2isfixed, img_h, img_w, device).unwrap();
    chk2::<f32>(cell2dist, img_h, img_w, device).unwrap();
    chk3::<f32>(cell2val, img_h, img_w, num_vdim, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            del_msh_cpu::grid2_partially_fixed::smooth_gauss_seidel_with_radius(
                (img_w as usize, img_h as usize),
                slice!(cell2isfixed, u8).unwrap(),
                slice!(cell2dist, f32).unwrap(),
                ratio,
                num_vdim as usize,
                slice_mut!(cell2val, f32).unwrap(),
            );
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            use del_cudarc_sys::{cu, cuda_check, CuVec};
            cuda_check!(cu::cuInit(0)).unwrap();
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            let num_cell = (img_w * img_h) as u32;
            let num_vals = num_cell as usize * num_vdim as usize;
            let function = del_cudarc_sys::cache_func::get_function_cached(
                "del_msh__grid2_partially_fixed",
                del_msh_cuda_kernels::get("grid2_partially_fixed").unwrap(),
                "smooth_gauss_seidel_with_radius",
            )
            .unwrap();
            // Temp buffer acts as the write target (Jacobi ping-pong)
            let tmp = CuVec::<f32>::with_capacity(num_vals).unwrap();
            {
                let mut b = del_cudarc_sys::Builder::new(stream);
                b.arg_u32(img_w as u32);
                b.arg_u32(img_h as u32);
                b.arg_data(&cell2isfixed.data);
                b.arg_data(&cell2dist.data);
                b.arg_f32(ratio);
                b.arg_u32(num_vdim as u32);
                b.arg_data(&cell2val.data); // pre (read)
                b.arg_dptr(tmp.dptr); // pos (write)
                b.launch_kernel(
                    function,
                    del_cudarc_sys::LaunchConfig::for_num_elems(num_cell),
                )
                .unwrap();
            }
            // Copy result back to cell2val in-place
            del_cudarc_sys::memcpy_d2d_32(
                cell2val.data as cu::CUdeviceptr,
                tmp.dptr,
                num_vals,
                stream,
            )
            .unwrap();
        }
        _ => {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "smooth_gauss_seidel_with_radius is only implemented for CPU and CUDA",
            ));
        }
    }
    Ok(())
}
