#[cfg(feature = "cuda")]
pub mod vtx2elem;

#[cfg(feature = "cuda")]
pub mod vtx2vtx;

#[cfg(feature = "cuda")]
pub fn load_get_function(
    file_name: &str,
    function_name: &str,
) -> Result<del_cudarc_sys::cu::CUfunction, String> {
    use del_cudarc_sys::cu;
    let fatbin = del_msh_cuda_kernels::get(file_name).ok_or("missing add.fatbin in kernels")?;
    let mut module: cu::CUmodule = std::ptr::null_mut();
    let res = unsafe { cu::cuModuleLoadData(&mut module as *mut _, fatbin.as_ptr() as *const _) };
    if res != cu::CUresult::CUDA_SUCCESS {
        return Err(format!("cuModuleLoadData failed: {:?}", res));
    }
    //
    let cname = std::ffi::CString::new(function_name).unwrap();
    let mut f: cu::CUfunction = std::ptr::null_mut();
    let res = unsafe { cu::cuModuleGetFunction(&mut f, module, cname.as_ptr()) };
    if res != cu::CUresult::CUDA_SUCCESS {
        return Err(format!("program not found: {}", "get_version"));
    }
    Ok(f)
}
