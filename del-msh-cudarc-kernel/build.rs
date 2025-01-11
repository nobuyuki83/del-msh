fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/bvhnode2aabb.cu");
    println!("cargo:rerun-if-changed=src/bvhnodes_morton.cu");
    println!("cargo:rerun-if-changed=src/aabb3_from_vtx2xyz.cu");
    println!("cargo:rerun-if-changed=src/edge2vtx.cu");
    //
    let path_out_dir = std::env::var("OUT_DIR").unwrap();
    //let path_out_dir = std::path::Path::new(&path_out_dir).join("cpp_headers").join("del_geo");
    let path_out_dir = std::path::Path::new(&path_out_dir).join("del_geo");
    // dbg!(&path_out_dir);
    std::fs::create_dir_all(&path_out_dir).unwrap();
    del_geo_cpp_headers::HEADERS.write_files(&path_out_dir);
    // dbg!("hoge");
    let glob_input = path_out_dir.join("*.h").into_os_string().into_string().unwrap();
    let builder = bindgen_cuda::Builder::default().include_paths_glob(&glob_input);
    //let path_out_dir = std::path::Path::new(&path_out_dir).join("cpp_headers").join("del_geo");
    //let builder = bindgen_cuda::Builder::default().include_paths(vec!(path_out_dir));
    //let builder = bindgen_cuda::Builder::default();
    //
    // let builder = bindgen_cuda::Builder::default().include_paths_glob("../cpp_header/*");
    println!("cargo:info={builder:?}");
    let bindings = builder.build_ptx().unwrap();
    bindings.write("src/lib.rs").unwrap();
}
