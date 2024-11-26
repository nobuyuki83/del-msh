fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/bvhnode2aabb.cu");
    println!("cargo:rerun-if-changed=src/bvhnodes_morton.cu");
    println!("cargo:rerun-if-changed=src/aabb3_from_vtx2xyz.cu");

    let builder = bindgen_cuda::Builder::default().include_paths_glob("../cpp_header/*");
    println!("cargo:info={builder:?}");
    let bindings = builder.build_ptx().unwrap();
    bindings.write("src/lib.rs").unwrap();
}
