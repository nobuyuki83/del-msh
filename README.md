# del-msh

This is a utility library for static mesh written completely in Rust. 

Originally, the code is written in C++ in [DelFEM2](https://github.com/nobuyuki83/delfem2), then it was ported to Rust. 

See [the documentation generated from code](https://docs.rs/del-msh)

[!WARNING]
`del-msh` is still in its initial development phase. Crates published to https://crates.io in the 0.0.x series do not obey SemVer and are unstable.

- [x] generating primitive meshes (sphere, cylinder, torus)
- [x] load/save wavefront obj mesh
- [x] unify indexes of texture vertex and position vertex
- [x] one-ring neighborhood 
- [x] adjacent element 
- [x] Kd-tree
- [x] Bounding Box Hierarchy (BVH)