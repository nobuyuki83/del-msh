# del-msh

This is a static mesh processing utility library for computer graphics prototyping 🪚 and research 🧑‍🔬.   

Originally, the code is written in C++ as [DelFEM2](https://github.com/nobuyuki83/delfem2), then it was completely re-written in Rust 🦀.

📔 See [the documentation generated from code](https://docs.rs/del-msh).

🐍 [Python binding](https://github.com/nobuyuki83/pydel-msh) is available.

> [!WARNING]
> **del-msh** is still in its initial development phase. Crates published to https://crates.io/crates/del-msh in the 0.1.x series do not obey SemVer and are unstable.

## Features

- [x] generating primitive meshes (sphere, cylinder, torus)
- [x] load/save wavefront obj mesh
- [x] unify indexes of texture vertex and position vertex
- [x] one-ring neighborhood 
- [x] adjacent element 
- [x] Kd-tree
- [x] Bounding Box Hierarchy (BVH)
- [x] intersection of triangle mesh (intersection, proximity, CCD)

