# How to run the examples

1. install rust: https://www.rust-lang.org/tools/install
```bash
# after the installation, please check the installation by  
cargo --version
```

2. download the library and check it can be compiled
```bash
git clone https://github.com/nobuyuki83/del-msh.git
cd del-msh
cargo test del-msh-cpu --release
```

3. running the demos
```
# running `0_trimesh3_remove_intersections` demo 
cd del-msh
cargo run --package del-msh-core --example 0_trimesh3_remove_intersections --release
# you will see input mesh and output mesh in `del-msh/target` directory 
```
