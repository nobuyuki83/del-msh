use criterion::{criterion_group, criterion_main, Criterion};
use del_msh_cpu::bvhnodes_morton::from_vtx2xyz;

fn bvh_morton(c: &mut Criterion) {
    let num_vtx = 100000;
    let vtx2xy: Vec<f32> = {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..num_vtx * 2).map(|_| rng.gen::<f32>()).collect()
    };
    c.bench_function("sorted_morton_code2", |b| {
        b.iter(|| from_vtx2xyz::<usize>(&vtx2xy, 2))
    });
}

criterion_group!(benches, bvh_morton);
criterion_main!(benches);
