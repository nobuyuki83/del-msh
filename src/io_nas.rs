fn parse(s: &str) -> f32 {
    if let Some(i) = s.rfind('-') {
        if i != 0 {
            let mut s = String::from(s);
            s.insert(i, 'e');
            let v = s.trim().parse::<f32>().unwrap();
            //  dbg!(i,&s,&v);
            return v;
        }
    } else if let Some(i) = s.rfind('+') {
        assert_ne!(i, 0);
        let mut s = String::from(s);
        s.insert(i, 'e');
        let v = s.trim().parse::<f32>().unwrap();
        //  dbg!(i,&s,&v);
        return v;
    }
    s.trim().parse::<f32>().ok().unwrap()
}

pub fn load_tri_mesh<P: AsRef<std::path::Path>>(
    path: P) -> (Vec<usize>, Vec<f32>)
{
    let mut vtx2xyz = vec!(0f32; 0);
    let mut vtx2idx = vec!(0usize; 0);
    let mut tri2idx = vec!(0usize; 0);
    //
    let file = std::fs::File::open(path).expect("file not found.");
    let reader = std::io::BufReader::new(file);
    use std::io::BufRead;
    for line in reader.lines().map_while(Result::ok) {
        // println!("{:?}", line);
        if line.starts_with("GRID") {
            assert_eq!(line.len(), 48);
            let _ = line.get(0..9);
            let idx = line.get(9..24);
            let idx = idx.unwrap().trim().parse::<usize>().unwrap();
            let v0 = line.get(24..32).unwrap();
            let v1 = line.get(32..40).unwrap();
            let v2 = line.get(40..48).unwrap();
            let v0 = parse(v0);
            let v1 = parse(v1);
            let v2 = parse(v2);
            vtx2xyz.push(v0);
            vtx2xyz.push(v1);
            vtx2xyz.push(v2);
            vtx2idx.push(idx);
        } else if line.starts_with("CTRIA3") {
            assert_eq!(line.len(), 48);
            let _ = line.get(0..8); // CTRIA3
            let _ = line.get(8..16);
            let _ = line.get(16..24); // I don't know what this number is for
            let v0 = line.get(24..32).unwrap().trim();
            let v1 = line.get(32..40).unwrap().trim();
            let v2 = line.get(40..48).unwrap().trim();
            let v0 = v0.parse::<usize>().unwrap();
            let v1 = v1.parse::<usize>().unwrap();
            let v2 = v2.parse::<usize>().unwrap();
            tri2idx.push(v0);
            tri2idx.push(v1);
            tri2idx.push(v2);
        }
    }
    //
    let &num_idx = vtx2idx.iter().max().unwrap();
    let mut idx2vtx = vec!(usize::MAX; num_idx + 1);
    vtx2idx.iter().enumerate().for_each(|(vtx, &idx)| idx2vtx[idx] = vtx);
    let mut tri2vtx = vec!(0usize; tri2idx.len());
    tri2vtx.iter_mut().zip(tri2idx).for_each(|(vtx, idx)| *vtx = idx2vtx[idx]);
    (tri2vtx, vtx2xyz)
}