//! methods to load and save files with Wavefront Obj format

use std::fs::File;
use std::io::{BufRead, BufReader, Write};

pub struct WavefrontObj<T> {
    pub vtx2xyz: Vec<T>,
    pub vtx2uv: Vec<T>,
    pub vtx2nrm: Vec<T>,
    pub elem2idx: Vec<usize>,
    pub idx2vtx_xyz: Vec<usize>,
    pub idx2vtx_uv: Vec<usize>,
    pub idx2vtx_nrm: Vec<usize>,
    pub elem2group: Vec<usize>,
    pub group2name: Vec<String>,
    pub elem2mtl: Vec<usize>,
    pub mtl_file_name: String,
    pub mtl2name: Vec<String>,
}

impl<T: std::str::FromStr + std::fmt::Display> WavefrontObj<T> {
    pub fn new() -> Self {
        WavefrontObj::<T> {
            vtx2xyz: Vec::new(),
            vtx2uv: Vec::new(),
            vtx2nrm: Vec::new(),
            elem2idx: Vec::new(),
            idx2vtx_uv: Vec::new(),
            idx2vtx_nrm: Vec::new(),
            idx2vtx_xyz: Vec::new(),
            elem2group: Vec::new(),
            group2name: Vec::new(),
            mtl_file_name: "".to_string(),
            elem2mtl: Vec::new(),
            mtl2name: Vec::new(),
        }
    }
    /// load wavefront obj file into the class
    pub fn load<P: AsRef<std::path::Path>>(
        &mut self,
        filename: P) {
        let mut elem2vtx_xyz0: Vec<i32> = vec!();
        let mut elem2vtx_uv0: Vec<i32> = vec!();
        let mut elem2vtx_nrm0: Vec<i32> = vec!();
        self.elem2group.clear();
        self.elem2mtl.clear();
        self.elem2idx = vec!(0);
        let mut name2group = std::collections::BTreeMap::<String, usize>::new();
        let mut name2mtl = std::collections::BTreeMap::<String, usize>::new();
        name2group.insert("_default".to_string(), 0);
        name2mtl.insert("_default".to_string(), 0);
        let mut i_group = 0_usize;
        let mut i_mtl = 0_usize;
        let f = File::open(filename).expect("file not found");
        let reader = BufReader::new(f);
        for line in reader.lines() {
            let line = line.unwrap();
            if line.is_empty() { continue; }
            let char0 = line.chars().nth(0);
            if char0.is_none() { continue; }
            let char0 = char0.unwrap();
            let char1 = line.chars().nth(1);
            if char1.is_none() { continue; }
            let char1 = char1.unwrap();
            if char0 == '#' { continue; }
            if char0 == 'v' && char1 == ' ' {
                let v: Vec<&str> = line.split_whitespace().collect();
                let x = v[1].parse::<T>().ok().unwrap();
                let y = v[2].parse::<T>().ok().unwrap();
                let z = v[3].parse::<T>().ok().unwrap();
                self.vtx2xyz.push(x);
                self.vtx2xyz.push(y);
                self.vtx2xyz.push(z);
            }
            if char0 == 'g' && char1 == ' ' {
                let v: Vec<&str> = line.split_whitespace().collect();
                let name = v[1].to_string();
                match name2group.get(&name) {
                    None => {
                        i_group = name2group.len();
                        name2group.insert(name, i_group);
                    }
                    Some(&v) => {i_group = v;}
                };
            }
            if char0 == 'm' && char1 == 't' {
                let v: Vec<&str> = line.split_whitespace().collect();
                self.mtl_file_name = v[1].to_string();
            }
            if char0 == 'u' && char1 == 's' {
                let v: Vec<&str> = line.split_whitespace().collect();
                let name = v[1].to_string();
                match name2mtl.get(&name) {
                    None => {
                        i_mtl = name2mtl.len();
                        name2mtl.insert(name, i_mtl);
                    }
                    Some(&v) => {i_mtl = v;}
                };
            }
            if char0 == 'v' && char1 == 'n' {
                let v: Vec<&str> = line.split_whitespace().collect();
                let x = v[1].parse::<T>().ok().unwrap();
                let y = v[2].parse::<T>().ok().unwrap();
                let z = v[3].parse::<T>().ok().unwrap();
                self.vtx2nrm.push(x);
                self.vtx2nrm.push(y);
                self.vtx2nrm.push(z);
            }
            if char0 == 'v' && char1 == 't' {
                let v: Vec<&str> = line.split_whitespace().collect();
                let u = v[1].parse::<T>().ok().unwrap();
                let v = v[2].parse::<T>().ok().unwrap();
                self.vtx2uv.push(u);
                self.vtx2uv.push(v);
            }
            if char0 == 'f' && char1 == ' ' {
                let v: Vec<&str> = line.split_whitespace().collect();
                for i in 1..v.len() { // skip first 'f'
                    let (ipnt, itex, inrm) = parse_vertex(v[i]);
                    elem2vtx_xyz0.push(ipnt);
                    elem2vtx_uv0.push(itex);
                    elem2vtx_nrm0.push(inrm);
                }
                self.elem2idx.push(elem2vtx_xyz0.len());
                self.elem2group.push(i_group);
                self.elem2mtl.push(i_mtl);
            }
        } // end loop over text
        self.group2name = vec!("".to_string(); name2group.len());
        for (name, &i_group) in name2group.iter() {
            self.group2name[i_group] = name.clone();
        }
        self.mtl2name = vec!("".to_string(); name2mtl.len());
        for (name, &i_mtl) in name2mtl.iter() {
            self.mtl2name[i_mtl] = name.clone();
        }
        {  // fix veretx_xyz index
            let nvtx_xyz = self.vtx2xyz.len() / 3;
            self.idx2vtx_xyz = elem2vtx_xyz0.iter().map(
                |i| if *i >= 0 { *i as usize } else { (nvtx_xyz as i32 + *i) as usize }).collect();
        }
        {  // fix veretx_uv index
            let nvtx_uv = self.vtx2uv.len() / 3;
            self.idx2vtx_uv = elem2vtx_uv0.iter().map(
                |i| if *i >= 0 { *i as usize } else { (nvtx_uv as i32 + *i) as usize }).collect();
        }
        {  // fix veretx_nrm index
            let nvtx_nrm = self.vtx2nrm.len() / 3;
            self.idx2vtx_nrm = elem2vtx_nrm0.iter().map(
                |i| if *i >= 0 { *i as usize } else { (nvtx_nrm as i32 + *i) as usize }).collect();
        }
    }
}

pub fn load_tri_mesh<P: AsRef<std::path::Path>>(
    filepath: P,
    scale: Option<f32>) -> (Vec<usize>, Vec<f32>) {
    let mut obj = WavefrontObj::<f32>::new();
    obj.load(&filepath);
    let tri2vtx = obj.idx2vtx_xyz;
    let mut vtx2xyz = obj.vtx2xyz;
    if scale.is_some() {
        crate::transform::normalize_coords3(&mut vtx2xyz, scale.unwrap());
    }
    (tri2vtx, vtx2xyz)
}

pub fn save_tri_mesh_texture(
    filepath: &str,
    tri2vtx_xyz: &[usize],
    vtx2xyz: &[f32],
    tri2vtx_uv: &[usize],
    vtx2uv: &[f32]) {
    assert_eq!(tri2vtx_xyz.len(), tri2vtx_uv.len());
    let mut file = File::create(filepath).expect("file not found.");
    for i_vtx in 0..vtx2xyz.len() / 3 {
        writeln!(file, "v {} {} {}",
                 vtx2xyz[i_vtx * 3 + 0], vtx2xyz[i_vtx * 3 + 1], vtx2xyz[i_vtx * 3 + 2]).expect("fail");
    }
    for i_vtx in 0..vtx2uv.len() / 2 {
        writeln!(file, "vt {} {}",
                 vtx2uv[i_vtx * 2 + 0], vtx2uv[i_vtx * 2 + 1]).expect("fail");
    }
    for i_tri in 0..tri2vtx_xyz.len() / 3 {
        writeln!(file, "f {}/{} {}/{} {}/{}",
                 tri2vtx_xyz[i_tri * 3 + 0] + 1, tri2vtx_uv[i_tri * 3 + 0] + 1,
                 tri2vtx_xyz[i_tri * 3 + 1] + 1, tri2vtx_uv[i_tri * 3 + 1] + 1,
                 tri2vtx_xyz[i_tri * 3 + 2] + 1, tri2vtx_uv[i_tri * 3 + 2] + 1).expect("fail");
    }
}

pub fn save_tri_mesh<P, T>(
    filepath: P,
    tri2vtx: &[usize],
    vtx2xyz: &[T])
where P: AsRef<std::path::Path>,
      T: std::fmt::Display
{
    let mut file = File::create(filepath).expect("file not found.");
    for i_vtx in 0..vtx2xyz.len() / 3 {
        writeln!(file, "v {} {} {}",
                 vtx2xyz[i_vtx * 3 + 0], vtx2xyz[i_vtx * 3 + 1], vtx2xyz[i_vtx * 3 + 2]).expect("fail");
    }
    for i_tri in 0..tri2vtx.len() / 3 {
        writeln!(file, "f {} {} {}",
                 tri2vtx[i_tri * 3 + 0] + 1,
                 tri2vtx[i_tri * 3 + 1] + 1,
                 tri2vtx[i_tri * 3 + 2] + 1).expect("fail");
    }
}

// -------------------------
// below: private functions

fn parse_vertex(str_in: &str) -> (i32, i32, i32) {
    let snums: Vec<&str> = str_in.split('/').collect();
    let mut nums: [i32; 3] = [0, 0, 0];
    for i in 0..snums.len() {
        nums[i] = snums[i].parse::<i32>().unwrap_or(0);
    }
    return (nums[0] - 1, nums[1] - 1, nums[2] - 1);
}

#[test]
fn test_parse_vertex() {
    assert_eq!(parse_vertex("1/2/3"), (0, 1, 2));
    assert_eq!(parse_vertex("1//3"), (0, -1, 2));
    assert_eq!(parse_vertex("1/2"), (0, 1, -1));
    assert_eq!(parse_vertex("1"), (0, -1, -1));
}