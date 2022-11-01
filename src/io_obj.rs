use std::fs::File;
use std::io::{BufRead, BufReader};

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

pub struct WavefrontObj<T> {
    pub vtx2xyz: Vec<T>,
    pub vtx2uv: Vec<T>,
    pub vtx2nrm: Vec<T>,
    pub elem2vtx_idx: Vec<usize>,
    pub elem2vtx_xyz: Vec<usize>,
    pub elem2vtx_uv: Vec<usize>,
    pub elem2vtx_nrm: Vec<usize>,
}

impl<T: std::str::FromStr+ std::fmt::Display> WavefrontObj<T> {
    pub fn new() -> Self {
        WavefrontObj::<T> {
            vtx2xyz: Vec::new(),
            vtx2uv: Vec::new(),
            vtx2nrm: Vec::new(),
            elem2vtx_xyz: Vec::new(),
            elem2vtx_uv: Vec::new(),
            elem2vtx_nrm: Vec::new(),
            elem2vtx_idx: Vec::new(),
        }
    }
    pub fn load(&mut self, filename: &str) {
        let mut elem2vtx_xyz0: Vec<i32> = vec!();
        let mut elem2vtx_uv0: Vec<i32> = vec!();
        let mut elem2vtx_nrm0: Vec<i32> = vec!();
        let f = File::open(filename).expect("file not found");
        self.elem2vtx_idx.push(0);
        let reader = BufReader::new(f);
        for line in reader.lines() {
            let line = line.unwrap();
            if line.is_empty() { continue; }
            let char0 = line.chars().nth(0);
            if char0.is_none(){ continue; }
            let char0 = char0.unwrap();
            let char1 = line.chars().nth(1);
            if char1.is_none(){ continue; }
            let char1 = char1.unwrap();
            if char0 == 'v' && char1 == ' ' {
                let v: Vec<&str> = line.split_whitespace().collect();
                let x = v[1].parse::<T>().ok().unwrap();
                let y = v[2].parse::<T>().ok().unwrap();
                let z = v[3].parse::<T>().ok().unwrap();
                self.vtx2xyz.push(x);
                self.vtx2xyz.push(y);
                self.vtx2xyz.push(z);
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
                self.elem2vtx_idx.push(elem2vtx_xyz0.len());
            }
        } // end loop over text
        {  // fix veretx_xyz index
            let nvtx_xyz = self.vtx2xyz.len() / 3;
            self.elem2vtx_xyz = elem2vtx_xyz0.iter().map(
                |i| if *i >= 0 { *i as usize } else { (nvtx_xyz as i32 + *i) as usize }).collect();
        }
        {  // fix veretx_uv index
            let nvtx_uv = self.vtx2uv.len() / 3;
            self.elem2vtx_uv = elem2vtx_uv0.iter().map(
                |i| if *i >= 0 { *i as usize } else { (nvtx_uv as i32 + *i) as usize }).collect();
        }
        {  // fix veretx_nrm index
            let nvtx_nrm = self.vtx2nrm.len() / 3;
            self.elem2vtx_nrm = elem2vtx_nrm0.iter().map(
                |i| if *i >= 0 { *i as usize } else { (nvtx_nrm as i32 + *i) as usize }).collect();
        }
    }
}