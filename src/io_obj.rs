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
    pub vtx_xyz: Vec<T>,
    pub vtx_uv: Vec<T>,
    pub elem_vtx_index: Vec<usize>,
    pub elem_vtx_xyz: Vec<i32>,
    pub elem_vtx_uv: Vec<i32>,
    pub elem_vtx_nrm: Vec<i32>,
}

impl<T: std::str::FromStr+ std::fmt::Display> WavefrontObj<T> {
    pub fn new() -> Self {
        WavefrontObj::<T> {
            vtx_xyz: Vec::new(),
            vtx_uv: Vec::new(),
            elem_vtx_xyz: Vec::new(),
            elem_vtx_uv: Vec::new(),
            elem_vtx_nrm: Vec::new(),
            elem_vtx_index: Vec::new(),
        }
    }
    pub fn load(&mut self, filename: &str) {
        let f = File::open(filename).expect("file not found");
        self.elem_vtx_index.push(0);
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
                self.vtx_xyz.push(x);
                self.vtx_xyz.push(y);
                self.vtx_xyz.push(z);
            }
            if char0 == 'v' && char1 == 't' {
                let v: Vec<&str> = line.split_whitespace().collect();
                let u = v[1].parse::<T>().ok().unwrap();
                let v = v[2].parse::<T>().ok().unwrap();
                self.vtx_uv.push(u);
                self.vtx_uv.push(v);
            }
            if char0 == 'f' && char1 == ' ' {
                let v: Vec<&str> = line.split_whitespace().collect();
                for i in 1..v.len() { // skip first 'f'
                    let (ipnt, itex, inrm) = parse_vertex(v[i]);
                    self.elem_vtx_xyz.push(ipnt);
                    self.elem_vtx_uv.push(itex);
                    self.elem_vtx_nrm.push(inrm);
                }
                self.elem_vtx_index.push(self.elem_vtx_xyz.len());
            }
        }
    }
}