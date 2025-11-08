include!(concat!(env!("OUT_DIR"), "/fatbins.rs"));

/// 例: "add" → FATBINバイト列
pub fn get(name: &str) -> Option<&'static [u8]> {
    all().get(name).copied()
}

/// 登録されているfatbin名一覧
pub fn list() -> Vec<&'static str> {
    let mut v: Vec<_> = all().keys().copied().collect();
    v.sort();
    v
}
