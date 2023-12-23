use pyo3::prelude::*;
use std::hash::{Hasher, Hash};
use fasthash::MurmurHasher;


fn hash_with_reuse(t: &str, n_buckets: u32, n_hashes: u32) -> Vec<u32> {
    let mut hasher: MurmurHasher = Default::default();
    (0..n_hashes).map(|_s| {
        t.hash(&mut hasher);
        let value = hasher.finish();
        value as u32 % n_buckets
    }).collect()
}


#[pyfunction]
fn hash_to_cols(input: &str, n_hashes: u32, n_buckets: u32) -> PyResult<Vec<u32>> {
    let result: Vec<u32> = input
    .to_lowercase()
    .split_whitespace()
    .flat_map(|ex| {
        hash_with_reuse(ex, n_buckets, n_hashes)
    })
    .collect();

    Ok(result)
}


/// A Python module implemented in Rust.
#[pymodule]
fn skbloom(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hash_to_cols, m)?)?;
    Ok(())
}
