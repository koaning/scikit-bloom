use pyo3::prelude::*;
use std::hash::{Hasher, Hash};
use std::collections::hash_map::DefaultHasher;


fn string_to_hash_mod(input: &str, size: &u32) -> u32 {
    let mut hasher = DefaultHasher::new();
    input.hash(&mut hasher);
    let hash = hasher.finish();
    hash as u32 % size
}


#[pyfunction]
fn hash_to_cols(input: &str, n_hashes: u32, n_buckets: u32) -> PyResult<Vec<u32>> {
    let result: Vec<u32> = input
    .to_lowercase()
    .split_whitespace()
    .flat_map(|ex| {
        (0..n_hashes).map(move |h| {
            let combined_str = format!("{}{}", ex, h);
            string_to_hash_mod(&combined_str, &n_buckets)
        })
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
