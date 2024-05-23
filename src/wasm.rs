use crate::finite::F256;
use crate::zkboogie_prove;
use crate::*;
use ark_bn254;
use ark_ff::PrimeField;
use ark_std::*;
use wasm_bindgen::prelude::*;
pub use wasm_bindgen_rayon::init_thread_pool;
type F = F256<ark_bn254::Fr>;
use crate::native::poseidon254_native::Poseidon254Native;
use crate::native::NativeBackend;
use itertools::Itertools;

#[wasm_bindgen]
pub fn zkboogie_prove_wasm(
    secpar: u8,
    hasher_prefix: Vec<u32>,
    circuit: Vec<u8>,
    input: Vec<String>,
) -> Vec<u8> {
    let circuit = Circuit::from_bytes_le(&circuit);
    let input = input
        .into_iter()
        .map(|hex| <F as FiniteRing>::from_str(&hex))
        .collect_vec();
    let hasher_prefix = hasher_prefix.into_iter().map(F::from).collect_vec();
    let proof = zkboogie_prove::<F, Poseidon254Native>(secpar, hasher_prefix, &circuit, &input)
        .expect("prove failed");
    let proof_bytes = proof.to_bytes_le();
    proof_bytes
}

#[wasm_bindgen]
pub fn zkboogie_verify_wasm(
    secpar: u8,
    hasher_prefix: Vec<u32>,
    circuit: Vec<u8>,
    expected_output: Vec<String>,
    proof: Vec<u8>,
) -> bool {
    let circuit = Circuit::from_bytes_le(&circuit);
    let hasher_prefix = hasher_prefix.into_iter().map(F::from).collect_vec();
    let expected_output = expected_output
        .into_iter()
        .map(|hex| <F as FiniteRing>::from_str(&hex))
        .collect_vec();
    let proof = ZKBoogieProof::<F, NativeBackend<F, Poseidon254Native>>::from_bytes_le(&proof);
    proof
        .verify_whole(secpar, hasher_prefix, &circuit, &expected_output)
        .expect("verify failed")
}
