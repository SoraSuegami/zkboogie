use crate::*;
use ark_ff::PrimeField;
use arkworks_native_gadgets::poseidon::*;

#[derive(Debug, Clone)]
pub struct Poseidon254Native<F: PrimeField> {
    inner: Poseidon<F>,
}
