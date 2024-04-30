use std::ops::{Add, Mul, Sub};

use ark_ff::biginteger::*;
use ark_ff::{Field, PrimeField};
use arkworks_native_gadgets::ark_std::rand::Rng;
// pub use num_bigint::*;
pub trait FiniteRing:
    Clone + Copy + PartialEq + Eq + std::fmt::Debug + From<u32> + Default
{
    // fn modulo() -> BigInt;
    fn zero() -> Self;
    fn one() -> Self;
    // fn value(&self) -> &BigInt;
    fn add(&self, other: &Self) -> Self;
    fn mul(&self, other: &Self) -> Self;
    fn sub(&self, other: &Self) -> Self;
    fn neg(&self) -> Self;
    fn rand<R: Rng>(rng: &mut R) -> Self;
    fn modulo_bits_size() -> u32;
}

pub trait FiniteField: FiniteRing {
    fn inv(&self) -> Self;
    fn div(&self, other: &Self) -> Self;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fp<F: Field>(pub F);

impl<F: Field> From<u32> for Fp<F> {
    fn from(value: u32) -> Self {
        Fp(F::from(value))
    }
}

impl<F: Field> FiniteRing for Fp<F> {
    fn zero() -> Self {
        let value = F::zero();
        Self(value)
    }

    fn one() -> Self {
        let value = F::one();
        Self(value)
    }

    fn add(&self, other: &Self) -> Self {
        let value = self.0 + other.0;
        Self(value)
    }

    fn mul(&self, other: &Self) -> Self {
        let value = self.0 * other.0;
        Self(value)
    }

    fn sub(&self, other: &Self) -> Self {
        let value = self.0 - other.0;
        Self(value)
    }

    fn neg(&self) -> Self {
        let value = -self.0;
        Self(value)
    }

    fn rand<R: Rng>(rng: &mut R) -> Self {
        let value = F::rand(rng);
        Self(value)
    }

    fn modulo_bits_size() -> u32 {
        F::BasePrimeField::MODULUS_BIT_SIZE
    }
}

impl<F: Field> Default for Fp<F> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<F: Field> Fp<F> {
    pub fn new(value: F) -> Self {
        Fp(value)
    }
}

// macro_rules! finite_field_impl {
//     ($struct_name:ident, $modulo:expr) => {
//         pub struct $struct_name {
//             value: num_bigint::BigUint,
//             modulo: num_bigint::BigUint,
//         }
//     };
// }
