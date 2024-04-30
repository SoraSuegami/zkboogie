use std::ops::{Add, Mul, Sub};

use ark_ff::{biginteger::*, Zero};
use ark_ff::{Field, FpParameters, PrimeField};
use arkworks_native_gadgets::ark_std::rand::Rng;
pub use num_bigint::*;
pub trait FiniteRing:
    Clone + Copy + PartialEq + Eq + std::fmt::Debug + From<u32> + Default
{
    // fn modulo() -> BigInt;
    fn zero() -> Self;
    fn one() -> Self;
    fn add(&self, other: &Self) -> Self;
    fn mul(&self, other: &Self) -> Self;
    fn sub(&self, other: &Self) -> Self;
    fn neg(&self) -> Self;
    fn rand<R: Rng>(rng: &mut R) -> Self;
    fn modulo_bits_size() -> u32;
    fn to_ternarys_le(&self) -> Vec<u8>;
    fn get_first_byte(&self) -> u8;
}

pub trait FiniteField: FiniteRing {
    fn inv(&self) -> Self;
    fn div(&self, other: &Self) -> Self;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fp<F: PrimeField>(pub F);

impl<F: PrimeField> From<u32> for Fp<F> {
    fn from(value: u32) -> Self {
        Fp(F::from(value))
    }
}

impl<F: PrimeField> FiniteRing for Fp<F> {
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
        F::Params::MODULUS_BITS
    }

    fn to_ternarys_le(&self) -> Vec<u8> {
        let mut ternarys = Vec::new();
        let mut value: BigUint = self.0.into_repr().try_into().unwrap();
        let three = BigUint::from(3u8);
        while value > BigUint::zero() {
            let ternary = (&value % &three).to_bytes_le()[0];
            ternarys.push(ternary);
            value /= &three;
        }
        ternarys
    }

    fn get_first_byte(&self) -> u8 {
        let bytes = self.0.into_repr().to_bytes_le();
        bytes[0]
    }
}

impl<F: PrimeField> Default for Fp<F> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<F: PrimeField> Fp<F> {
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
