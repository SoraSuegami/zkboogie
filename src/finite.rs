use std::ops::{Add, Mul, Sub};

use ark_ff::{biginteger::*, Zero};
use ark_ff::{Field, FpParameters, PrimeField};
use arkworks_native_gadgets::ark_std::rand::Rng;
pub use num_bigint::*;
pub trait FiniteRing:
    Clone + Copy + PartialEq + Eq + std::fmt::Debug + From<u32> + Default + Send + Sync
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
    fn bytes_size() -> usize;
    fn to_ternarys_le(&self) -> Vec<u8>;
    fn to_bytes_le(&self) -> Vec<u8>;
    fn get_first_byte(&self) -> u8 {
        self.to_bytes_le()[0]
    }
    fn from_bytes_le(bytes: &[u8]) -> Self;
}

pub trait FiniteField: FiniteRing {
    fn inv(&self) -> Self;
    fn div(&self, other: &Self) -> Self;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct F256<F: PrimeField>(pub F);

impl<F: PrimeField> From<u32> for F256<F> {
    fn from(value: u32) -> Self {
        Self(F::from(value))
    }
}

impl<F: PrimeField> FiniteRing for F256<F> {
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

    fn bytes_size() -> usize {
        32
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

    fn to_bytes_le(&self) -> Vec<u8> {
        self.0.into_repr().to_bytes_le()
    }

    fn from_bytes_le(bytes: &[u8]) -> Self {
        let mut value = <F as PrimeField>::BigInt::default();
        let mut bytes = bytes.to_vec();
        for _ in 0..(32 - bytes.len()) {
            bytes.push(0);
        }
        value.read_le(&mut bytes.to_vec().as_slice()).unwrap();
        let value = F::from_repr(value).unwrap();
        Self(value)
    }
}

impl<F: PrimeField> Default for F256<F> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<F: PrimeField> F256<F> {
    pub fn new(value: F) -> Self {
        Self(value)
    }
}

// use plonky2::field::types::Field;
// use plonky2::hash::hash_types::RichField;
// use plonky2::plonk::circuit_data::{
//     CircuitConfig, CircuitData, ProverCircuitData, VerifierCircuitData,
// };
// use plonky2::plonk::config::{
//     AlgebraicHasher, GenericConfig, KeccakGoldilocksConfig, PoseidonGoldilocksConfig,
// };
// pub const D: usize = 2;
// pub type C = PoseidonGoldilocksConfig;
// // pub type C = KeccakGoldilocksConfig;
// // type F = <C as GenericConfig<D>>::F;
// pub struct F64(<C as GenericConfig<D>>::F);
// impl From<u32> for F64 {
//     fn from(value: u32) -> Self {
//         Self(<C as GenericConfig<D>>::F::from(value))
//     }
// }

// impl FiniteRing for F64 {
//     fn zero() -> Self {
//         let value = <C as GenericConfig<D>>::F::zero();
//         Self(value)
//     }

//     fn one() -> Self {
//         let value = <C as GenericConfig<D>>::F::one();
//         Self(value)
//     }

//     fn add(&self, other: &Self) -> Self {
//         let value = self.0 + other.0;
//         Self(value)
//     }

//     fn mul(&self, other: &Self) -> Self {
//         let value = self.0 * other.0;
//         Self(value)
//     }

//     fn sub(&self, other: &Self) -> Self {
//         let value = self.0 - other.0;
//         Self(value)
//     }

//     fn neg(&self) -> Self {
//         let value = -self.0;
//         Self(value)
//     }

//     fn rand<R: Rng>(rng: &mut R) -> Self {
//         let value = <C as GenericConfig<D>>::F::rand(rng);
//         Self(value)
//     }

//     fn modulo_bits_size() -> u32 {
//         <C as GenericConfig<D>>::F::Params::MODULUS_BITS
//     }

//     fn to_ternarys_le(&self) -> Vec<u8> {
//         let mut ternarys = Vec::new();
//         let mut value: BigUint = self.0.into_repr().try_into().unwrap();
//         let three = BigUint::from(3u8);
//         while value > BigUint::zero() {
//             let ternary = (&value % &three).to_bytes_le()[0];
//             ternarys.push(ternary);
//             value /= &three;
//         }
//         ternarys
//     }

//     fn get_first_byte(&self) -> u8 {
//         let bytes = self.0.into_repr().to_bytes_le();
//         bytes[0]
//     }
// }
