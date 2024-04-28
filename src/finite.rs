use std::ops::{Add, Mul, Sub};

pub use num_bigint::*;

pub trait FiniteRing: Clone + Copy + PartialEq + Eq + std::fmt::Debug + From<u32> {
    fn modulo() -> BigUint;
    fn zero() -> Self;
    fn one() -> Self;
    fn value(&self) -> &BigUint;
    fn add(&self, other: &Self) -> Self;
    fn mul(&self, other: &Self) -> Self;
    fn sub(&self, other: &Self) -> Self;
    fn neg(&self) -> Self;
}

pub trait FiniteField: FiniteRing {
    fn inv(&self) -> Self;
    fn div(&self, other: &Self) -> Self;
}
