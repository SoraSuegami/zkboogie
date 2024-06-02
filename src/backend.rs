use std::fmt::Debug;

use crate::*;
pub mod ark;
pub mod native;

use self::finite::FiniteRing;

pub trait Backend<F: FiniteRing>: Clone + Sized {
    type V: Debug + Clone;
    type Error: std::error::Error;

    fn new(hasher_prefix: Vec<F>) -> Result<Self, Self::Error>;
    fn load_value(&mut self, a: &F) -> Result<Self::V, Self::Error>;
    fn expose_value(&mut self, a: &Self::V) -> Result<(), Self::Error>;
    fn constant(&mut self, a: &F) -> Result<Self::V, Self::Error>;
    fn add(&mut self, a: &Self::V, b: &Self::V) -> Result<Self::V, Self::Error>;
    fn mul(&mut self, a: &Self::V, b: &Self::V) -> Result<Self::V, Self::Error>;
    fn sub(&mut self, a: &Self::V, b: &Self::V) -> Result<Self::V, Self::Error>;
    fn neg(&mut self, a: &Self::V) -> Result<Self::V, Self::Error>;
    fn eq(&mut self, a: &Self::V, b: &Self::V) -> Result<Self::V, Self::Error>;
    fn to_ternarys_le(&mut self, a: &Self::V) -> Vec<Self::V>;
    // fn mod_const(&mut self, a: &Self::V, n: u8) -> Result<Self::V, Self::Error>;

    fn zero(&mut self) -> Result<Self::V, Self::Error> {
        self.constant(&F::zero())
    }

    fn one(&mut self) -> Result<Self::V, Self::Error> {
        self.constant(&F::one())
    }

    fn lc(&mut self, coeffs: &[F], vars: &[Self::V]) -> Result<Self::V, Self::Error> {
        let mut sum = self.constant(&F::zero())?;
        for (coeff, var) in coeffs.into_iter().zip(vars.into_iter()) {
            let coeff = self.constant(coeff)?;
            let muled = self.mul(&coeff, &var)?;
            sum = self.add(&sum, &muled)?;
        }
        Ok(sum)
    }

    fn neq(&mut self, a: &Self::V, b: &Self::V) -> Result<Self::V, Self::Error> {
        let is_eq = self.eq(a, b)?;
        let one: <Self as Backend<F>>::V = self.one()?;
        self.sub(&one, &is_eq)
    }

    fn hash_to_one(&mut self, input: &[Self::V]) -> Result<Self::V, Self::Error>;
    fn hash_to_multi(&mut self, input: &[Self::V]) -> Result<Vec<Self::V>, Self::Error>;

    // fn hash_input_share(
    //     &mut self,
    //     rand_seed: &[Self::V],
    //     input_idx: u32,
    // ) -> Result<Self::V, Self::Error>;
    // fn hash_mul_pad(
    //     &mut self,
    //     rand_seed: &[Self::V],
    //     input: &[Self::V],
    // ) -> Result<Self::V, Self::Error>;
    // fn hash_commit(
    //     &mut self,
    //     rand_seed: &[Self::V],
    //     input: &[Self::V],
    // ) -> Result<Vec<Self::V>, Self::Error>;
    // fn hash_each_transcript(&mut self, input: &[Self::V]) -> Result<Vec<Self::V>, Self::Error>;
    // fn hash_challenge(&mut self, input: &[Self::V]) -> Result<Vec<Self::V>, Self::Error>;
}
