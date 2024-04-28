use std::fmt::Debug;

use crate::*;

use self::finite::FiniteRing;

pub trait Backend<F: FiniteRing>: Debug + Clone + Sized + Default {
    type V: Debug + Clone;
    type Error: std::error::Error;

    fn new(hasher_prefix: Vec<F>, rand_seed: &[F]) -> Result<Self, Self::Error>;
    fn hash_with_seed(&mut self, seed: &F, input: &[F]) -> Result<Self::V, Self::Error>;
    fn load_value(&mut self, a: &Self::V) -> Result<(), Self::Error>;
    fn expose_value(&mut self, a: &Self::V) -> Result<(), Self::Error>;
    fn constant(&mut self, a: &F) -> Result<Self::V, Self::Error>;
    fn add(&mut self, a: &Self::V, b: &Self::V) -> Result<Self::V, Self::Error>;
    fn mul(&mut self, a: &Self::V, b: &Self::V) -> Result<Self::V, Self::Error>;
    fn sub(&mut self, a: &Self::V, b: &Self::V) -> Result<Self::V, Self::Error>;
    fn neg(&mut self, a: &Self::V) -> Result<Self::V, Self::Error>;
    fn eq(&mut self, a: &Self::V, b: &Self::V) -> Result<Self::V, Self::Error>;

    fn lc(&mut self, coeffs: &[F], vars: &[Self::V]) -> Result<Self::V, Self::Error> {
        let mut sum = self.constant(&F::zero())?;
        for (coeff, var) in coeffs.into_iter().zip(vars.into_iter()) {
            let coeff = self.constant(coeff)?;
            let muled = self.mul(&coeff, &var)?;
            sum = self.add(&sum, &muled)?;
        }
        Ok(sum)
    }
}
