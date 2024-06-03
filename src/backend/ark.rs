pub mod poseidon254_ark;
pub(crate) use poseidon254_ark::*;

use ark_ff::Field;
use ark_r1cs_std::eq::EqGadget;
use ark_r1cs_std::fields::FieldVar;
use ark_r1cs_std::R1CSVar;
use ark_r1cs_std::ToConstraintFieldGadget;
use ark_relations::lc;
use ark_relations::r1cs::LinearCombination;
use ark_relations::r1cs::Variable;
use std::collections::BTreeMap;
use std::marker::PhantomData;
// use arkworks_native_gadgets::poseidon::PoseidonError;
use crate::backend::*;
use crate::finite::*;
use crate::*;
use ark_bn254::Fr;
use ark_r1cs_std::{alloc::*, fields::fp::*};
use ark_relations::r1cs::{ConstraintSystem, ConstraintSystemRef, SynthesisError};
use itertools::Itertools;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ArkError {
    #[error(transparent)]
    Synthesis(#[from] SynthesisError),
    // Poseidon(PoseidonError),
}

pub type F = F256<Fr>;

pub trait ArkHasher: Clone + Default {
    fn new(cs: &mut ConstraintSystemRef<Fr>) -> Result<Self, ArkError>;
    fn hash(&self, inputs: &[FpVar<Fr>]) -> Result<Vec<FpVar<Fr>>, ArkError>;
}

#[derive(Clone)]
pub struct ArkBackend<H: ArkHasher> {
    pub hasher_prefix: Vec<F>,
    // pub hasher: H,
    pub cs: ConstraintSystemRef<Fr>,
    pub mode: AllocationMode,
    pub exposed: Vec<FpVar<Fr>>, // pub witnesses: BTreeMap<Variable, F>,
    // pub instances: BTreeMap<Varia>,
    pub hasher: H,
}

impl<H: ArkHasher> Backend<F> for ArkBackend<H> {
    type V = FpVar<Fr>;
    type Error = ArkError;

    fn new(hasher_prefix: Vec<F>) -> Result<Self, Self::Error> {
        // let hasher = NativeHasher::new();
        let mut cs = ConstraintSystem::<Fr>::new_ref();
        let hasher = H::new(&mut cs)?;
        Ok(ArkBackend {
            hasher_prefix,
            mode: AllocationMode::Witness,
            // hasher,
            cs,
            exposed: vec![],
            hasher,
        })
    }

    fn load_value(&mut self, a: &F) -> Result<Self::V, Self::Error> {
        let var = FpVar::new_variable(self.cs.clone(), || Ok(a.0), self.mode)?;
        // self.witnesses.insert(var, *a);
        Ok(var)
    }

    fn expose_value(&mut self, a: &Self::V) -> Result<(), Self::Error> {
        self.exposed.push(a.clone());
        Ok(())
    }

    fn constant(&mut self, a: &F) -> Result<Self::V, Self::Error> {
        let var = FpVar::new_constant(self.cs.clone(), a.0)?;
        Ok(var)
    }

    fn add(&mut self, a: &Self::V, b: &Self::V) -> Result<Self::V, Self::Error> {
        Ok(a + b)
    }

    fn mul(&mut self, a: &Self::V, b: &Self::V) -> Result<Self::V, Self::Error> {
        Ok(a * b)
    }

    fn sub(&mut self, a: &Self::V, b: &Self::V) -> Result<Self::V, Self::Error> {
        Ok(a - b)
    }

    fn neg(&mut self, a: &Self::V) -> Result<Self::V, Self::Error> {
        Ok(a.negate()?)
    }

    fn eq(&mut self, a: &Self::V, b: &Self::V) -> Result<Self::V, Self::Error> {
        let is_eq = a.is_eq(b)?;
        let fields = is_eq.to_constraint_field()?;
        assert_eq!(fields.len(), 1);
        Ok(fields[0].clone())
    }

    fn force_zero(&mut self, a: &Self::V) -> Result<(), Self::Error> {
        let zero = self.zero()?;
        a.enforce_equal(&zero)?;
        Ok(())
    }

    fn to_ternarys_le(&mut self, a: &Self::V) -> Vec<Self::V> {
        let value = a.value().unwrap();
        let mut ternarys_le = F256(value).to_ternarys_le();
        let num_max_ternarys = 162;
        ternarys_le.resize(num_max_ternarys, 0);
        let mut composed_value = self.constant(&F::zero()).unwrap();
        let mut coeff = self.constant(&F::one()).unwrap();
        let mut alloc_ternarys_le = vec![];
        for ternary in ternarys_le {
            let ternary_var = self.load_value(&F::from(ternary as u32)).unwrap();
            composed_value += &ternary_var * &coeff;
            coeff *= &self.constant(&F::from(3u32)).unwrap();
            alloc_ternarys_le.push(ternary_var);
        }
        a.enforce_equal(&composed_value).unwrap();
        alloc_ternarys_le
    }

    fn zero(&mut self) -> Result<Self::V, Self::Error> {
        self.constant(&F::zero())
    }

    fn one(&mut self) -> Result<Self::V, Self::Error> {
        self.constant(&F::one())
    }

    fn hash_to_one(&mut self, input: &[Self::V]) -> Result<Self::V, Self::Error> {
        let outputs = self.hash_to_multi(input)?;
        Ok(outputs[0].clone())
    }

    fn hash_to_multi(&mut self, input: &[Self::V]) -> Result<Vec<Self::V>, Self::Error> {
        let mut inputs = vec![];
        for v in self.hasher_prefix.clone().iter() {
            inputs.push(self.constant(v)?);
        }
        for v in input.iter() {
            inputs.push(v.clone());
        }
        let outputs = self.hasher.hash(&inputs)?;
        Ok(outputs)
    }
}
