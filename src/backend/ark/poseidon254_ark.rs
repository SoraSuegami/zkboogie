use ark::ArkError;
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
use ark_ff::Zero;
use ark_r1cs_std::{alloc::*, fields::fp::*};
use ark_relations::r1cs::{ConstraintSystem, ConstraintSystemRef, SynthesisError};
use arkworks_native_gadgets::poseidon::{sbox::PoseidonSbox, *};
use arkworks_r1cs_gadgets::poseidon::{FieldHasherGadget, PoseidonGadget, PoseidonParametersVar};
use arkworks_utils::{bytes_matrix_to_f, bytes_vec_to_f, poseidon_params::*, Curve};
use itertools::Itertools;
use thiserror::Error;

use super::ArkHasher;

#[derive(Clone, Default)]
pub struct Poseidon254Ark {
    inner: PoseidonGadget<Fr>,
}

impl ArkHasher for Poseidon254Ark {
    fn new(cs: &mut ConstraintSystemRef<Fr>) -> Result<Self, ArkError> {
        let pos_data = setup_poseidon_params(Curve::Bn254, 5, 5).unwrap();
        let mds_f = bytes_matrix_to_f(&pos_data.mds);
        let rounds_f = bytes_vec_to_f(&pos_data.rounds);
        let params = PoseidonParameters {
            mds_matrix: mds_f,
            round_keys: rounds_f,
            full_rounds: pos_data.full_rounds,
            partial_rounds: pos_data.partial_rounds,
            sbox: PoseidonSbox(pos_data.exp),
            width: pos_data.width,
        };
        let native = Poseidon::new(params);
        let inner = PoseidonGadget::from_native(cs, native)?;
        Ok(Self { inner })
    }

    fn hash(&self, inputs: &[FpVar<Fr>]) -> Result<Vec<FpVar<Fr>>, ArkError> {
        let mut last_hash = FpVar::<Fr>::Constant(Fr::zero());
        for chunk in inputs.chunks(3) {
            let mut input = vec![last_hash.clone()];
            input.extend_from_slice(chunk);
            input.resize(4, FpVar::<Fr>::Constant(Fr::zero()));
            last_hash = self
                .inner
                .hash(&input)
                .map_err(|err| ArkError::Synthesis(err))?;
        }
        Ok(vec![last_hash])
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::native::poseidon254_native::Poseidon254Native;
    use crate::native::NativeHasher;
    use ark_bn254::Fr;
    use ark_r1cs_std::alloc::AllocVar;
    use ark_r1cs_std::fields::fp::FpVar;
    use ark_relations::r1cs::ConstraintSystem;

    #[test]
    fn test_poseidon_ark() {
        let mut cs = ConstraintSystem::<Fr>::new_ref();
        let hasher = Poseidon254Ark::new(&mut cs).unwrap();
        let input: Vec<FpVar<Fr>> = (0..12)
            .map(|i| {
                let x = Fr::from(i as u128);
                FpVar::<Fr>::new_witness(ark_relations::ns!(cs, "input"), || Ok(x)).unwrap()
            })
            .collect();
        let output = hasher.hash(&input).unwrap();
        assert_eq!(output.len(), 1);

        let native_hasher = Poseidon254Native::new();
        let native_output = native_hasher
            .hash(
                &input
                    .iter()
                    .map(|x| F256(x.value().unwrap()))
                    .collect::<Vec<_>>()[..],
            )
            .unwrap();
        assert_eq!(F256(output[0].value().unwrap()), native_output[0]);
    }
}
