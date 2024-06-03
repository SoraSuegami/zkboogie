use self::native::{NativeBackend, NativeError, NativeHasher};
use crate::*;
// use ark_std::{end_timer, start_timer};
use crate::ark::*;
use ark_bn254::Fr;
use ark_relations::r1cs::SynthesisError;
use bincode;
use common::*;
use core::hash;
pub use encode::*;
use folding_schemes::{
    commitment::{kzg::KZG, pedersen::Pedersen},
    folding::nova::{
        decider_eth::{prepare_calldata, Decider as DeciderEth},
        Nova,
    },
    frontend::FCircuit,
    Decider, Error, FoldingScheme,
};
use itertools::Itertools;
use native::poseidon254_native::Poseidon254Native;
use rand::thread_rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, f64::consts::E};

#[derive(Debug, Clone)]
pub struct ZKBoogieEachVerifierCircuitParams {
    pub secpar: u8,
    pub e: u8,
    pub circuit: Circuit<F>,
    pub hasher_prefix: Vec<F>,
    // pub expected_output: Vec<F>,
}

#[derive(Debug, Clone)]
pub struct ZKBoogieEachVerifierCircuit {
    pub params: ZKBoogieEachVerifierCircuitParams,
}

impl FCircuit<Fr> for ZKBoogieEachVerifierCircuit {
    type Params = ZKBoogieEachVerifierCircuitParams;

    fn new(params: Self::Params) -> Result<Self, Error> {
        Ok(Self { params })
    }

    fn state_len(&self) -> usize {
        let num_repeat = compute_num_repeat(self.params.secpar) as usize;
        self.params.circuit.num_outputs() + num_repeat
    }

    fn external_inputs_len(&self) -> usize {
        5 + self.params.circuit.num_inputs()
            + self.params.circuit.num_outputs()
            + self.params.circuit.num_gates()
            + 2 * self.params.circuit.num_mul_gate()
    }

    fn step_native(
        &self,
        i: usize,
        z_i: Vec<Fr>,
        external_inputs: Vec<Fr>,
    ) -> Result<Vec<Fr>, Error> {
        let is_enable = external_inputs[0].clone();
        let each_proof =
            ZKBoogieEachProof::<F, NativeBackend<F, Poseidon254Native>>::from_field_vec(
                &external_inputs[1..]
                    .iter()
                    .map(|v| F256(v.clone()))
                    .collect_vec(),
                self.params.e,
                &self.params.circuit,
            );
        let mut next_z = z_i.clone();
        next_z[self.params.circuit.num_outputs() + i] = is_enable * each_proof.transcript_digest.0;
        Ok(next_z)
    }

    fn generate_step_constraints(
        &self,
        cs: ark_relations::r1cs::ConstraintSystemRef<Fr>,
        i: usize,
        z_i: Vec<ark_r1cs_std::fields::fp::FpVar<Fr>>,
        external_inputs: Vec<ark_r1cs_std::fields::fp::FpVar<Fr>>, // inputs that are not part of the state
    ) -> Result<Vec<ark_r1cs_std::fields::fp::FpVar<Fr>>, ark_relations::r1cs::SynthesisError> {
        let mut back = ArkBackend::<Poseidon254Ark>::new(self.params.hasher_prefix.clone())
            .expect("Failed to create backend");
        let is_enable = external_inputs[0].clone();
        let each_proof = ZKBoogieEachProof::<F, ArkBackend<Poseidon254Ark>>::from_field_vec(
            &external_inputs[1..],
            self.params.e,
            &self.params.circuit,
        );
        let expected_output = &z_i[0..self.params.circuit.num_outputs()];
        let is_valid = each_proof
            .verify_each(&mut back, &self.params.circuit, expected_output)
            .expect("Failed to verify each proof");
        {
            let one = back.one().unwrap();
            let subed = back.sub(&is_valid, &one).unwrap();
            let muled = back.mul(&is_enable, &subed).unwrap();
            back.force_zero(&muled).unwrap();
        }
        let transcript_digest = back.mul(&is_enable, &each_proof.transcript_digest).unwrap();
        let mut next_z = z_i.clone();
        next_z[self.params.circuit.num_outputs() + i] = transcript_digest;
        Ok(next_z)
    }
}
