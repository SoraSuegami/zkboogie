use self::native::{NativeBackend, NativeError, NativeHasher};
use crate::*;
// use ark_std::{end_timer, start_timer};
use bincode;
use core::hash;
pub use encode::*;
use itertools::Itertools;
use rand::thread_rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, f64::consts::E};

#[derive(Debug, Clone)]
pub struct ZKBoogieEachProver<F: FiniteRing, H: NativeHasher<F>> {
    view0: PlayerState<F, NativeBackend<F, H>>,
    view1: PlayerState<F, NativeBackend<F, H>>,
    view2: PlayerState<F, NativeBackend<F, H>>,
}

impl<F: FiniteRing, H: NativeHasher<F>> ZKBoogieEachProver<F, H> {
    pub fn new() -> Self {
        Self {
            view0: PlayerState::new(0),
            view1: PlayerState::new(1),
            view2: PlayerState::new(2),
        }
    }

    pub(crate) fn commit(
        &mut self,
        back: &mut NativeBackend<F, H>,
        circuit: &Circuit<F>,
        input: &[F],
    ) -> Result<Vec<F>, NativeError> {
        self.commit_input(back, input)?;
        let s1 = back.hash_to_one(&[
            self.view0.input_commit.unwrap(),
            self.view1.input_commit.unwrap(),
            self.view2.input_commit.unwrap(),
        ])?;
        self.commit_views(back, circuit, &s1)?;
        let s2 = back.hash_to_one(&[
            s1,
            self.view0.view_commit.unwrap(),
            self.view1.view_commit.unwrap(),
            self.view2.view_commit.unwrap(),
        ])?;
        self.commit_rand_pads(back, &s2)?;
        let transcript_digest = back.hash_to_multi(&[
            s2,
            self.view0.rand_commit.unwrap(),
            self.view1.rand_commit.unwrap(),
            self.view2.rand_commit.unwrap(),
        ])?;
        Ok(transcript_digest)
    }

    pub(crate) fn response(
        self,
        circuit: &Circuit<F>,
        e: u8,
        transcript_digest: F,
    ) -> Result<ZKBoogieEachProof<F, NativeBackend<F, H>>, NativeError> {
        let (e_view, e1_view, e2_view) = match e {
            0 => (self.view0, self.view1, self.view2),
            1 => (self.view1, self.view2, self.view0),
            2 => (self.view2, self.view0, self.view1),
            _ => panic!("Invalid challenge"),
        };
        let e_input_shares = (0..circuit.num_inputs())
            .map(|input_idx| e_view.wire_shares[&GateId(input_idx as u32)])
            .collect_vec();
        let e2_input_commit = e2_view.input_commit.unwrap();
        let e1_wire_shares = e1_view.wire_shares;
        let e2_view_commit = e2_view.view_commit.unwrap();
        let e_rand = e_view.rand_pads;
        let e1_rand = e1_view.rand_pads;
        let e2_rand_commit = e2_view.rand_commit.unwrap();
        let e2_output_shares = circuit
            .output_ids
            .iter()
            .map(|id| e2_view.wire_shares[id])
            .collect_vec();
        let proof = ZKBoogieEachProof {
            e,
            e_input_shares,
            e2_input_commit,
            e1_wire_shares,
            e2_view_commit,
            e_rand,
            e1_rand,
            e2_rand_commit,
            e2_output_shares,
            transcript_digest,
        };
        Ok(proof)
    }

    fn commit_input(
        &mut self,
        back: &mut NativeBackend<F, H>,
        input: &[F],
    ) -> Result<(), NativeError> {
        for i in 0..input.len() {
            let input_idx = i as u32;
            self.view0
                .gen_input_share(back, input_idx, &input[i], &self.view1, &self.view2)?;
            self.view1
                .gen_input_share(back, input_idx, &input[i], &self.view2, &self.view0)?;
            self.view2
                .gen_input_share(back, input_idx, &input[i], &self.view0, &self.view1)?;
        }
        self.view0.commit_input_shares(back)?;
        self.view1.commit_input_shares(back)?;
        self.view2.commit_input_shares(back)?;
        Ok(())
    }

    fn commit_views(
        &mut self,
        back: &mut NativeBackend<F, H>,
        circuit: &Circuit<F>,
        rand: &F,
    ) -> Result<(), NativeError> {
        let gates = circuit.enumerate_gates();
        for gate in gates[circuit.num_inputs()..].iter() {
            match gate.gate_type {
                GateType::Input(_) => {
                    panic!("Expected non-input gate");
                }
                GateType::ConstAdd(_) => {
                    self.view0.run_const_add(back, gate.clone())?;
                    self.view1.run_const_add(back, gate.clone())?;
                    self.view2.run_const_add(back, gate.clone())?;
                }
                GateType::ConstMul(_) => {
                    self.view0.run_const_mul(back, gate.clone())?;
                    self.view1.run_const_mul(back, gate.clone())?;
                    self.view2.run_const_mul(back, gate.clone())?;
                }
                GateType::Add => {
                    self.view0.run_add(back, gate.clone())?;
                    self.view1.run_add(back, gate.clone())?;
                    self.view2.run_add(back, gate.clone())?;
                }
                GateType::Mul => {
                    self.view0.run_mul(back, gate.clone(), &mut self.view1)?;
                    self.view1.run_mul(back, gate.clone(), &mut self.view2)?;
                    self.view2.run_mul(back, gate.clone(), &mut self.view0)?;
                }
                GateType::Neg => {
                    self.view0.run_neg(back, gate.clone())?;
                    self.view1.run_neg(back, gate.clone())?;
                    self.view2.run_neg(back, gate.clone())?;
                }
                GateType::Const(_) => {
                    self.view0.run_const(back, gate.clone())?;
                    self.view1.run_const(back, gate.clone())?;
                    self.view2.run_const(back, gate.clone())?;
                }
            }
        }
        self.view0.commit_view(back, rand)?;
        self.view1.commit_view(back, rand)?;
        self.view2.commit_view(back, rand)?;
        Ok(())
    }

    fn commit_rand_pads(
        &mut self,
        back: &mut NativeBackend<F, H>,
        rand: &F,
    ) -> Result<(), NativeError> {
        self.view0.commit_rand_pads(back, rand)?;
        self.view1.commit_rand_pads(back, rand)?;
        self.view2.commit_rand_pads(back, rand)?;
        Ok(())
    }
}

pub fn zkboogie_prove<F: FiniteRing, H: NativeHasher<F>>(
    secpar: u8,
    hasher_prefix: Vec<F>,
    circuit: &Circuit<F>,
    input: &[F],
) -> Result<ZKBoogieProof<F, NativeBackend<F, H>>, NativeError> {
    let num_repeat = compute_num_repeat(secpar);
    let commit_phases = (0..num_repeat)
        .into_par_iter()
        .map(|_| {
            let mut prover = ZKBoogieEachProver::new();
            let mut back = NativeBackend::new(hasher_prefix.clone()).unwrap();
            // let timer = start_timer!(|| "commit");
            let transcript_digest = prover.commit(&mut back, circuit, input).unwrap();
            // end_timer!(timer);
            (prover, transcript_digest)
        })
        .collect::<Vec<_>>();

    let challenge_inputs = commit_phases
        .iter()
        .flat_map(|(_, digest)| digest.clone())
        .collect_vec();
    let mut back = NativeBackend::<F, H>::new(hasher_prefix.clone()).unwrap();
    // let challenge_timer = start_timer!(|| "challenge");
    let challenge = back.hash_to_multi(&challenge_inputs)?;
    // end_timer!(challenge_timer);
    let challenge_ternarys = challenge
        .into_iter()
        .flat_map(|c| back.to_ternarys_le(&c))
        .collect_vec();
    let each_proof = commit_phases
        .into_par_iter()
        .enumerate()
        .map(|(idx, (mut prover, transcript_digest))| {
            let e: u8 = challenge_ternarys[idx].get_first_byte();
            // let timer = start_timer!(|| "response");
            let response = prover.response(circuit, e, challenge_inputs[idx]).unwrap();
            // end_timer!(timer);
            response
        })
        .collect::<Vec<_>>();
    Ok(ZKBoogieProof { each_proof })
}
