mod encode;
use self::native::{NativeBackend, NativeError, NativeHasher};
use crate::*;
use ark_std::{end_timer, start_timer};
use bincode;
use core::hash;
pub use encode::*;
use itertools::Itertools;
use rand::thread_rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, f64::consts::E};

#[derive(Debug, Clone)]
pub struct PlayerState<F: FiniteRing, B: Backend<F>> {
    pub player_idx: u8,
    pub num_input: usize,
    // pub rand_seed: Vec<B::V>,
    pub wire_shares: BTreeMap<GateId, B::V>,
    pub rand_pads: BTreeMap<GateId, B::V>,
    pub input_commit: Option<B::V>,
    pub view_commit: Option<B::V>,
    pub rand_commit: Option<B::V>,
}

impl<F: FiniteRing, B: Backend<F>> PlayerState<F, B> {
    pub fn new(player_idx: u8) -> Self {
        Self {
            player_idx,
            num_input: 0,
            // rand_seed,
            wire_shares: BTreeMap::new(),
            rand_pads: BTreeMap::new(),
            input_commit: None,
            view_commit: None,
            rand_commit: None,
        }
    }

    // pub fn rand_seed(&self) -> &[B::V] {
    //     &self.rand_seed
    // }

    pub fn wire_share(&self, gate_id: GateId) -> Option<&B::V> {
        self.wire_shares.get(&gate_id)
    }

    pub fn rand_pad(&self, gate_id: GateId) -> Option<&B::V> {
        self.rand_pads.get(&gate_id)
    }

    pub fn gen_input_share(
        &mut self,
        back: &mut B,
        input_idx: u32,
        input: &B::V,
        e1_view: &Self,
        e2_view: &Self,
    ) -> Result<(), B::Error> {
        // println!("gen_input_share");
        // let timer = start_timer!(|| "gen_input_share");
        let gate_id = GateId::new(input_idx);
        if self.player_idx < 2 {
            // let value = hash_input_share(back, &self.rand_seed, input_idx)?;
            let value = back.load_value(&F::rand(&mut thread_rng()))?;
            self.wire_shares.insert(gate_id, value);
            // end_timer!(timer);
            self.num_input += 1;
            Ok(())
        } else {
            // if let GateType::Input(input_idx) = gate.gate_type {
            let value_0 = &e1_view.wire_shares[&gate_id];
            let value_1 = &e2_view.wire_shares[&gate_id];
            let value_2 = {
                let sub0 = back.sub(input, value_0)?;
                back.sub(&sub0, value_1)?
            };
            self.wire_shares.insert(gate_id, value_2);
            // end_timer!(timer);
            self.num_input += 1;
            Ok(())
            // } else {
            //     panic!("Expected input gate");
            // }
        }
    }

    pub fn add_input_share(&mut self, gate: Gate<F>, share: B::V) {
        self.wire_shares.insert(gate.gate_id, share);
        self.num_input += 1;
    }

    pub fn commit_input_shares(&mut self, back: &mut B) -> Result<B::V, B::Error> {
        let mut input_shares = vec![];
        for i in 0..self.num_input {
            let gate_id = GateId::new(i as u32);
            let share = self.wire_shares[&gate_id].clone();
            input_shares.push(share);
        }
        let commit = back.hash_to_one(&input_shares)?;
        self.input_commit = Some(commit.clone());
        Ok(commit)
    }

    pub fn run_const_add(&mut self, back: &mut B, gate: Gate<F>) -> Result<(), B::Error> {
        // println!("run_const_add");
        // let timer = start_timer!(|| "run_const_add");
        let constant = if let GateType::ConstAdd(c) = gate.gate_type {
            c
        } else {
            panic!("Expected constant gate")
        };
        let value = &self.wire_shares[&gate.inputs[0]];
        let new_value = if self.player_idx == 0 {
            let constant = back.constant(&constant)?;
            back.add(&value, &constant)?
        } else {
            value.clone()
        };
        self.wire_shares.insert(gate.gate_id, new_value);
        // end_timer!(timer);
        Ok(())
    }

    pub fn run_const_mul(&mut self, back: &mut B, gate: Gate<F>) -> Result<(), B::Error> {
        // println!("run_const_mul");
        // let timer = start_timer!(|| "run_const_mul");
        let constant = if let GateType::ConstMul(c) = gate.gate_type {
            c
        } else {
            panic!("Expected constant gate")
        };
        let value = &self.wire_shares[&gate.inputs[0]];
        let constant = back.constant(&constant)?;
        let new_value = back.mul(&value, &constant)?;
        self.wire_shares.insert(gate.gate_id, new_value);
        // end_timer!(timer);
        Ok(())
    }

    pub fn run_add(&mut self, back: &mut B, gate: Gate<F>) -> Result<(), B::Error> {
        // println!("run_add");
        // let timer = start_timer!(|| "run_add");
        let left = &self.wire_shares[&gate.inputs[0]];
        let right = &self.wire_shares[&gate.inputs[1]];
        let new_value = back.add(&left, &right)?;
        self.wire_shares.insert(gate.gate_id, new_value);
        // end_timer!(timer);
        Ok(())
    }

    pub fn run_mul(
        &mut self,
        back: &mut B,
        gate: Gate<F>,
        other_view: &mut Self,
    ) -> Result<(), B::Error> {
        // println!("run_mul");
        // let timer = start_timer!(|| "run_mul");
        let self_left = &self.wire_shares[&gate.inputs[0]];
        let self_right = &self.wire_shares[&gate.inputs[1]];
        let other_left = &other_view.wire_shares[&gate.inputs[0]];
        let other_right = &other_view.wire_shares[&gate.inputs[1]];
        let new_value = {
            let term1 = back.mul(&self_left, &self_right)?;
            let term2 = back.mul(&other_left, &self_right)?;
            let term3 = back.mul(&self_left, &other_right)?;
            // let gate_id = back.constant(&F::from(gate.gate_id.0))?;
            // let term4_timer = start_timer!(|| "poseidon term4");
            // let term4 = back.hash_mul_pad(&self.rand_seed, &[gate_id.clone()])?;
            // end_timer!(term4_timer);
            let term4 = self.get_rand_term(back, &gate.gate_id)?;
            // let term5_timer = start_timer!(|| "poseidon term5");
            // let term5 = back.hash_mul_pad(&other_view.rand_seed, &[gate_id])?;
            // end_timer!(term5_timer);
            let term5 = other_view.get_rand_term(back, &gate.gate_id)?;
            let add1 = back.add(&term1, &term2)?;
            let add2 = back.add(&add1, &term3)?;
            let add3 = back.add(&add2, &term4)?;
            back.sub(&add3, &term5)?
        };
        self.wire_shares.insert(gate.gate_id, new_value);
        // end_timer!(timer);
        Ok(())
    }

    pub fn run_neg(&mut self, back: &mut B, gate: Gate<F>) -> Result<(), B::Error> {
        let timer = start_timer!(|| "run_neg");
        let value = &self.wire_shares[&gate.inputs[0]];
        let new_value = back.neg(&value)?;
        self.wire_shares.insert(gate.gate_id, new_value);
        end_timer!(timer);
        Ok(())
    }

    pub fn run_const(&mut self, back: &mut B, gate: Gate<F>) -> Result<(), B::Error> {
        // let timer = start_timer!(|| "run_const");
        let constant = if let GateType::Const(c) = gate.gate_type {
            c
        } else {
            panic!("Expected constant gate")
        };
        if self.player_idx == 0 {
            self.wire_shares
                .insert(gate.gate_id, back.constant(&constant)?);
        } else {
            self.wire_shares.insert(gate.gate_id, back.zero()?);
        }
        // end_timer!(timer);
        Ok(())
    }

    pub fn commit_view(&mut self, back: &mut B, rand: &B::V) -> Result<B::V, B::Error> {
        let view = self.wire_shares.values().cloned().collect_vec();
        let commit = rlc(back, rand, &view)?;
        self.view_commit = Some(commit.clone());
        Ok(commit)
    }

    pub fn commit_rand_pads(&mut self, back: &mut B, rand: &B::V) -> Result<B::V, B::Error> {
        let rand_pads = self.rand_pads.values().cloned().collect_vec();
        let commit = rlc(back, rand, &rand_pads)?;
        self.rand_commit = Some(commit.clone());
        Ok(commit)
    }

    fn get_rand_term(&mut self, back: &mut B, gate_id: &GateId) -> Result<B::V, B::Error> {
        let term = self.rand_pads.get(gate_id).cloned();
        if let Some(term) = term {
            Ok(term)
        } else {
            let term = back.load_value(&F::rand(&mut thread_rng()))?;
            self.rand_pads.insert(*gate_id, term.clone());
            Ok(term)
        }
    }
}

#[derive(Debug, Clone)]
pub struct ZKBoogieEachProof<F: FiniteRing, B: Backend<F>> {
    pub e: u8,
    pub e_input_shares: Vec<B::V>,
    pub e2_input_commit: B::V,
    pub e1_wire_shares: BTreeMap<GateId, B::V>,
    // pub e1_view: PlayerState<F, B>,
    pub e2_view_commit: B::V,
    pub e_rand: BTreeMap<GateId, B::V>,
    pub e1_rand: BTreeMap<GateId, B::V>,
    // pub e1_rand: Vec<B::V>,
    pub e2_rand_commit: B::V,
    pub e2_output_shares: Vec<B::V>,
    pub transcript_digest: B::V,
}

impl<F: FiniteRing, B: Backend<F>> ZKBoogieEachProof<F, B> {
    pub fn verify_each(
        &self,
        back: &mut B,
        circuit: &Circuit<F>,
        expected_outputs: &[B::V],
    ) -> Result<B::V, B::Error> {
        assert!(self.e < 3, "Invalid challenge");
        let mut e_view = PlayerState::<F, B>::new(self.e);
        let mut e1_view = PlayerState::<F, B>::new((self.e + 1) % 3);
        e1_view.wire_shares = self.e1_wire_shares.clone();
        e1_view.num_input = circuit.num_inputs();
        let mut is_valid: <B as Backend<F>>::V = back.constant(&F::one())?;
        // println!("proof e: {}", self.e);
        for gate in circuit.enumerate_gates().into_iter() {
            match gate.gate_type {
                GateType::Input(input_idx) => {
                    e_view.add_input_share(
                        gate.clone(),
                        self.e_input_shares[input_idx as usize].clone(),
                    );
                    // e1_inputs.push(self.e1_view.wire_shares[&gate.gate_id].clone());
                    // if self.e == 0 {
                    //     // let self_share_value = hash_input_share(back, &self.e_seed, input_idx)?;
                    //     e_view.add_input_share(gate.clone(), self.e_input_shares[input_idx as usize].clone());
                    //     // let next_share_value = hash_input_share(back, &self.e1_seed, input_idx)?;
                    //     // let is_eq =
                    //     //     back.eq(&next_share_value, &self.e1_view.wire_shares[&gate.gate_id])?;
                    //     // is_valid = back.mul(&is_valid, &is_eq)?;
                    // } else if self.e == 1 {
                    //     let self_share_value = hash_input_share(back, &self.e_seed, input_idx)?;
                    //     e_view.add_input_share(gate.clone(), self_share_value);
                    //     let third_input_shares = self.third_input_shares.as_ref().unwrap();
                    //     let is_eq = back.eq(
                    //         &third_input_shares[input_idx as usize],
                    //         &self.e1_view.wire_shares[&gate.gate_id],
                    //     )?;
                    //     is_valid = back.mul(&is_valid, &is_eq)?;
                    // } else {
                    //     let third_input_shares = self.third_input_shares.as_ref().unwrap();
                    //     e_view.add_input_share(
                    //         gate.clone(),
                    //         third_input_shares[input_idx as usize].clone(),
                    //     );
                    //     let next_share_value =hash_input_share(back, &self.e1_seed, input_idx)?;
                    //     let is_eq =
                    //         back.eq(&next_share_value, &self.e1_view.wire_shares[&gate.gate_id])?;
                    //     is_valid = back.mul(&is_valid, &is_eq)?;
                    // }
                }
                GateType::ConstAdd(_) => {
                    e_view.run_const_add(back, gate.clone())?;
                }
                GateType::ConstMul(_) => {
                    e_view.run_const_mul(back, gate.clone())?;
                }
                GateType::Add => {
                    e_view.run_add(back, gate.clone())?;
                }
                GateType::Mul => {
                    e_view
                        .rand_pads
                        .insert(gate.gate_id, self.e_rand[&gate.gate_id].clone());
                    e1_view
                        .rand_pads
                        .insert(gate.gate_id, self.e1_rand[&gate.gate_id].clone());
                    e_view.run_mul(back, gate.clone(), &mut e1_view)?;
                }
                GateType::Neg => {
                    e_view.run_neg(back, gate.clone())?;
                }
                GateType::Const(_) => {
                    e_view.run_const(back, gate.clone())?;
                }
            }
        }
        let e0_input_commit = e_view.commit_input_shares(back)?;
        let e1_input_commit = e1_view.commit_input_shares(back)?;
        let s1_input = match self.e {
            0 => vec![
                e0_input_commit,
                e1_input_commit,
                self.e2_input_commit.clone(),
            ],
            1 => vec![
                self.e2_input_commit.clone(),
                e0_input_commit,
                e1_input_commit,
            ],
            2 => vec![
                e1_input_commit,
                self.e2_input_commit.clone(),
                e0_input_commit,
            ],
            _ => panic!("invalid challenge"),
        };
        let s1 = back.hash_to_one(&s1_input)?;
        let e0_view_commit = e_view.commit_view(back, &s1)?;
        let e1_view_commit = e1_view.commit_view(back, &s1)?;
        let s2_input = match self.e {
            0 => vec![
                s1,
                e0_view_commit,
                e1_view_commit,
                self.e2_view_commit.clone(),
            ],
            1 => vec![
                s1,
                self.e2_view_commit.clone(),
                e0_view_commit,
                e1_view_commit,
            ],
            2 => vec![
                s1,
                e1_view_commit,
                self.e2_view_commit.clone(),
                e0_view_commit,
            ],
            _ => panic!("invalid challenge"),
        };
        let s2 = back.hash_to_one(&s2_input)?;
        let e0_rand_commit = e_view.commit_rand_pads(back, &s2)?;
        let e1_rand_commit = e1_view.commit_rand_pads(back, &s2)?;
        let s3_input = match self.e {
            0 => vec![
                s2,
                e0_rand_commit,
                e1_rand_commit,
                self.e2_rand_commit.clone(),
            ],
            1 => vec![
                s2,
                self.e2_rand_commit.clone(),
                e0_rand_commit,
                e1_rand_commit,
            ],
            2 => vec![
                s2,
                e1_rand_commit,
                self.e2_rand_commit.clone(),
                e0_rand_commit,
            ],
            _ => panic!("invalid challenge"),
        };
        let s3 = back.hash_to_one(&s3_input)?;
        is_valid = back.eq(&s3, &self.transcript_digest)?;
        let e0_output_shares = circuit
            .output_ids
            .iter()
            .map(|id| e_view.wire_shares[id].clone())
            .collect_vec();
        let e1_output_shares = circuit
            .output_ids
            .iter()
            .map(|id| e1_view.wire_shares[id].clone())
            .collect_vec();
        for idx in 0..circuit.output_ids.len() {
            let recovered = {
                let add1 = back.add(&e0_output_shares[idx], &e1_output_shares[idx])?;
                back.add(&add1, &self.e2_output_shares[idx])?
            };
            back.expose_value(&expected_outputs[idx])?;
            let is_eq = back.eq(&recovered, &expected_outputs[idx])?;
            is_valid = back.mul(&is_valid, &is_eq)?;
        }

        Ok(is_valid)
    }
}

#[derive(Debug, Clone)]
pub struct ZKBoogieProof<F: FiniteRing, B: Backend<F>> {
    pub each_proof: Vec<ZKBoogieEachProof<F, B>>,
}

impl<F: FiniteRing, B: Backend<F>> ZKBoogieProof<F, B> {
    pub fn verify_final(&self, secpar: u8, back: &mut B) -> Result<B::V, B::Error> {
        let num_repeat = compute_num_repeat(secpar);
        debug_assert_eq!(self.each_proof.len(), num_repeat as usize);
        let mut challenge_inputs = vec![];
        for proof in self.each_proof.iter() {
            back.expose_value(&proof.transcript_digest)?;
            challenge_inputs.push(proof.transcript_digest.clone());
        }
        let challenge = back.hash_to_multi(&challenge_inputs)?;
        let challenge_ternarys = challenge
            .into_iter()
            .flat_map(|c| back.to_ternarys_le(&c))
            .collect_vec();
        let mut is_valid: <B as Backend<F>>::V = back.constant(&F::one())?;
        debug_assert!(challenge_ternarys.len() > num_repeat as usize);
        for idx in 0..num_repeat {
            let e_proof = back.load_value(&F::from(self.each_proof[idx as usize].e as u32))?;
            // back.expose_value(&e_proof)?;
            let is_eq = back.eq(&e_proof, &challenge_ternarys[idx as usize])?;
            is_valid = back.mul(&is_valid, &is_eq)?;
        }
        Ok(is_valid)
    }
}

impl<F: FiniteRing, H: NativeHasher<F>> ZKBoogieProof<F, NativeBackend<F, H>> {
    pub fn from_bytes_le(bytes: &[u8]) -> Self {
        let encoded: EncodedZKBoogieProof = bincode::deserialize(&bytes).unwrap();
        encoded.to_raw()
    }

    pub fn to_bytes_le(self) -> Vec<u8> {
        let encoded = EncodedZKBoogieProof::from_raw(self);
        bincode::serialize(&encoded).unwrap()
    }

    pub fn verify_whole(
        &self,
        secpar: u8,
        hasher_prefix: Vec<F>,
        circuit: &Circuit<F>,
        expected_output: &[F],
    ) -> Result<bool, NativeError> {
        let num_repeat = compute_num_repeat(secpar);
        debug_assert_eq!(self.each_proof.len(), num_repeat as usize);
        let is_valids = self
            .each_proof
            .par_iter()
            .map(|proof| {
                let mut back = NativeBackend::new(hasher_prefix.clone()).unwrap();
                proof
                    .verify_each(&mut back, circuit, expected_output)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        if is_valids.contains(&F::zero()) {
            return Ok(false);
        }

        let mut back = NativeBackend::new(hasher_prefix.clone()).unwrap();
        let is_final_valid = self.verify_final(secpar, &mut back)?;
        Ok(is_final_valid == F::one())
    }
}

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
            let timer = start_timer!(|| "commit");
            let transcript_digest = prover.commit(&mut back, circuit, input).unwrap();
            end_timer!(timer);
            (prover, transcript_digest)
        })
        .collect::<Vec<_>>();

    let challenge_inputs = commit_phases
        .iter()
        .flat_map(|(_, digest)| digest.clone())
        .collect_vec();
    let mut back = NativeBackend::<F, H>::new(hasher_prefix.clone()).unwrap();
    let challenge_timer = start_timer!(|| "challenge");
    let challenge = back.hash_to_multi(&challenge_inputs)?;
    end_timer!(challenge_timer);
    let challenge_ternarys = challenge
        .into_iter()
        .flat_map(|c| back.to_ternarys_le(&c))
        .collect_vec();
    let each_proof = commit_phases
        .into_par_iter()
        .enumerate()
        .map(|(idx, (mut prover, transcript_digest))| {
            let e: u8 = challenge_ternarys[idx].get_first_byte();
            let timer = start_timer!(|| "response");
            let response = prover.response(circuit, e, challenge_inputs[idx]).unwrap();
            end_timer!(timer);
            response
        })
        .collect::<Vec<_>>();
    Ok(ZKBoogieProof { each_proof })
}

fn rlc<F: FiniteRing, B: Backend<F>>(
    back: &mut B,
    rand: &B::V,
    vector: &[B::V],
) -> Result<B::V, B::Error> {
    let mut sum = B::zero(back)?;
    let mut coeff = B::one(back)?;
    for v in vector {
        let term = back.mul(&coeff, v)?;
        sum = back.add(&sum, &term)?;
        coeff = back.mul(&coeff, &rand)?;
    }
    Ok(sum)
}

fn compute_num_repeat(secpar: u8) -> u8 {
    let denom = (3f64.log2() - 1.0).ceil();
    (secpar as f64 / denom).ceil() as u8
}

#[cfg(test)]
mod test {
    use std::fs::File;

    use self::{
        backend::native::poseidon254_native::Poseidon254Native, circuit::CircuitBuilder,
        finite::F256,
    };
    use super::frontend::{
        build_circuit_from_circom, gen_circom_circuit_inputs, gen_circom_circuit_outputs,
    };
    use super::*;
    use ark_bn254::Fr;
    use ark_std::*;
    type F = F256<ark_bn254::Fr>;

    #[test]
    fn test_add2_mul() {
        let mut circuit_builder = CircuitBuilder::<F>::new();
        let inputs = circuit_builder.inputs(3);
        let add1 = circuit_builder.add(&inputs[0], &inputs[1]);
        let add2 = circuit_builder.add(&inputs[1], &inputs[2]);
        let mul = circuit_builder.mul(&add1, &add2);
        let circuit = circuit_builder.output(&[mul]);

        let mut rng: rand::prelude::StdRng = ark_std::test_rng();
        let inputs = vec![F::rand(&mut rng), F::rand(&mut rng), F::rand(&mut rng)];
        let expected_output = {
            let add1 = inputs[0].add(&inputs[1]);
            let add2 = inputs[1].add(&inputs[2]);
            let mul = add1.mul(&add2);
            vec![mul]
        };

        let secpar = 100;
        let hasher_prefix = vec![];
        let prove_time = start_timer!(|| "Proving");
        let proof = zkboogie_prove::<F, Poseidon254Native>(
            secpar,
            hasher_prefix.clone(),
            &circuit,
            &inputs,
        )
        .unwrap();
        end_timer!(prove_time);

        let is_valid = proof
            .verify_whole(secpar, hasher_prefix, &circuit, &expected_output)
            .unwrap();
        assert!(is_valid);
    }

    #[test]
    fn test_all_ops() {
        let mut circuit_builder = CircuitBuilder::<F>::new();
        let inputs = circuit_builder.inputs(4);
        let const_add = circuit_builder.const_add(F::from(1u32), &inputs[0]);
        let const_mul = circuit_builder.const_mul(F::from(2u32), &inputs[1]);
        let add1 = circuit_builder.add(&const_add, &inputs[2]);
        let mul = circuit_builder.mul(&const_mul, &inputs[3]);
        let add2 = circuit_builder.add(&add1, &mul);
        let circuit = circuit_builder.output(&[add1, mul, add2]);

        let mut rng: rand::prelude::StdRng = ark_std::test_rng();
        let inputs = vec![
            F::rand(&mut rng),
            F::rand(&mut rng),
            F::rand(&mut rng),
            F::rand(&mut rng),
        ];
        let expected_output = {
            let const_add = F::from(1u32).add(&inputs[0]);
            let const_mul = F::from(2u32).mul(&inputs[1]);
            let add1 = const_add.add(&inputs[2]);
            let mul = const_mul.mul(&inputs[3]);
            let add2 = add1.add(&mul);
            vec![add1, mul, add2]
        };

        let secpar = 100;
        let hasher_prefix = vec![];
        // let mut back = NativeBackend::<F, Poseidon254Native>::new(hasher_prefix).unwrap();
        let prove_time = start_timer!(|| "Proving");
        let proof = zkboogie_prove::<F, Poseidon254Native>(
            secpar,
            hasher_prefix.clone(),
            &circuit,
            &inputs,
        )
        .unwrap();
        end_timer!(prove_time);
        let is_valid = proof
            .verify_whole(secpar, hasher_prefix, &circuit, &expected_output)
            .unwrap();
        assert!(is_valid);
    }
}
