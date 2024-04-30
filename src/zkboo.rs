use self::{
    backend::Backend,
    circuit::{Circuit, Gate, GateId, GateType},
    finite::FiniteRing,
};
use crate::backend::native::*;
use crate::*;
use arkworks_native_gadgets::ark_std::rand::Rng;
use itertools::Itertools;
use std::{collections::BTreeMap, marker::PhantomData};
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct View<F: FiniteRing, B: Backend<F>> {
    player_idx: u8,
    rand_seed: Vec<B::V>,
    wire_shares: BTreeMap<GateId, B::V>,
}

impl<F: FiniteRing, B: Backend<F>> View<F, B> {
    pub fn new(player_idx: u8, rand_seed: Vec<B::V>) -> Self {
        View {
            player_idx,
            rand_seed,
            wire_shares: BTreeMap::new(),
        }
    }

    pub fn rand_seed(&self) -> &[B::V] {
        &self.rand_seed
    }

    pub fn wire_share(&self, gate_id: GateId) -> Option<&B::V> {
        self.wire_shares.get(&gate_id)
    }

    // pub fn compressed_shares(&self) -> Vec<(GateId, B::V)> {
    //     self.wire_shares
    //         .iter()
    //         .filter(|(_, (_, is_indp))| *is_indp)
    //         .map(|(gate_id, (value, _))| (*gate_id, value.clone()))
    //         .collect_vec()
    // }

    pub fn commit_view(&self, back: &mut B) -> Result<Vec<B::V>, B::Error> {
        let shares = self.wire_shares.values().cloned().collect_vec();
        let commit = back.hash_commit(self.rand_seed(), &shares)?;
        Ok(commit)
    }

    pub fn gen_input_share(
        &mut self,
        back: &mut B,
        gate: Gate<F>,
        input: &B::V,
        e1_view: &mut View<F, B>,
        e2_view: &mut View<F, B>,
    ) -> Result<(), B::Error> {
        if self.player_idx < 2 {
            if let GateType::Input(input_idx) = gate.gate_type {
                let value = back.hash_input_share(&self.rand_seed, input_idx)?;
                self.wire_shares.insert(gate.gate_id, value);
                Ok(())
            } else {
                panic!("Expected input gate");
            }
        } else {
            if let GateType::Input(input_idx) = gate.gate_type {
                let value_0 = back.hash_input_share(&e1_view.rand_seed, input_idx)?;
                let value_1 = back.hash_input_share(&e2_view.rand_seed, input_idx)?;
                let value_2 = {
                    let sub0 = back.sub(input, &value_0)?;
                    back.sub(&sub0, &value_1)?
                };
                self.wire_shares.insert(gate.gate_id, value_2);
                Ok(())
            } else {
                panic!("Expected input gate");
            }
        }
    }

    pub fn add_input_share(&mut self, gate: Gate<F>, share: B::V) {
        let is_indp = if self.player_idx < 2 { false } else { true };
        self.wire_shares.insert(gate.gate_id, share);
    }

    pub fn run_const_add(&mut self, back: &mut B, gate: Gate<F>) -> Result<(), B::Error> {
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
        Ok(())
    }

    pub fn run_const_mul(&mut self, back: &mut B, gate: Gate<F>) -> Result<(), B::Error> {
        let constant = if let GateType::ConstMul(c) = gate.gate_type {
            c
        } else {
            panic!("Expected constant gate")
        };
        let value = &self.wire_shares[&gate.inputs[0]];
        let constant = back.constant(&constant)?;
        let new_value = back.mul(&value, &constant)?;
        self.wire_shares.insert(gate.gate_id, new_value);
        Ok(())
    }

    pub fn run_add(&mut self, back: &mut B, gate: Gate<F>) -> Result<(), B::Error> {
        let left = &self.wire_shares[&gate.inputs[0]];
        let right = &self.wire_shares[&gate.inputs[1]];
        let new_value = back.add(&left, &right)?;
        self.wire_shares.insert(gate.gate_id, new_value);
        Ok(())
    }

    pub fn run_mul(
        &mut self,
        back: &mut B,
        gate: Gate<F>,
        other_view: &View<F, B>,
    ) -> Result<(), B::Error> {
        let self_left = &self.wire_shares[&gate.inputs[0]];
        let self_right = &self.wire_shares[&gate.inputs[1]];
        let other_left = &other_view.wire_shares[&gate.inputs[0]];
        let other_right = &other_view.wire_shares[&gate.inputs[1]];
        let new_value = {
            let term1 = back.mul(&self_left, &self_right)?;
            let term2 = back.mul(&other_left, &self_right)?;
            let term3 = back.mul(&self_left, &other_right)?;
            let gate_id = back.constant(&F::from(gate.gate_id.0))?;
            let term4 = back.hash_mul_pad(&self.rand_seed, &[gate_id.clone()])?;
            let term5 = back.hash_mul_pad(&other_view.rand_seed, &[gate_id])?;
            let add1 = back.add(&term1, &term2)?;
            let add2 = back.add(&add1, &term3)?;
            let add3 = back.add(&add2, &term4)?;
            back.sub(&add3, &term5)?
        };
        self.wire_shares.insert(gate.gate_id, new_value);
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ZKBooEachProof<F: FiniteRing, B: Backend<F>> {
    pub e2_commitment: Vec<B::V>,
    pub e1_view: View<F, B>,
    pub e_seed: Vec<B::V>,
    pub e1_seed: Vec<B::V>,
    pub transcript_digest: Vec<B::V>,
}

// #[derive(Debug, Clone)]
// pub struct ZKBooEachVerifyResult<F: FiniteRing, B: Backend<F>> {
//     pub is_valid: B::V,
//     pub transcript_digest: Vec<B::V>,
// }

impl<F: FiniteRing, B: Backend<F>> ZKBooEachProof<F, B> {
    pub fn verify_each(
        &self,
        back: &mut B,
        circuit: &Circuit<F>,
        expected_outputs: &[B::V],
        e: u8,
    ) -> Result<B::V, B::Error> {
        assert!(e < 3, "Invalid challenge");
        let mut self_view = View::<F, B>::new(e, self.e_seed.clone());
        let mut is_valid: <B as Backend<F>>::V = back.constant(&F::one())?;
        for gate in circuit.enumerate_gates().into_iter() {
            match gate.gate_type {
                GateType::Input(input_idx) => {
                    if e == 0 {
                        let self_share_value = back.hash_input_share(&self.e_seed, input_idx)?;
                        self_view.add_input_share(gate.clone(), self_share_value);
                        let next_share_value = back.hash_input_share(&self.e1_seed, input_idx)?;
                        let is_eq =
                            back.eq(&next_share_value, &self.e1_view.wire_shares[&gate.gate_id])?;
                        is_valid = back.mul(&is_valid, &is_eq)?;
                    } else if e == 1 {
                        let self_share_value = back.hash_input_share(&self.e_seed, input_idx)?;
                        self_view.add_input_share(gate.clone(), self_share_value);
                    } else {
                        let next_share_value = back.hash_input_share(&self.e1_seed, input_idx)?;
                        let is_eq =
                            back.eq(&next_share_value, &self.e1_view.wire_shares[&gate.gate_id])?;
                        is_valid = back.mul(&is_valid, &is_eq)?;
                    }
                }
                GateType::ConstAdd(_) => {
                    self_view.run_const_add(back, gate.clone())?;
                }
                GateType::ConstMul(_) => {
                    self_view.run_const_mul(back, gate.clone())?;
                }
                GateType::Add => {
                    self_view.run_add(back, gate.clone())?;
                }
                GateType::Mul => {
                    self_view.run_mul(back, gate.clone(), &self.e1_view)?;
                }
            }
        }

        let mut e_outputs = Vec::new();
        let mut e1_outputs = Vec::new();
        let mut e2_outputs = Vec::new();
        for (output_idx, output_gate_id) in circuit.output_ids.iter().enumerate() {
            let e_output = self_view.wire_shares[output_gate_id].clone();
            let e1_output = self.e1_view.wire_shares[output_gate_id].clone();
            let e2_output = {
                let sub0 = back.sub(&expected_outputs[output_idx], &e_output)?;
                back.sub(&sub0, &e1_output)?
            };
            e_outputs.push(e_output);
            e1_outputs.push(e1_output);
            e2_outputs.push(e2_output);
        }
        let e_commit = self_view.commit_view(back)?;
        let e1_commit = self.e1_view.commit_view(back)?;
        let e2_commit = self.e2_commitment.clone();
        let transcript_digest = back.hash_each_transcript(
            &[
                e_outputs, e1_outputs, e2_outputs, e_commit, e1_commit, e2_commit,
            ]
            .concat(),
        )?;
        for (given_v, self_v) in self.transcript_digest.iter().zip(transcript_digest.iter()) {
            let is_eq = back.eq(given_v, self_v)?;
            is_valid = back.mul(&is_valid, &is_eq)?;
        }
        Ok(is_valid)
    }
}

#[derive(Debug, Clone)]
pub struct ZKBooProof<F: FiniteRing, B: Backend<F>> {
    pub each_proof: Vec<ZKBooEachProof<F, B>>,
}

// impl<F: FiniteRing, B: Backend<F>> ZKBooProof<F, B> {
//     pub fn verify(
//         &self,
//         secpar: u8,
//         back: &mut B,
//         circuit: &Circuit<F>,
//         expected_outputs: &[B::V],
//     ) -> Result<B::V, B::Error> {
//         let mut is_valid: <B as Backend<F>>::V = back.constant(&F::one())?;
//         for (e, each_proof) in self.each_proof.iter().enumerate() {
//             let is_valid_e = each_proof.verify_each(back, circuit, expected_outputs, e as u8)?;
//             is_valid = back.mul(&is_valid, &is_valid_e)?;
//         }
//         Ok(is_valid)
//     }
// }

// pub fn zkboo_prove<F:FiniteRing, H:NativeHasher<F>>(
//     secpar: u8,
//     back: &mut NativeBackend<F, NativeBackend<F, H>>,

// )

// #[derive(Debug, Clone)]
// pub struct ZKBooProver<F: FiniteRing, H: NativeHasher<F>> {
//     _f: PhantomData<F>,
//     _h: PhantomData<H>,
// }

// impl<F: FiniteRing, H: NativeHasher<F>> ZKBooProver<F, H> {
//     pub fn prove<R: Rng>(
//         back: &mut NativeBackend<F, H>,
//         circuit: Circuit<F>,
//         input: Vec<F>,
//         rng: &mut R,
//     ) -> ZKBooProver<F, NativeBackend<F, H>> {
//         let seed_len = {
//             let ceil = (256 as f64 / F::modulo_bits_size() as f64).ceil();
//             if ceil < 1.0 {
//                 1
//             } else {
//                 ceil as usize
//             }
//         };
//         let rand_seeds: Vec<Vec<F>> = (0..3)
//             .map(|_| (0..seed_len).map(|_| F::rand(rng)).collect_vec())
//             .collect_vec();
//         let views = (0..3)
//             .map(|idx| View::<F, NativeBackend<F, H>>::new(idx as u8, rand_seeds[idx]))
//             .collect_vec();
//         let gates = circuit.enumerate_gates();
//         for gate in gates.into_iter() {
//             match gate.gate_type {
//                 GateType::Input(input_idx) => {
//                     for view in &mut views {
//                         let player_idx = view.player_idx as usize;
//                         view.add_input_share(
//                             &mut back,
//                             gate,
//                             &input[input_idx as usize],
//                             &mut views[(player_idx + 1) % 3],
//                             &mut views[(player_idx + 2) % 3],
//                         );
//                     }
//                 }
//                 GateType::ConstAdd(_) => {
//                     for view in &mut views {
//                         view.run_const_add(&mut back, gate);
//                     }
//                 }
//                 GateType::ConstMul(_) => {
//                     for view in &mut views {
//                         view.run_const_mul(&mut back, gate);
//                     }
//                 }
//                 GateType::Add => {
//                     for view in &mut views {
//                         view.run_add(&mut back, gate);
//                     }
//                 }
//                 GateType::Mul => {
//                     for view in &mut views {
//                         let player_idx = view.player_idx as usize;
//                         view.run_mul(&mut back, gate, &views[(player_idx + 1) % 3]);
//                     }
//                 }
//             }
//         }
//         let zkboo = ZKBooProver {
//             views,
//             circuit,
//             input,
//         };
//         zkboo
//     }
// }
