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
        e1_view: &View<F, B>,
        e2_view: &View<F, B>,
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
    pub e: u8,
    pub e1_view: View<F, B>,
    pub e_seed: Vec<B::V>,
    pub e1_seed: Vec<B::V>,
    pub third_input_shares: Vec<B::V>,
    pub transcript_digest: Vec<B::V>,
}

impl<F: FiniteRing, B: Backend<F>> ZKBooEachProof<F, B> {
    pub fn verify_each(
        &self,
        back: &mut B,
        circuit: &Circuit<F>,
        expected_outputs: &[B::V],
    ) -> Result<B::V, B::Error> {
        assert!(self.e < 3, "Invalid challenge");
        let mut self_view = View::<F, B>::new(self.e, self.e_seed.clone());
        let mut is_valid: <B as Backend<F>>::V = back.constant(&F::one())?;
        for gate in circuit.enumerate_gates().into_iter() {
            match gate.gate_type {
                GateType::Input(input_idx) => {
                    if self.e == 0 {
                        let self_share_value = back.hash_input_share(&self.e_seed, input_idx)?;
                        self_view.add_input_share(gate.clone(), self_share_value);
                        let next_share_value = back.hash_input_share(&self.e1_seed, input_idx)?;
                        let is_eq =
                            back.eq(&next_share_value, &self.e1_view.wire_shares[&gate.gate_id])?;
                        is_valid = back.mul(&is_valid, &is_eq)?;
                    } else if self.e == 1 {
                        let self_share_value = back.hash_input_share(&self.e_seed, input_idx)?;
                        self_view.add_input_share(gate.clone(), self_share_value);
                    } else {
                        self_view.add_input_share(
                            gate.clone(),
                            self.third_input_shares[input_idx as usize].clone(),
                        );
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
        let transcript_input = if self.e == 0 {
            [
                e_outputs, e1_outputs, e2_outputs, e_commit, e1_commit, e2_commit,
            ]
            .concat()
        } else if self.e == 1 {
            [
                e2_outputs, e_outputs, e1_outputs, e2_commit, e_commit, e1_commit,
            ]
            .concat()
        } else {
            [
                e1_outputs, e2_outputs, e_outputs, e1_commit, e2_commit, e_commit,
            ]
            .concat()
        };
        let transcript_digest = back.hash_each_transcript(&transcript_input)?;
        for (given_v, self_v) in self.transcript_digest.iter().zip(transcript_digest.iter()) {
            let is_eq = back.eq(given_v, self_v)?;
            is_valid = back.mul(&is_valid, &is_eq)?;
        }
        Ok(is_valid)
    }
}

#[derive(Debug, Clone)]
pub struct ZKBooEachProver<F: FiniteRing, H: NativeHasher<F>> {
    view0: View<F, NativeBackend<F, H>>,
    view1: View<F, NativeBackend<F, H>>,
    view2: View<F, NativeBackend<F, H>>,
}

impl<F: FiniteRing, H: NativeHasher<F>> ZKBooEachProver<F, H> {
    pub fn new<R: Rng>(rng: &mut R) -> Self {
        let seed_len = {
            let ceil = (256 as f64 / F::modulo_bits_size() as f64).ceil();
            if ceil < 1.0 {
                1
            } else {
                ceil as usize
            }
        };
        let rand_seeds: Vec<Vec<F>> = (0..3)
            .map(|_| (0..seed_len).map(|_| F::rand(rng)).collect_vec())
            .collect_vec();
        Self {
            view0: View::new(0, rand_seeds[0].clone()),
            view1: View::new(1, rand_seeds[1].clone()),
            view2: View::new(2, rand_seeds[2].clone()),
        }
    }

    pub fn commit(
        &mut self,
        back: &mut NativeBackend<F, H>,
        circuit: &Circuit<F>,
        input: &[F],
    ) -> Result<Vec<F>, NativeError> {
        let gates = circuit.enumerate_gates();
        for gate in gates.into_iter() {
            for player_idx in 0..3 {
                let (self_view, next_view, next_next_view) = if player_idx == 0 {
                    (&mut self.view0, &self.view1, &self.view2)
                } else if player_idx == 1 {
                    (&mut self.view1, &self.view2, &self.view0)
                } else {
                    (&mut self.view2, &self.view0, &self.view1)
                };
                match gate.gate_type {
                    GateType::Input(input_idx) => {
                        self_view.gen_input_share(
                            back,
                            gate.clone(),
                            &input[input_idx as usize],
                            next_view,
                            next_next_view,
                        )?;
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
                        self_view.run_mul(back, gate.clone(), next_view)?;
                    }
                }
            }
        }
        let mut outputs0 = vec![];
        let mut outputs1 = vec![];
        let mut outputs2 = vec![];
        for output_gate_id in circuit.output_ids.iter() {
            outputs0.push(self.view0.wire_shares[output_gate_id].clone());
            outputs1.push(self.view1.wire_shares[output_gate_id].clone());
            outputs2.push(self.view2.wire_shares[output_gate_id].clone());
        }
        let commit0 = self.view0.commit_view(back)?;
        let commit1 = self.view1.commit_view(back)?;
        let commit2 = self.view2.commit_view(back)?;
        let transcript_input = [outputs0, outputs1, outputs2, commit0, commit1, commit2].concat();
        let transcript_digest = back.hash_each_transcript(&transcript_input)?;
        Ok(transcript_digest)
    }

    pub fn response(
        &mut self,
        back: &mut NativeBackend<F, H>,
        circuit: &Circuit<F>,
        transcript_digest: Vec<F>,
        e: u8,
    ) -> Result<ZKBooEachProof<F, NativeBackend<F, H>>, NativeError> {
        debug_assert!(e < 3, "Invalid challenge");
        let e2_commit = if e == 0 {
            self.view2.commit_view(back)?
        } else if e == 1 {
            self.view0.commit_view(back)?
        } else {
            self.view1.commit_view(back)?
        };
        let e1_view = if e == 0 {
            &self.view1
        } else if e == 1 {
            &self.view2
        } else {
            &self.view0
        };
        let e_seed = if e == 0 {
            self.view0.rand_seed()
        } else if e == 1 {
            self.view1.rand_seed()
        } else {
            self.view2.rand_seed()
        };
        let e1_seed = if e == 0 {
            self.view1.rand_seed()
        } else if e == 1 {
            self.view2.rand_seed()
        } else {
            self.view0.rand_seed()
        };
        let third_input_shares = circuit
            .input_ids
            .iter()
            .map(|input_id| self.view2.wire_shares[input_id])
            .collect_vec();
        let proof = ZKBooEachProof {
            e2_commitment: e2_commit,
            e,
            e1_view: e1_view.clone(),
            e_seed: e_seed.to_vec(),
            e1_seed: e1_seed.to_vec(),
            third_input_shares,
            transcript_digest,
        };
        Ok(proof)
    }
}

#[derive(Debug, Clone)]
pub struct ZKBooProof<F: FiniteRing, B: Backend<F>> {
    pub each_proof: Vec<ZKBooEachProof<F, B>>,
}

impl<F: FiniteRing, B: Backend<F>> ZKBooProof<F, B> {
    pub fn verify_final(&self, secpar: u8, back: &mut B) -> Result<B::V, B::Error> {
        let num_repeat = compute_num_repeat(secpar);
        debug_assert_eq!(self.each_proof.len(), num_repeat as usize);
        let mut challenge_inputs = vec![];
        for proof in self.each_proof.iter() {
            for v in proof.transcript_digest.iter() {
                back.expose_value(v)?;
                challenge_inputs.push(v.clone());
            }
        }
        let challenge = back.hash_challenge(&challenge_inputs)?;
        let challenge_ternarys = challenge
            .into_iter()
            .flat_map(|c| back.to_ternarys_le(&c))
            .collect_vec();
        let mut is_valid: <B as Backend<F>>::V = back.constant(&F::one())?;
        debug_assert!(challenge_ternarys.len() > num_repeat as usize);
        for idx in 0..num_repeat {
            let e_proof = back.load_value(&F::from(self.each_proof[idx as usize].e as u32))?;
            back.expose_value(&e_proof)?;
            let is_eq = back.eq(&e_proof, &challenge_ternarys[idx as usize])?;
            is_valid = back.mul(&is_valid, &is_eq)?;
        }
        Ok(is_valid)
    }
}

impl<F: FiniteRing, H: NativeHasher<F>> ZKBooProof<F, NativeBackend<F, H>> {
    pub fn verify_whole(
        &self,
        secpar: u8,
        back: &mut NativeBackend<F, H>,
        circuit: &Circuit<F>,
        expected_output: &[F],
    ) -> Result<bool, NativeError> {
        let num_repeat = compute_num_repeat(secpar);
        debug_assert_eq!(self.each_proof.len(), num_repeat as usize);
        for proof in self.each_proof.iter() {
            let is_valid = proof.verify_each(back, circuit, expected_output)?;
            if is_valid != F::one() {
                return Ok(false);
            }
        }
        let is_final_valid = self.verify_final(secpar, back)?;
        Ok(is_final_valid == F::one())
    }
}

pub fn zkboo_prove<F: FiniteRing, H: NativeHasher<F>, R: Rng>(
    secpar: u8,
    rng: &mut R,
    back: &mut NativeBackend<F, H>,
    circuit: &Circuit<F>,
    input: &[F],
) -> Result<ZKBooProof<F, NativeBackend<F, H>>, NativeError> {
    let num_repeat = compute_num_repeat(secpar);
    let mut provers = vec![];
    let mut transcript_digests = vec![];
    for _ in 0..num_repeat {
        let mut prover = ZKBooEachProver::new(rng);
        let transcript_digest = prover.commit(back, &circuit, &input)?;
        transcript_digests.push(transcript_digest);
        provers.push(prover);
    }
    let challenge_inputs = transcript_digests.iter().cloned().flatten().collect_vec();
    let challenge = back.hash_challenge(&challenge_inputs)?;
    let challenge_ternarys = challenge
        .into_iter()
        .flat_map(|c| back.to_ternarys_le(&c))
        .collect_vec();
    let mut each_proof = vec![];
    for idx in 0..num_repeat {
        let e: u8 = challenge_ternarys[idx as usize].get_first_byte();
        let proof = provers[idx as usize].response(
            back,
            circuit,
            transcript_digests[idx as usize].clone(),
            e,
        )?;
        each_proof.push(proof);
    }
    Ok(ZKBooProof { each_proof })
}

fn compute_num_repeat(secpar: u8) -> u8 {
    let denom = (3f64.log2() - 1.0).ceil();
    (secpar as f64 / denom).ceil() as u8
}

// pub fn zkboo_prove<F:FiniteRing, H:NativeHasher<F>>(
//     secpar: u8,
//     back: &mut NativeBackend<F, NativeBackend<F, H>>,

// )

// fn zkboo_prove_each<F: FiniteRing, H: NativeHasher<F>, R:Rng>(
//     secpar: u8,
//     back: &mut NativeBackend<F, H>,
//     circuit: Circuit<F>,
//     input: Vec<F>,
//     rng: &mut R
// ) -> ZKBooEachProof<F, >

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

#[cfg(test)]
mod test {
    use self::{circuit::CircuitBuilder, finite::Fp, poseidon254_native::Poseidon254Native};
    use super::*;
    use ark_std::*;

    type F = Fp<ark_bn254::Fr>;

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

        let hasher_prefix = vec![];
        let mut back = NativeBackend::<F, Poseidon254Native>::new(hasher_prefix).unwrap();
        let proof = zkboo_prove(2, &mut rng, &mut back, &circuit, &inputs).unwrap();
        let is_valid = proof
            .verify_whole(2, &mut back, &circuit, &expected_output)
            .unwrap();
        assert!(is_valid);
    }
}
