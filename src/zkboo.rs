use std::collections::HashMap;

use itertools::Itertools;

use crate::*;

use self::{
    backend::Backend,
    circuit::{Circuit, Gate, GateId, GateType},
    finite::FiniteRing,
};

#[derive(Debug, Clone)]
pub struct View<F: FiniteRing, B: Backend<F>> {
    player_idx: u8,
    rand_seed: F,
    wire_shares: HashMap<GateId, (B::V, bool)>,
}

impl<F: FiniteRing, B: Backend<F>> View<F, B> {
    pub fn new(player_idx: u8, rand_seed: F) -> Self {
        View {
            player_idx,
            rand_seed,
            wire_shares: HashMap::new(),
        }
    }

    pub fn rand_seed(&self) -> F {
        self.rand_seed
    }

    pub fn wire_share(&self, gate_id: GateId) -> Option<&(B::V, bool)> {
        self.wire_shares.get(&gate_id)
    }

    pub fn compressed_shares(&self) -> Vec<(GateId, B::V)> {
        self.wire_shares
            .iter()
            .filter(|(_, (_, is_indp))| *is_indp)
            .map(|(gate_id, (value, _))| (*gate_id, value.clone()))
            .collect_vec()
    }

    pub fn add_input_share(&mut self, gate_id: GateId, value: B::V) {
        self.wire_shares.insert(gate_id, (value, true));
    }

    pub fn run_const_add(&mut self, back: &mut B, gate: Gate<F>) -> Result<B::V, B::Error> {
        let constant = if let GateType::ConstAdd(c) = gate.gate_type {
            c
        } else {
            panic!("Expected constant gate")
        };
        let value = &self.wire_shares[&gate.inputs[0]].0;
        let new_value = if self.player_idx == 0 {
            let constant = back.constant(&constant)?;
            back.add(&value, &constant)?
        } else {
            value.clone()
        };
        self.wire_shares
            .insert(gate.gate_id, (new_value.clone(), false));
        Ok(new_value)
    }

    pub fn run_const_mul(&mut self, back: &mut B, gate: Gate<F>) -> Result<B::V, B::Error> {
        let constant = if let GateType::ConstMul(c) = gate.gate_type {
            c
        } else {
            panic!("Expected constant gate")
        };
        let value = &self.wire_shares[&gate.inputs[0]].0;
        let constant = back.constant(&constant)?;
        let new_value = back.mul(&value, &constant)?;
        self.wire_shares
            .insert(gate.gate_id, (new_value.clone(), false));
        Ok(new_value)
    }

    pub fn run_add(&mut self, back: &mut B, gate: Gate<F>) -> Result<B::V, B::Error> {
        let left = &self.wire_shares[&gate.inputs[0]].0;
        let right = &self.wire_shares[&gate.inputs[1]].0;
        let new_value = back.add(&left, &right)?;
        self.wire_shares
            .insert(gate.gate_id, (new_value.clone(), false));
        Ok(new_value)
    }

    pub fn run_mul(
        &mut self,
        back: &mut B,
        gate: Gate<F>,
        other_view: &View<F, B>,
    ) -> Result<B::V, B::Error> {
        let self_left = &self.wire_shares[&gate.inputs[0]].0;
        let self_right = &self.wire_shares[&gate.inputs[1]].0;
        let other_left = &other_view.wire_shares[&gate.inputs[0]].0;
        let other_right = &other_view.wire_shares[&gate.inputs[1]].0;
        let new_value = {
            let term1 = back.mul(&self_left, &self_right)?;
            let term2 = back.mul(&other_left, &self_right)?;
            let term3 = back.mul(&self_left, &other_right)?;
            let term4 = back.hash_with_seed(&self.rand_seed, &[F::from(gate.gate_id.0)])?;
            let term5 = back.hash_with_seed(&other_view.rand_seed, &[F::from(gate.gate_id.0)])?;
            let add1 = back.add(&term1, &term2)?;
            let add2 = back.add(&add1, &term3)?;
            let add3 = back.add(&add2, &term4)?;
            back.sub(&add3, &term5)?
        };
        self.wire_shares
            .insert(gate.gate_id, (new_value.clone(), false));
        Ok(new_value)
    }
}

#[derive(Debug, Clone)]
pub struct ZKBooProver<F: FiniteRing, B: Backend<F>> {
    pub views: Vec<View<F, B>>,
    pub circuit: Circuit<F>,
}

impl<F: FiniteRing, B: Backend<F>> ZKBooProver<F, B> {
    pub fn new(rand_seeds: [F; 3], circuit: Circuit<F>) -> Self {
        let views = (0..3)
            .map(|idx| View::new(idx as u8, rand_seeds[idx]))
            .collect_vec();
        let mut zkboo = ZKBooProver { views, circuit };
        zkboo
    }
}
