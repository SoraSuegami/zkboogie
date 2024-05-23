mod encode;
pub use encode::*;
use itertools::Itertools;

use self::finite::FiniteRing;
use crate::*;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, ops::*, vec};

#[derive(Debug, Clone, Copy)]
pub enum GateType<F: FiniteRing> {
    Input(u32),
    ConstAdd(F),
    ConstMul(F),
    Add,
    Mul,
    Neg,
    Const(F),
    // Output(u32),
}

impl<F: FiniteRing> GateType<F> {
    pub fn type_id(&self) -> usize {
        match self {
            GateType::Input(_) => 0,
            GateType::ConstAdd(_) => 1,
            GateType::ConstMul(_) => 2,
            GateType::Add => 3,
            GateType::Mul => 4,
            GateType::Neg => 5,
            GateType::Const(_) => 6,
            // GateType::Output(_) => 5,
        }
    }

    pub fn num_input(&self) -> usize {
        match self {
            GateType::Input(_) => 1,
            GateType::ConstAdd(_) => 1,
            GateType::ConstMul(_) => 1,
            GateType::Add => 2,
            GateType::Mul => 2,
            GateType::Neg => 1,
            GateType::Const(_) => 0,
            // GateType::Output(_) => 1,
        }
    }

    pub fn eval(&self, inputs: &[F]) -> F {
        match self {
            GateType::Input(_) => panic!("Input gate should not be evaluated"),
            GateType::ConstAdd(x) => inputs[0].add(x),
            GateType::ConstMul(x) => inputs[0].mul(x),
            GateType::Add => inputs[0].add(&inputs[1]),
            GateType::Mul => inputs[0].mul(&inputs[1]),
            GateType::Neg => inputs[0].neg(),
            GateType::Const(x) => *x,
            // GateType::Output(_) => inputs[0],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct GateId(pub u32);

impl GateId {
    pub fn new(id: u32) -> Self {
        GateId(id)
    }

    pub fn id(&self) -> u32 {
        self.0
    }
}

impl Add for GateId {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        GateId(self.0 + other.0)
    }
}

impl Sub for GateId {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        GateId(self.0 - other.0)
    }
}

#[derive(Debug, Clone)]
pub struct Gate<F: FiniteRing> {
    pub gate_type: GateType<F>,
    pub gate_id: GateId,
    pub inputs: Vec<GateId>,
}

impl<F: FiniteRing> Gate<F> {
    pub fn new(gate_type: GateType<F>, gate_id: GateId, inputs: Vec<GateId>) -> Self {
        Gate {
            gate_type,
            gate_id,
            inputs,
        }
    }

    pub fn input(&self, idx: usize) -> GateId {
        self.inputs[idx]
    }
}

#[derive(Debug, Clone)]
pub struct Circuit<F: FiniteRing> {
    pub gates: BTreeMap<GateId, Gate<F>>,
    pub input_ids: Vec<GateId>,
    pub output_ids: Vec<GateId>,
}

impl<F: FiniteRing> Circuit<F> {
    pub fn new(
        gates: BTreeMap<GateId, Gate<F>>,
        input_ids: Vec<GateId>,
        output_ids: Vec<GateId>,
    ) -> Self {
        Circuit {
            gates,
            input_ids,
            output_ids,
        }
    }

    pub fn num_gates(&self) -> usize {
        self.gates.len()
    }

    pub fn num_inputs(&self) -> usize {
        self.input_ids.len()
    }

    pub fn num_outputs(&self) -> usize {
        self.output_ids.len()
    }

    pub fn input_id(&self, idx: usize) -> GateId {
        self.input_ids[idx]
    }

    pub fn output_id(&self, idx: usize) -> GateId {
        self.output_ids[idx]
    }

    pub fn gate(&self, gate_id: &GateId) -> &Gate<F> {
        &self.gates[gate_id]
    }

    pub fn enumerate_gates(&self) -> Vec<Gate<F>> {
        let mut gates: Vec<_> = self.gates.iter().collect();
        gates.sort_by_key(|(id, _)| *id);
        gates.into_iter().map(|(_, gate)| gate.clone()).collect()
    }

    pub fn eval(&self, inputs: &[F]) -> Vec<F> {
        eval_circuit(self, inputs)
    }

    pub fn to_bytes_le(self) -> Vec<u8> {
        let encoded: EncodedCircuit = EncodedCircuit::from_raw(self.clone());
        bincode::serialize(&encoded).unwrap()
    }

    pub fn from_bytes_le(bytes: &[u8]) -> Self {
        let encoded: EncodedCircuit = bincode::deserialize(bytes).unwrap();
        encoded.to_raw()
    }
}

#[derive(Debug, Clone)]
pub struct CircuitBuilder<F: FiniteRing> {
    pub gates: BTreeMap<GateId, Gate<F>>,
    pub input_ids: Vec<GateId>,
    pub output_ids: Vec<GateId>,
}

impl<F: FiniteRing> CircuitBuilder<F> {
    pub fn new() -> Self {
        Self {
            gates: BTreeMap::new(),
            input_ids: Vec::new(),
            output_ids: Vec::new(),
        }
    }

    pub fn inputs(&mut self, num_inputs: usize) -> Vec<GateId> {
        debug_assert_eq!(self.input_ids.len(), 0);
        debug_assert_eq!(self.gates.len(), 0);
        let gate_ids = (0..num_inputs)
            .map(|idx| GateId::new(idx as u32))
            .collect_vec();
        let mut inputs = vec![];
        for id in &gate_ids {
            self.input_ids.push(*id);
            let input_gate = Gate::new(GateType::Input::<F>(id.0), *id, vec![]);
            self.gates.insert(*id, input_gate);
            inputs.push(*id);
        }
        inputs
    }

    pub fn output(mut self, output_ids: &[GateId]) -> Circuit<F> {
        self.output_ids = output_ids.to_vec();
        Circuit::new(self.gates, self.input_ids, self.output_ids)
    }

    pub fn const_add(&mut self, constant: F, input: &GateId) -> GateId {
        let gate_id = GateId::new(self.gates.len() as u32);
        let gate = Gate::new(GateType::ConstAdd::<F>(constant), gate_id, vec![*input]);
        self.gates.insert(gate_id, gate);
        gate_id
    }

    pub fn const_mul(&mut self, constant: F, input: &GateId) -> GateId {
        let gate_id = GateId::new(self.gates.len() as u32);
        let gate = Gate::new(GateType::ConstMul::<F>(constant), gate_id, vec![*input]);
        self.gates.insert(gate_id, gate);
        gate_id
    }

    pub fn add(&mut self, input_l: &GateId, input_r: &GateId) -> GateId {
        let gate_id = GateId::new(self.gates.len() as u32);
        let gate = Gate::new(GateType::Add::<F>, gate_id, vec![*input_l, *input_r]);
        self.gates.insert(gate_id, gate);
        gate_id
    }

    pub fn mul(&mut self, input_l: &GateId, input_r: &GateId) -> GateId {
        let gate_id = GateId::new(self.gates.len() as u32);
        let gate = Gate::new(GateType::Mul::<F>, gate_id, vec![*input_l, *input_r]);
        self.gates.insert(gate_id, gate);
        gate_id
    }

    pub fn neg(&mut self, input: &GateId) -> GateId {
        let gate_id = GateId::new(self.gates.len() as u32);
        let gate = Gate::new(GateType::Neg::<F>, gate_id, vec![*input]);
        self.gates.insert(gate_id, gate);
        gate_id
    }

    pub fn sub(&mut self, input_l: &GateId, input_r: &GateId) -> GateId {
        let neg = self.neg(input_r);
        self.add(input_l, &neg)
    }

    pub fn constant(&mut self, constant: F) -> GateId {
        let gate_id = GateId::new(self.gates.len() as u32);
        let gate = Gate::new(GateType::Const::<F>(constant), gate_id, vec![]);
        self.gates.insert(gate_id, gate);
        gate_id
    }
}

fn eval_circuit<F: FiniteRing>(circuit: &Circuit<F>, inputs: &[F]) -> Vec<F> {
    let mut values = BTreeMap::new();
    debug_assert_eq!(inputs.len(), circuit.num_inputs());
    for (idx, input) in inputs.iter().enumerate() {
        values.insert(circuit.input_id(idx), *input);
    }
    let num_inputs = circuit.num_inputs();
    for gate in circuit.enumerate_gates()[num_inputs..].into_iter() {
        let inputs = gate
            .inputs
            .iter()
            .map(|input_id| values[input_id])
            .collect_vec();
        let value = gate.gate_type.eval(&inputs);
        values.insert(gate.gate_id, value);
    }
    circuit
        .output_ids
        .iter()
        .map(|output_id| values[output_id])
        .collect_vec()
}

pub fn gen_random_circuit<F: FiniteRing>(num_input: u32, num_add: u32, num_mul: u32) -> Vec<u8> {
    let mut circuit_builder = CircuitBuilder::<F>::new();
    let inputs = circuit_builder.inputs(num_input as usize);
    let mut num_used_input = 1;
    let mut num_used_add = 0;
    let num_gates = num_add + num_mul;
    let mut last_input_l = inputs[0];
    let mut last_input_r = inputs[0];
    for _ in 0..num_gates {
        let new_wire = if num_used_add < num_add {
            num_used_add += 1;
            circuit_builder.add(&last_input_l, &last_input_r)
        } else {
            circuit_builder.mul(&last_input_l, &last_input_r)
        };
        last_input_l = if num_used_input < num_input {
            num_used_input += 1;
            inputs[num_used_input as usize - 1]
        } else {
            new_wire
        };
        last_input_r = new_wire;
    }
    let circuit = circuit_builder.output(&[last_input_r]);
    circuit.to_bytes_le()
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_random_circuit() {
        gen_random_circuit::<F256<ark_bn254::Fr>>(1, 128, 128);
    }
}
