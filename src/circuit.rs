use itertools::Itertools;

use self::finite::FiniteRing;
use crate::*;
use std::{collections::HashMap, ops::*, vec};

#[derive(Debug, Clone, Copy)]
pub enum GateType<F: FiniteRing> {
    Input(u32),
    ConstAdd(F),
    ConstMul(F),
    Add,
    Mul,
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
            // GateType::Output(_) => inputs[0],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
    pub gates: HashMap<GateId, Gate<F>>,
    pub input_ids: Vec<GateId>,
    pub output_ids: Vec<GateId>,
}

impl<F: FiniteRing> Circuit<F> {
    pub fn new(
        gates: HashMap<GateId, Gate<F>>,
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

    pub fn enumerate_gates(self) -> Vec<Gate<F>> {
        let mut gates: Vec<_> = self.gates.iter().collect();
        gates.sort_by_key(|(id, _)| *id);
        gates.into_iter().map(|(_, gate)| gate.clone()).collect()
    }
}

#[derive(Debug, Clone)]
pub struct CircuitBuilder<F: FiniteRing> {
    pub gates: HashMap<GateId, Gate<F>>,
    pub input_ids: Vec<GateId>,
    pub output_ids: Vec<GateId>,
}

impl<F: FiniteRing> CircuitBuilder<F> {
    pub fn new() -> Self {
        Self {
            gates: HashMap::new(),
            input_ids: Vec::new(),
            output_ids: Vec::new(),
        }
    }

    pub fn inputs(&mut self, num_inputs: usize) {
        debug_assert_eq!(self.input_ids.len(), 0);
        debug_assert_eq!(self.gates.len(), 0);
        let gate_ids = (0..num_inputs)
            .map(|idx| GateId::new(idx as u32))
            .collect_vec();
        for id in &gate_ids {
            self.input_ids.push(*id);
            let input_gate = Gate::new(GateType::Input::<F>(id.0), *id, vec![]);
            self.gates.insert(*id, input_gate);
        }
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
}
