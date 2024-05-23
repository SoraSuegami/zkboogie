use std::collections::BTreeMap;

use crate::{Circuit, FiniteRing, Gate, GateId, GateType};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub enum EncodedGateType {
    Input(u32),
    ConstAdd(Vec<u8>),
    ConstMul(Vec<u8>),
    Add,
    Mul,
    Neg,
    Const(Vec<u8>),
}

impl EncodedGateType {
    pub fn from_raw<F: FiniteRing>(raw: GateType<F>) -> Self {
        match raw {
            GateType::Input(x) => EncodedGateType::Input(x),
            GateType::ConstAdd(x) => EncodedGateType::ConstAdd(x.to_bytes_le()),
            GateType::ConstMul(x) => EncodedGateType::ConstMul(x.to_bytes_le()),
            GateType::Add => EncodedGateType::Add,
            GateType::Mul => EncodedGateType::Mul,
            GateType::Neg => EncodedGateType::Neg,
            GateType::Const(x) => EncodedGateType::Const(x.to_bytes_le()),
        }
    }

    pub fn to_raw<F: FiniteRing>(&self) -> GateType<F> {
        match self {
            EncodedGateType::Input(x) => GateType::Input(*x),
            EncodedGateType::ConstAdd(x) => GateType::ConstAdd(F::from_bytes_le(x)),
            EncodedGateType::ConstMul(x) => GateType::ConstMul(F::from_bytes_le(x)),
            EncodedGateType::Add => GateType::Add,
            EncodedGateType::Mul => GateType::Mul,
            EncodedGateType::Neg => GateType::Neg,
            EncodedGateType::Const(x) => GateType::Const(F::from_bytes_le(x)),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct EncodedGate {
    pub gate_type: EncodedGateType,
    pub gate_id: GateId,
    pub inputs: Vec<GateId>,
}

impl EncodedGate {
    pub fn from_raw<F: FiniteRing>(raw: Gate<F>) -> Self {
        EncodedGate {
            gate_type: EncodedGateType::from_raw(raw.gate_type),
            gate_id: raw.gate_id,
            inputs: raw.inputs,
        }
    }

    pub fn to_raw<F: FiniteRing>(self) -> Gate<F> {
        Gate {
            gate_type: self.gate_type.to_raw(),
            gate_id: self.gate_id,
            inputs: self.inputs,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct EncodedCircuit {
    pub gates: BTreeMap<GateId, EncodedGate>,
    pub input_ids: Vec<GateId>,
    pub output_ids: Vec<GateId>,
}

impl EncodedCircuit {
    pub fn from_raw<F: FiniteRing>(raw: Circuit<F>) -> Self {
        EncodedCircuit {
            gates: BTreeMap::from_iter(
                raw.gates
                    .into_iter()
                    .map(|(k, v)| (k, EncodedGate::from_raw(v))),
            ),
            input_ids: raw.input_ids,
            output_ids: raw.output_ids,
        }
    }

    pub fn to_raw<F: FiniteRing>(self) -> Circuit<F> {
        Circuit {
            gates: BTreeMap::from_iter(self.gates.into_iter().map(|(k, v)| (k, v.to_raw()))),
            input_ids: self.input_ids,
            output_ids: self.output_ids,
        }
    }
}
