use crate::finite::FiniteRing;
use crate::*;
use ark_bn254::{Bn254, Fr};
use ark_circom::*;
use ark_ff::{BigInteger, Zero};
use ark_ff::{BigInteger256, PrimeField};
use itertools::Itertools;
use num_bigint::BigInt;
use num_bigint::ToBigInt;
use std::path::Path;
use std::str::FromStr;
use thiserror::Error;

use self::{
    circuit::{Circuit, CircuitBuilder},
    finite::Fp,
};

type F = Fp<Fr>;

#[derive(Error, Debug)]
pub enum FrontendError {
    #[error("circom config generation failed: {0}")]
    CircomConfigError(String),
    #[error("circom circuit witness calculation failed: {0}")]
    CircomWitnessError(String),
}

pub fn build_circuit_from_circom<P: AsRef<Path>>(
    r1cs: P,
    wasm: P,
) -> Result<Circuit<F>, FrontendError> {
    let cfg = CircomConfig::<Bn254>::new(wasm, r1cs)
        .map_err(|err| FrontendError::CircomConfigError(err.to_string()))?;
    let r1cs = cfg.r1cs;
    let mut circuit_builder = CircuitBuilder::<F>::new();
    let num_inputs = r1cs.num_variables;
    let inputs = circuit_builder.inputs(num_inputs);
    let mut zs = vec![];
    for constraint in r1cs.constraints.into_iter() {
        let (vec_a, vec_b, vec_c) = constraint;
        let mut sumed = vec![];
        for vec in [vec_a, vec_b, vec_c] {
            let mut sum = None;
            for (idx, (var_idx, coeff)) in vec.into_iter().enumerate() {
                let term = circuit_builder.const_mul(Fp(coeff), &inputs[var_idx]);
                sum = if idx == 0 {
                    Some(term)
                } else {
                    Some(circuit_builder.add(&sum.unwrap(), &term))
                };
            }
            sumed.push(sum.unwrap());
        }
        let muled = circuit_builder.mul(&sumed[0], &sumed[1]);
        let z = circuit_builder.sub(&sumed[2], &muled);
        zs.push(z);
    }
    let public_inputs_outputs = (0..r1cs.num_inputs).map(|idx| inputs[idx]).collect_vec();
    let circuit = circuit_builder.output(&[public_inputs_outputs, zs].concat());
    Ok(circuit)
}

pub fn gen_circom_circuit_inputs<P: AsRef<Path>>(
    r1cs: P,
    wasm: P,
    inputs: Vec<(impl ToString, F)>,
) -> Result<Vec<F>, FrontendError> {
    let cfg = CircomConfig::<Bn254>::new(wasm, r1cs)
        .map_err(|err| FrontendError::CircomConfigError(err.to_string()))?;
    let mut builder = CircomBuilder::new(cfg);
    for (name, val) in inputs.into_iter() {
        builder.push_input(
            name,
            BigInt::from_bytes_le(num_bigint::Sign::Plus, &val.0.into_repr().to_bytes_le()),
        );
    }
    let circom_circuit = builder
        .build()
        .map_err(|err| FrontendError::CircomWitnessError(err.to_string()))?;
    let inputs = circom_circuit
        .witness
        .unwrap()
        .into_iter()
        .map(|x| Fp(x))
        .collect_vec();
    Ok(inputs)
}

pub fn gen_circom_circuit_outputs<P: AsRef<Path>>(
    r1cs: P,
    wasm: P,
    public_inputs: Vec<F>,
    public_outputs: Vec<F>,
) -> Result<Vec<F>, FrontendError> {
    let cfg = CircomConfig::<Bn254>::new(wasm, r1cs)
        .map_err(|err| FrontendError::CircomConfigError(err.to_string()))?;
    let mut outputs = vec![Fp::one()];
    outputs.append(&mut public_outputs.into_iter().collect_vec());
    outputs.append(&mut public_inputs.into_iter().collect_vec());
    for _ in 0..cfg.r1cs.constraints.len() {
        outputs.push(Fp::zero());
    }
    Ok(outputs)
}

#[cfg(test)]
mod test {
    use self::{circuit::CircuitBuilder, finite::Fp};
    use super::*;
    use ark_std::*;

    type F = Fp<ark_bn254::Fr>;

    #[test]
    fn test_1_circom() {
        let r1cs_path = "./test_circom/test1.r1cs";
        let wasm_path = "./test_circom/test1_js/test1.wasm";

        let circuit = build_circuit_from_circom(r1cs_path, wasm_path).unwrap();
        let mut rng = ark_std::test_rng();
        let inputs = (0..circuit.num_inputs())
            .map(|_| F::rand(&mut rng))
            .collect_vec();
        let public_output = inputs[0].mul(&inputs[1]);
        let expected_output =
            gen_circom_circuit_outputs(r1cs_path, wasm_path, vec![], vec![public_output]).unwrap();
        let circuit_inputs = gen_circom_circuit_inputs(
            r1cs_path,
            wasm_path,
            vec![("a", inputs[0].clone()), ("b", inputs[1])],
        )
        .unwrap();
        let output = circuit.eval(&circuit_inputs);
        assert_eq!(output, expected_output);
    }
}
