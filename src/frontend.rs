use crate::finite::FiniteRing;
use crate::*;
use ark_bn254::{Bn254, Fr};
use ark_circom::circom::{R1CSFile, R1CS};
use ark_circom::*;
use ark_ff::{BigInteger, Field, Zero};
use ark_ff::{BigInteger256, PrimeField};
use itertools::Itertools;
use num_bigint::BigInt;
use num_bigint::ToBigInt;
use petgraph::graph::*;
use petgraph::visit::EdgeRef;
use rayon::vec;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{Read, Seek};
use std::path::Path;
use std::str::FromStr;
use thiserror::Error;

use self::{
    circuit::{Circuit, CircuitBuilder},
    finite::F256,
};

type F = F256<Fr>;

#[derive(Error, Debug)]
pub enum FrontendError {
    #[error("r1cs loading failed: {0}")]
    R1CSError(String),
    #[error("circom circuit witness calculation failed: {0}")]
    CircomWitnessError(String),
}

type R1CSVariablesGraph = Graph<Vec<usize>, ()>;
pub struct AnalyzedR1CS {
    pub r1cs: R1CS<Bn254>,
    pub graph: R1CSVariablesGraph,
    pub node_of_var_pairs: HashMap<Vec<usize>, NodeIndex>,
    pub non_implied_wires: HashSet<usize>,
}

impl AnalyzedR1CS {
    pub fn new(r1cs: R1CS<Bn254>) -> Self {
        let (graph, node_of_var_pairs) = AnalyzedR1CS::make_r1cs_variable_graph(&r1cs);
        let non_implied_wires =
            AnalyzedR1CS::find_non_implied_wires(&r1cs, &graph, &node_of_var_pairs);
        AnalyzedR1CS {
            r1cs,
            graph,
            node_of_var_pairs,
            non_implied_wires,
        }
    }

    fn make_r1cs_variable_graph(
        r1cs: &R1CS<Bn254>,
    ) -> (R1CSVariablesGraph, HashMap<Vec<usize>, NodeIndex>) {
        let mut graph = R1CSVariablesGraph::new();
        let mut node_of_var_pairs = HashMap::new();
        for idx in 0..r1cs.num_inputs {
            let node = graph.add_node(vec![idx]);
            node_of_var_pairs.insert(vec![idx], node);
        }
        for constraint in r1cs.constraints.iter() {
            let (a_part, b_part, c_part) = constraint;
            let mut depend_vars = HashSet::new();
            for (var_wire, _) in a_part.iter() {
                if *var_wire == 5678 {
                    println!("a constaint {:?}", constraint);
                }
                if !depend_vars.contains(var_wire) {
                    depend_vars.insert(*var_wire);
                }
                if node_of_var_pairs.get(&vec![*var_wire]).is_none() {
                    let node = graph.add_node(vec![*var_wire]);
                    node_of_var_pairs.insert(vec![*var_wire], node);
                }
            }
            for (var_wire, _) in b_part.iter() {
                if *var_wire == 5678 {
                    println!("b constaint {:?}", constraint);
                }
                if !depend_vars.contains(var_wire) {
                    depend_vars.insert(*var_wire);
                }
                if node_of_var_pairs.get(&vec![*var_wire]).is_none() {
                    // println!("b new node {}", *var_wire);
                    let node = graph.add_node(vec![*var_wire]);
                    node_of_var_pairs.insert(vec![*var_wire], node);
                }
            }
            let mut c_vars = c_part.iter().map(|(var_wire, _)| *var_wire).collect_vec();
            c_vars.sort();
            if node_of_var_pairs.get(&c_vars).is_none() {
                let node = graph.add_node(c_vars.clone());
                node_of_var_pairs.insert(c_vars.clone(), node);
            }
            let c_vars_node = node_of_var_pairs.get(&c_vars).unwrap().clone();
            for var in depend_vars.into_iter() {
                let node = node_of_var_pairs.get(&vec![var]).unwrap();
                graph.add_edge(*node, c_vars_node, ());
            }
            let mut c_nodes = vec![];
            for c_var in c_vars.iter() {
                if *c_var == 5678 {
                    println!("c constaint {:?}", constraint);
                }
                if node_of_var_pairs.get(&vec![*c_var]).is_none() {
                    // println!("c new node {}", *c_var);
                    let node = graph.add_node(vec![*c_var]);
                    node_of_var_pairs.insert(vec![*c_var], node);
                }
                let c_node = node_of_var_pairs.get(&vec![*c_var]).unwrap().clone();
                c_nodes.push(c_node);
            }
            for x in c_nodes.iter() {
                for y in c_nodes.iter() {
                    if x == y {
                        continue;
                    }
                    graph.add_edge(c_vars_node, *x, ());
                    graph.add_edge(*y, *x, ());
                }
            }
        }
        (graph, node_of_var_pairs)
    }

    fn find_non_implied_wires(
        r1cs: &R1CS<Bn254>,
        graph: &R1CSVariablesGraph,
        node_of_var_pairs: &HashMap<Vec<usize>, NodeIndex>,
    ) -> HashSet<usize> {
        let num_inputs = r1cs.num_inputs;
        let mut non_implied_wires = HashSet::new();
        let mut derived_nodes = HashSet::<NodeIndex>::new();

        let find_derived_nodes = |start_node_idx: NodeIndex, derived_nodes: &HashSet<NodeIndex>| {
            let mut new_derived_nodes = vec![];
            while let Some(next_node_idx) = graph.neighbors(start_node_idx).next() {
                if derived_nodes.contains(&next_node_idx) {
                    continue;
                }
                let coming_edges =
                    graph.edges_directed(next_node_idx, petgraph::Direction::Incoming);
                let mut is_not_derived = false;
                for coming_edge in coming_edges.into_iter() {
                    let coming_node = coming_edge.source();
                    if !derived_nodes.contains(&coming_node) {
                        is_not_derived = true;
                        break;
                    }
                }
                if next_node_idx == node_of_var_pairs[&vec![5678]] {
                    // println!("coming edges: {:?}", coming_edges);
                    println!("is_not_derived: {}", is_not_derived);
                }
                if is_not_derived {
                    break;
                } else {
                    new_derived_nodes.push(next_node_idx);
                }
            }
            new_derived_nodes
        };
        for idx in 0..num_inputs {
            non_implied_wires.insert(idx);
            let node = node_of_var_pairs.get(&vec![idx]).unwrap();
            derived_nodes.insert(*node);
            let new_derived_nodes = find_derived_nodes(*node, &derived_nodes);
            for new_node in new_derived_nodes {
                derived_nodes.insert(new_node);
            }
        }
        let not_derived_nodes_single = node_of_var_pairs
            .iter()
            .filter(|(vars, node_id)| {
                if vars.len() != 1 {
                    return false;
                }
                !derived_nodes.contains(node_id)
            })
            .collect_vec();
        for (vars, node_id) in not_derived_nodes_single {
            if vars.len() == 0 && vars[0] == 5678 {
                println!("not derived node: {:?}", vars);
            }
            derived_nodes.insert(*node_id);
            let new_derived_nodes = find_derived_nodes(*node_id, &derived_nodes);
            if new_derived_nodes.len() == 0 {
                derived_nodes.remove(node_id);
                continue;
            }
            for new_node in new_derived_nodes {
                derived_nodes.insert(new_node);
            }
            non_implied_wires.insert(vars[0]);
        }

        for idx in 0..r1cs.num_variables {
            if !non_implied_wires.contains(&idx)
                && !derived_nodes.contains(&node_of_var_pairs[&vec![idx]])
            {
                non_implied_wires.insert(idx);
            }
        }
        non_implied_wires
    }
}

pub fn build_circuit_from_circom<R: Read + Seek>(r1cs: R) -> Result<Circuit<F>, FrontendError> {
    let r1csfile =
        R1CSFile::<Bn254>::new(r1cs).map_err(|err| FrontendError::R1CSError(err.to_string()))?;
    let r1cs = R1CS::from(r1csfile);
    let analyzed_r1cs = AnalyzedR1CS::new(r1cs);
    let mut wire_to_gate_id = HashMap::<usize, GateId>::new();

    let mut circuit_builder = CircuitBuilder::<F>::new();
    let inputs = circuit_builder.inputs(analyzed_r1cs.non_implied_wires.len());
    for (wire_id, gate_id) in analyzed_r1cs.non_implied_wires.iter().zip(inputs.iter()) {
        wire_to_gate_id.insert(*wire_id, *gate_id);
    }

    let num_inputs = analyzed_r1cs.r1cs.num_variables;
    let mut zs = vec![];
    // let mut wire_id_to_gate_id = HashMap::<usize, GateId>::new();
    // let mut gate_id_to_wire_id = HashMap::<GateId, usize>::new();

    for (constaint_idx, constraint) in analyzed_r1cs.r1cs.constraints.iter().enumerate() {
        let (vec_a, vec_b, vec_c) = constraint;

        let mut a_sum = circuit_builder.constant(F::zero());
        for (var_idx, coeff) in vec_a.into_iter() {
            println!("a var_idx: {}", var_idx);
            let term = circuit_builder.const_mul(F256(*coeff), &wire_to_gate_id[&var_idx]);
            a_sum = circuit_builder.add(&a_sum, &term);
        }

        let mut b_sum = circuit_builder.constant(F::zero());
        for (var_idx, coeff) in vec_b.into_iter() {
            let term = circuit_builder.const_mul(F256(*coeff), &wire_to_gate_id[&var_idx]);
            b_sum = circuit_builder.add(&b_sum, &term);
        }

        let mut known_c_coeff_wires = vec![];
        let mut unknown_c_coeff_wires = vec![];
        for (var_idx, coeff) in vec_b.into_iter() {
            if wire_to_gate_id.contains_key(&var_idx) {
                known_c_coeff_wires.push((coeff, var_idx));
            } else {
                unknown_c_coeff_wires.push((coeff, var_idx));
            }
        }

        if unknown_c_coeff_wires.len() == 1 {
            let unknown_coeff_inv = unknown_c_coeff_wires[0].0.inverse().unwrap();
            let mut unknown_c_var = circuit_builder.mul(&a_sum, &b_sum);
            for (coeff, var_idx) in known_c_coeff_wires.into_iter() {
                let coeff = -unknown_coeff_inv * coeff;
                let term = circuit_builder.const_mul(F256(coeff), &wire_to_gate_id[&var_idx]);
                unknown_c_var = circuit_builder.add(&unknown_c_var, &term);
            }
            wire_to_gate_id.insert(*unknown_c_coeff_wires[0].1, unknown_c_var);
            let z = circuit_builder.constant(F::zero());
            zs.push(z);
        } else if unknown_c_coeff_wires.len() == 0 {
            let mut sum = circuit_builder.constant(F::zero());
            for (var_idx, coeff) in vec_c.into_iter() {
                let term = circuit_builder.const_mul(F256(*coeff), &wire_to_gate_id[&var_idx]);
                sum = circuit_builder.add(&sum, &term);
            }
            let muled = circuit_builder.mul(&a_sum, &b_sum);
            let z = circuit_builder.sub(&sum, &muled);
            zs.push(z);
        } else {
            panic!(
                "the number of unknow c wires in the {}-th constraint is {}",
                constaint_idx,
                unknown_c_coeff_wires.len()
            );
        }
        // let muled = circuit_builder.mul(&sumed[0], &sumed[1]);
        // let z = circuit_builder.sub(&sumed[2], &muled);
        // zs.push(z);
    }
    println!("num inputs: {}", num_inputs);
    let public_inputs_outputs = (0..analyzed_r1cs.r1cs.num_inputs)
        .map(|idx| inputs[idx])
        .collect_vec();
    let circuit = circuit_builder.output(&[public_inputs_outputs, zs].concat());
    Ok(circuit)
}

// pub fn build_circuit_from_circom<R: Read + Seek>(r1cs: R) -> Result<Circuit<F>, FrontendError> {
//     let r1csfile =
//         R1CSFile::<Bn254>::new(r1cs).map_err(|err| FrontendError::R1CSError(err.to_string()))?;
//     let r1cs = R1CS::from(r1csfile);
//     println!("before make a graph");
//     make_r1cs_variable_graph(&r1cs);
//     println!("after make a graph");
//     let mut circuit_builder = CircuitBuilder::<F>::new();
//     let num_inputs = r1cs.num_variables;
//     let inputs = circuit_builder.inputs(num_inputs);
//     let mut zs = vec![];
//     // let mut wire_id_to_gate_id = HashMap::<usize, GateId>::new();
//     // let mut gate_id_to_wire_id = HashMap::<GateId, usize>::new();
//     let mut num_minimum_input = 0;
//     let mut set_wire_id = HashSet::new();
//     for idx in 0..r1cs.num_inputs {
//         // wire_id_to_gate_id.insert(idx, GateId(idx as u32));
//         // gate_id_to_wire_id.insert(GateId(idx as u32), idx);
//         set_wire_id.insert(idx);
//         num_minimum_input += 1;
//     }

//     for constraint in r1cs.constraints.into_iter() {
//         let (vec_a, vec_b, vec_c) = constraint;
//         let mut is_a_one = false;
//         let mut is_b_one = false;
//         let mut sumed = vec![];

//         let mut known_c_wires = vec![];
//         let mut unknown_c_wires = vec![];
//         for (idx, vec) in [vec_a, vec_b, vec_c].into_iter().enumerate() {
//             let is_one = vec.len() == 1 && vec[0].0 == 0;
//             if idx == 0 {
//                 is_a_one = is_one;
//             } else if idx == 1 {
//                 is_b_one = is_one;
//             }
//             let mut sum = circuit_builder.constant(F::zero());
//             for (var_idx, coeff) in vec.into_iter() {
//                 let term = circuit_builder.const_mul(F256(coeff), &inputs[var_idx]);
//                 sum = circuit_builder.add(&sum, &term);
//                 if idx < 2 {
//                     if !set_wire_id.contains(&var_idx) {
//                         set_wire_id.insert(var_idx);
//                         num_minimum_input += 1;
//                     }
//                 } else {
//                     if set_wire_id.contains(&var_idx) {
//                         known_c_wires.push(var_idx);
//                     } else {
//                         unknown_c_wires.push(var_idx);
//                     }
//                 }
//             }
//             sumed.push(sum);
//         }
//         if is_a_one {
//             let z = circuit_builder.sub(&sumed[2], &sumed[1]);
//             zs.push(z);
//         } else if is_b_one {
//             let z = circuit_builder.sub(&sumed[2], &sumed[0]);
//             zs.push(z);
//         } else {
//             let muled = circuit_builder.mul(&sumed[0], &sumed[1]);
//             let z = circuit_builder.sub(&sumed[2], &muled);
//             zs.push(z);
//         }
//         if unknown_c_wires.len() == 1 {
//             // println!("unknown_c_wires is one.");
//             set_wire_id.insert(unknown_c_wires[0]);
//         } else {
//             for wire_id in unknown_c_wires {
//                 set_wire_id.insert(wire_id);
//                 num_minimum_input += 1;
//             }
//         }
//         // let muled = circuit_builder.mul(&sumed[0], &sumed[1]);
//         // let z = circuit_builder.sub(&sumed[2], &muled);
//         // zs.push(z);
//     }
//     println!("num inputs: {}", num_inputs);
//     println!("num minimum inputs: {}", num_minimum_input);
//     println!("set_wire_id len: {}", set_wire_id.len());
//     let public_inputs_outputs = (0..r1cs.num_inputs).map(|idx| inputs[idx]).collect_vec();
//     let circuit = circuit_builder.output(&[public_inputs_outputs, zs].concat());
//     Ok(circuit)
// }

pub fn gen_circom_circuit_inputs(witness: Vec<Fr>) -> Result<Vec<F>, FrontendError> {
    // let cfg = CircomConfig::<Bn254>::new(wasm, r1cs)
    //     .map_err(|err| FrontendError::CircomConfigError(err.to_string()))?;
    // let mut builder = CircomBuilder::new(cfg);
    // for (name, val) in inputs.into_iter() {
    //     builder.push_input(
    //         name,
    //         BigInt::from_bytes_le(num_bigint::Sign::Plus, &val.0.into_repr().to_bytes_le()),
    //     );
    // }
    // let circom_circuit = builder
    //     .build()
    //     .map_err(|err| FrontendError::CircomWitnessError(err.to_string()))?;
    let inputs = witness.into_iter().map(|x| F256(x)).collect_vec();
    Ok(inputs)
}

pub fn gen_circom_circuit_outputs<R: Read + Seek>(
    r1cs: R,
    public_inputs: Vec<Fr>,
    public_outputs: Vec<Fr>,
) -> Result<Vec<F>, FrontendError> {
    let r1csfile =
        R1CSFile::<Bn254>::new(r1cs).map_err(|err| FrontendError::R1CSError(err.to_string()))?;
    let r1cs = R1CS::from(r1csfile);
    let mut outputs = vec![F256::one()];
    outputs.append(&mut public_outputs.into_iter().map(|x| F256(x)).collect_vec());
    outputs.append(&mut public_inputs.into_iter().map(|x| F256(x)).collect_vec());
    for _ in 0..r1cs.constraints.len() {
        outputs.push(F256::zero());
    }
    Ok(outputs)
}

pub fn gen_random_circom_circuit(num_input: u32, num_add: u32, num_mul: u32) -> String {
    let mut circom_strs = vec![];
    circom_strs.push(format!("pragma circom 2.0.0;"));
    circom_strs.push(format!(
        "template RandomCircuitInput{}Add{}Mul{}() {{",
        num_input, num_add, num_mul
    ));
    circom_strs.push(format!("\tsignal input in[{}];", num_input));
    circom_strs.push(format!("\tsignal output out;"));
    let mut num_used_input = 1;
    let mut num_used_add = 0;
    let num_gates = num_add + num_mul;
    let mut last_input_l = "in[0]".to_string();
    let mut last_input_r = "in[0]".to_string();
    let mut num_var = 0;
    for _ in 0..num_gates {
        let new_var = format!("var{}", num_var);
        num_var += 1;
        if num_used_add < num_add {
            num_used_add += 1;
            circom_strs.push(format!(
                "\tsignal {} <== {} + {};",
                new_var, last_input_l, last_input_r
            ));
        } else {
            circom_strs.push(format!(
                "\tsignal {} <== {} * {};",
                new_var, last_input_l, last_input_r
            ));
        };
        last_input_l = if num_used_input < num_input {
            num_used_input += 1;
            format!("in[{}]", num_used_input - 1)
        } else {
            new_var.clone()
        };
        last_input_r = new_var;
    }
    circom_strs.push(format!("\tout <== {};", last_input_r));
    circom_strs.push(format!("}}"));
    circom_strs.push(format!(
        "component main = RandomCircuitInput{}Add{}Mul{}();",
        num_input, num_add, num_mul
    ));
    circom_strs.join("\n")
}

#[cfg(test)]
mod test {
    use self::{circuit::CircuitBuilder, finite::F256};
    use super::*;
    use ark_std::*;

    type F = F256<ark_bn254::Fr>;

    #[ignore]
    #[test]
    fn test_1_circom() {
        let r1cs = File::open("./test_circom/test1.r1cs").unwrap();
        let circuit = build_circuit_from_circom(r1cs).unwrap();
        let mut rng = ark_std::test_rng();
        let inputs = (0..circuit.num_inputs())
            .map(|_| F::rand(&mut rng))
            .collect_vec();
        let public_output = inputs[0].mul(&inputs[1]);
        let r1cs = File::open("./test_circom/test1.r1cs").unwrap();
        let expected_output =
            gen_circom_circuit_outputs(r1cs, vec![], vec![public_output.0]).unwrap();
        let witness = vec![Fr::one(), public_output.0, inputs[0].0, inputs[1].0];
        let circuit_inputs = gen_circom_circuit_inputs(witness).unwrap();
        let output = circuit.eval(&circuit_inputs);
        assert_eq!(output, expected_output);
    }

    #[ignore]
    #[test]
    fn test_gen_random_circom_circuit() {
        let params = [
            (128, 128, 128),
            (128, 256, 256),
            (128, 512, 512),
            (128, 1024, 1024),
            (128, 2048, 2048),
            (128, 4096, 4096),
            (128, 8192, 8192),
            (128, 16384, 16384),
        ];
        for (num_input, num_add, num_mul) in params.iter() {
            let circuit = gen_random_circom_circuit(*num_input, *num_add, *num_mul);
            fs::write(
                format!("./input{}_add{}_mul{}.circom", num_input, num_add, num_mul),
                circuit,
            )
            .unwrap();
        }
    }
}
