use std::{fs::File, io::BufReader};

use ark_bn254::Fr;
use ark_ff::PrimeField;
use ark_std::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use itertools::Itertools;
use serde_json;
use std::str::FromStr;
use zkboo_recursion::{
    native::{poseidon254_native::Poseidon254Native, NativeBackend},
    *,
};

type F = F256<ark_bn254::Fr>;

fn bench_twitter_circom(c: &mut Criterion) {
    let r1cs = BufReader::new(File::open("./benches/circom_twitter_data/twitter.r1cs").unwrap());
    let circuit = build_circuit_from_circom(r1cs).unwrap();
    println!("num gates {}", circuit.num_gates());
    let mut rng = ark_std::test_rng();
    let r1cs = BufReader::new(File::open("./benches/circom_twitter_data/twitter.r1cs").unwrap());
    let public_inputs =
        vec![Fr::from_str("1163446621798851219159656704542204983322218017645").unwrap()];
    let public_outputs = vec![
        Fr::from_str(
            "1983664618407009423875829639306275185491946247764487749439145140682408188330",
        )
        .unwrap(),
        Fr::from_str("131061634216091175196322682").unwrap(),
    ];
    let expected_output = gen_circom_circuit_outputs(r1cs, public_inputs, public_outputs).unwrap();
    let witness = {
        let file = File::open("./benches/circom_twitter_data/twitter-witness.json").unwrap();
        let bufreader = BufReader::new(file);
        let v: Vec<String> = serde_json::from_reader(bufreader).unwrap();
        v.into_iter()
            .map(|x| Fr::from_str(&x).unwrap())
            .collect_vec()
    };
    println!("before gen_circom_circuit_inputs");
    let circuit_inputs = gen_circom_circuit_inputs(witness).unwrap();
    println!("after gen_circom_circuit_inputs");

    let secpar = 1;
    println!("prover starts");
    let mut group = c.benchmark_group("twitter_circom_native");
    group.sample_size(10);
    group.bench_function("generate zkboo proof for twitter_circom circuit", |b| {
        b.iter(|| {
            let prove_time = start_timer!(|| "Proving");
            let _ = zkboo_prove::<F, Poseidon254Native, _>(
                secpar,
                &mut rng,
                vec![],
                &circuit,
                &circuit_inputs,
            )
            .unwrap();
            end_timer!(prove_time);
        })
    });

    let proof =
        zkboo_prove::<F, Poseidon254Native, _>(secpar, &mut rng, vec![], &circuit, &circuit_inputs)
            .unwrap();
    group.bench_function("verify the zkboo proof for twitter_circom circuit", |b| {
        b.iter(|| {
            let verify_time = start_timer!(|| "Verifying");
            let is_valid = proof
                .verify_whole(secpar, vec![], &circuit, &expected_output)
                .unwrap();
            end_timer!(verify_time);
            assert!(is_valid);
        })
    });
}

criterion_group!(benches, bench_twitter_circom,);
criterion_main!(benches);
