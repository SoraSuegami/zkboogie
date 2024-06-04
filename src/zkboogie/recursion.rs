use self::native::{NativeBackend, NativeError, NativeHasher};
use crate::*;
// use ark_std::{end_timer, start_timer};
use crate::ark::*;
use ark_bn254::{constraints::GVar, Bn254, Fr, G1Projective as G1};
use ark_ff::{One, Zero};
use ark_groth16::{Groth16, ProvingKey, VerifyingKey as G16VerifierKey};
use ark_grumpkin::{constraints::GVar as GVar2, Projective as G2};
use ark_poly_commit::kzg10::VerifierKey as KZGVerifierKey;
use ark_relations::r1cs::SynthesisError;
use arkworks_native_gadgets::prelude::ark_crypto_primitives::snark::SNARK;
use bincode;
use common::*;
use core::hash;
pub use encode::*;
use folding_schemes::{
    ccs::r1cs::R1CS,
    commitment::{
        kzg::{ProverKey as KZGProverKey, KZG},
        pedersen::Pedersen,
        CommitmentScheme,
    },
    folding::nova::{
        decider_eth::{prepare_calldata, Decider as DeciderEth},
        decider_eth_circuit::DeciderEthCircuit,
        get_r1cs, Nova, ProverParams, VerifierParams,
    },
    frontend::FCircuit,
    transcript::poseidon::poseidon_test_config,
    Decider, Error, FoldingScheme,
};
use itertools::Itertools;
use native::poseidon254_native::Poseidon254Native;
use rand::thread_rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, f64::consts::E};

#[derive(Debug, Clone)]
pub struct ZKBoogieEachVerifierCircuitParams {
    pub secpar: u8,
    pub e: u8,
    pub circuit: Circuit<F>,
    pub hasher_prefix: Vec<F>,
    // pub expected_output: Vec<F>,
}

#[derive(Debug, Clone)]
pub struct ZKBoogieEachVerifierCircuit {
    pub params: ZKBoogieEachVerifierCircuitParams,
}

impl FCircuit<Fr> for ZKBoogieEachVerifierCircuit {
    type Params = ZKBoogieEachVerifierCircuitParams;

    fn new(params: Self::Params) -> Result<Self, Error> {
        Ok(Self { params })
    }

    fn state_len(&self) -> usize {
        let num_repeat = compute_num_repeat(self.params.secpar) as usize;
        self.params.circuit.num_outputs() + num_repeat
    }

    fn external_inputs_len(&self) -> usize {
        5 + self.params.circuit.num_inputs()
            + self.params.circuit.num_outputs()
            + self.params.circuit.num_gates()
            + 2 * self.params.circuit.num_mul_gate()
    }

    fn step_native(
        &self,
        i: usize,
        z_i: Vec<Fr>,
        external_inputs: Vec<Fr>,
    ) -> Result<Vec<Fr>, Error> {
        let is_enable = external_inputs[0].clone();
        let each_proof =
            ZKBoogieEachProof::<F, NativeBackend<F, Poseidon254Native>>::from_field_vec(
                &external_inputs[1..]
                    .iter()
                    .map(|v| F256(v.clone()))
                    .collect_vec(),
                self.params.e,
                &self.params.circuit,
            );
        let mut next_z = z_i.clone();
        next_z[self.params.circuit.num_outputs() + i] = is_enable * each_proof.transcript_digest.0;
        Ok(next_z)
    }

    fn generate_step_constraints(
        &self,
        cs: ark_relations::r1cs::ConstraintSystemRef<Fr>,
        i: usize,
        z_i: Vec<ark_r1cs_std::fields::fp::FpVar<Fr>>,
        external_inputs: Vec<ark_r1cs_std::fields::fp::FpVar<Fr>>, // inputs that are not part of the state
    ) -> Result<Vec<ark_r1cs_std::fields::fp::FpVar<Fr>>, ark_relations::r1cs::SynthesisError> {
        let mut back = ArkBackend::<Poseidon254Ark>::new(self.params.hasher_prefix.clone())
            .expect("Failed to create backend");
        let is_enable = external_inputs[0].clone();
        let each_proof = ZKBoogieEachProof::<F, ArkBackend<Poseidon254Ark>>::from_field_vec(
            &external_inputs[1..],
            self.params.e,
            &self.params.circuit,
        );
        let expected_output = &z_i[0..self.params.circuit.num_outputs()];
        let is_valid = each_proof
            .verify_each(&mut back, &self.params.circuit, expected_output)
            .expect("Failed to verify each proof");
        {
            let one = back.one().unwrap();
            let subed = back.sub(&is_valid, &one).unwrap();
            let muled = back.mul(&is_enable, &subed).unwrap();
            back.force_zero(&muled).unwrap();
        }
        let transcript_digest = back.mul(&is_enable, &each_proof.transcript_digest).unwrap();
        let mut next_z = z_i.clone();
        next_z[self.params.circuit.num_outputs() + i] = transcript_digest;
        Ok(next_z)
    }
}

pub type NOVA =
    Nova<G1, GVar, G2, GVar2, ZKBoogieEachVerifierCircuit, KZG<'static, Bn254>, Pedersen<G2>>;
pub type DeciderFCircuit = DeciderEth<
    G1,
    GVar,
    G2,
    GVar2,
    ZKBoogieEachVerifierCircuit,
    KZG<'static, Bn254>,
    Pedersen<G2>,
    Groth16<Bn254>,
    NOVA,
>;

pub fn fold_setup(
    secpar: u8,
    circuit: &Circuit<F>,
    hasher_prefix: &[F],
) -> [(
    ProverParams<G1, G2, KZG<'static, Bn254>, Pedersen<G2>>,
    VerifierParams<G1, G2>,
); 3] {
    let f_circuit0 = build_f_circuit(secpar, 0, circuit, hasher_prefix);
    let f_circuit1 = build_f_circuit(secpar, 1, circuit, hasher_prefix);
    let f_circuit2 = build_f_circuit(secpar, 2, circuit, hasher_prefix);
    let mut rng = rand::rngs::OsRng;
    let poseidon_config = poseidon_test_config::<Fr>();
    let (r1cs, cf_r1cs) = get_r1cs::<G1, GVar, G2, GVar2, ZKBoogieEachVerifierCircuit>(
        &poseidon_config,
        f_circuit0.clone(),
    )
    .unwrap();
    let cs_len = r1cs.A.n_rows;
    let cf_cs_len = cf_r1cs.A.n_rows;
    let (kzg_pk, kzg_vk): (KZGProverKey<G1>, KZGVerifierKey<Bn254>) =
        KZG::<Bn254>::setup(&mut rng, cs_len).unwrap();
    let (fs_prover_params0, fs_verifier_params0) = init_nova_ivc_params(f_circuit0, kzg_pk.clone());
    let (fs_prover_params1, fs_verifier_params1) = init_nova_ivc_params(f_circuit1, kzg_pk.clone());
    let (fs_prover_params2, fs_verifier_params2) = init_nova_ivc_params(f_circuit2, kzg_pk.clone());
    [
        (fs_prover_params0, fs_verifier_params0),
        (fs_prover_params1, fs_verifier_params1),
        (fs_prover_params2, fs_verifier_params2),
    ]
}

pub fn build_f_circuit(
    secpar: u8,
    e: u8,
    circuit: &Circuit<F>,
    hasher_prefix: &[F],
) -> ZKBoogieEachVerifierCircuit {
    ZKBoogieEachVerifierCircuit::new(ZKBoogieEachVerifierCircuitParams {
        secpar,
        e,
        circuit: circuit.clone(),
        hasher_prefix: hasher_prefix.to_vec(),
    })
    .unwrap()
}

pub fn fold_proof(
    secpar: u8,
    circuit: &Circuit<F>,
    hasher_prefix: &[F],
    expected_output: &[F],
    proof: &ZKBoogieProof<F, NativeBackend<F, Poseidon254Native>>,
    fs_params: &[(
        ProverParams<G1, G2, KZG<'static, Bn254>, Pedersen<G2>>,
        VerifierParams<G1, G2>,
    ); 3],
) -> Vec<NOVA> {
    let mut novas = vec![];
    for i in 0..3 {
        novas.push(fold_proof_one(
            secpar,
            circuit,
            hasher_prefix,
            expected_output,
            i as u8,
            proof,
            &fs_params[i].0,
            fs_params[i].1.clone(),
        ));
    }
    novas
}

fn fold_proof_one(
    secpar: u8,
    circuit: &Circuit<F>,
    hasher_prefix: &[F],
    expected_output: &[F],
    e: u8,
    proof: &ZKBoogieProof<F, NativeBackend<F, Poseidon254Native>>,
    fs_prover_params: &ProverParams<G1, G2, KZG<'static, Bn254>, Pedersen<G2>>,
    fs_verifier_params: VerifierParams<G1, G2>,
) -> NOVA {
    let n_steps = compute_num_repeat(secpar) as usize;
    let f_circuit = ZKBoogieEachVerifierCircuit::new(ZKBoogieEachVerifierCircuitParams {
        secpar,
        e,
        circuit: circuit.clone(),
        hasher_prefix: hasher_prefix.to_vec(),
    })
    .unwrap();
    // get the CM & CF_CM len
    // let (r1cs, cf_r1cs) = get_r1cs::<G1, GVar, G2, GVar2, FC>(&poseidon_config, f_circuit).unwrap();
    let mut z_0: Vec<Fr> = expected_output.into_iter().map(|v| v.0).collect_vec();
    z_0.resize_with(f_circuit.state_len(), || Fr::zero());
    // let (fs_prover_params, fs_verifier_params, kzg_vk) =
    //     init_nova_ivc_params(r1cs, cf_r1cs, kzg_pk.clone());
    let mut nova = NOVA::init(fs_prover_params, f_circuit.clone(), z_0.clone()).unwrap();
    for i in 0..n_steps {
        let each_proof = &proof.each_proof[i];
        let enable = if each_proof.e == e {
            Fr::one()
        } else {
            Fr::zero()
        };
        let external_inputs = vec![enable]
            .into_iter()
            .chain(
                each_proof
                    .to_field_vec(circuit)
                    .into_iter()
                    .map(|v| v.0)
                    .collect_vec(),
            )
            .collect_vec();
        nova.prove_step(external_inputs).unwrap();
    }

    let (running_instance, incoming_instance, cyclefold_instance) = nova.instances();
    NOVA::verify(
        fs_verifier_params,
        z_0,
        nova.state(), // latest state
        Fr::from(n_steps as u32),
        running_instance,
        incoming_instance,
        cyclefold_instance,
    )
    .unwrap();
    // let rng = rand::rngs::OsRng;
    // let proof = DeciderFCircuit::prove(
    //     (g16_pk, fs_prover_params.cs_params.clone()),
    //     rng,
    //     nova.clone(),
    // )
    // .unwrap();

    // let verified = DeciderFCircuit::verify(
    //     (g16_vk.clone(), kzg_vk.clone()),
    //     nova.i,
    //     nova.z_0.clone(),
    //     nova.z_i.clone(),
    //     &nova.U_i,
    //     &nova.u_i,
    //     &proof,
    // )
    // .unwrap();
    // assert!(verified);
    nova
}

// oritiginal: https://github.com/privacy-scaling-explorations/sonobe/blob/main/examples/utils.rs
pub(crate) fn init_nova_ivc_params<FC: FCircuit<Fr>>(
    f_circuit: FC,
    kzg_pk: KZGProverKey<'static, G1>,
) -> (
    ProverParams<G1, G2, KZG<'static, Bn254>, Pedersen<G2>>,
    VerifierParams<G1, G2>,
) {
    let mut rng = rand::rngs::OsRng;
    let poseidon_config = poseidon_test_config::<Fr>();
    let (r1cs, cf_r1cs) = get_r1cs::<G1, GVar, G2, GVar2, FC>(&poseidon_config, f_circuit).unwrap();
    let cs_len = r1cs.A.n_rows;
    let cf_cs_len = cf_r1cs.A.n_rows;
    // let (kzg_pk, kzg_vk): (KZGProverKey<G1>, KZGVerifierKey<Bn254>) =
    //     KZG::<Bn254>::setup(&mut rng, cs_len).unwrap();
    // let (pedersen_params, _) = Pedersen::<G1>::setup(&mut rng, cf_len).unwrap();
    // let (kzg_pk, kzg_vk): (KZGProverKey<G1>, KZGVerifierKey<Bn254>) =
    //     KZG::<Bn254>::setup(&mut rng, cs_len).unwrap();
    let (cf_pedersen_params, _) = Pedersen::<G2>::setup(&mut rng, cf_cs_len).unwrap();

    let fs_prover_params = ProverParams::<G1, G2, KZG<Bn254>, Pedersen<G2>> {
        poseidon_config: poseidon_config.clone(),
        cs_params: kzg_pk,
        cf_cs_params: cf_pedersen_params,
    };
    let fs_verifier_params = VerifierParams::<G1, G2> {
        poseidon_config: poseidon_config.clone(),
        r1cs,
        cf_r1cs,
    };
    (fs_prover_params, fs_verifier_params)
}

pub(crate) fn init_ivc_and_decider_params<FC: FCircuit<Fr>>(
    nova: &Nova<G1, GVar, G2, GVar2, FC, KZG<'static, Bn254>, Pedersen<G2>>,
) -> (ProvingKey<Bn254>, G16VerifierKey<Bn254>) {
    let mut rng = rand::rngs::OsRng;
    // let start = Instant::now();
    // println!("generated Nova folding params: {:?}", start.elapsed());
    // let z_0 = vec![Fr::zero(); f_circuit.state_len()];
    // let nova = NOVA::init(fs_prover_params, f_circuit, z_0.clone()).unwrap();

    let decider_circuit =
        DeciderEthCircuit::<G1, GVar, G2, GVar2, KZG<Bn254>, Pedersen<G2>>::from_nova::<FC>(
            nova.clone(),
        )
        .unwrap();
    // let start = Instant::now();
    let (g16_pk, g16_vk) =
        Groth16::<Bn254>::circuit_specific_setup(decider_circuit.clone(), &mut rng).unwrap();
    // println!(
    //     "generated G16 (Decider circuit) params: {:?}",
    //     start.elapsed()
    // );
    (g16_pk, g16_vk)
}
