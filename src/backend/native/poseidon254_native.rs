use self::finite::{FiniteRing, Fp};

use super::{NativeError, NativeHasher};
use crate::*;
use ark_bn254::{self, Bn254};
use ark_ff::PrimeField;
use arkworks_native_gadgets::poseidon::{sbox::PoseidonSbox, *};
use arkworks_utils::{bytes_matrix_to_f, bytes_vec_to_f, poseidon_params::*, Curve};

#[derive(Debug, Clone, Default)]
pub struct Poseidon254Native {
    inner: Poseidon<ark_bn254::Fr>,
}

impl NativeHasher<Fp<ark_bn254::Fr>> for Poseidon254Native {
    fn new() -> Self {
        let pos_data = setup_poseidon_params(Curve::Bn254, 5, 5).unwrap();
        let mds_f = bytes_matrix_to_f(&pos_data.mds);
        let rounds_f = bytes_vec_to_f(&pos_data.rounds);
        let params = PoseidonParameters {
            mds_matrix: mds_f,
            round_keys: rounds_f,
            full_rounds: pos_data.full_rounds,
            partial_rounds: pos_data.partial_rounds,
            sbox: PoseidonSbox(pos_data.exp),
            width: pos_data.width,
        };
        let inner = Poseidon::new(params);
        Self { inner }
    }

    fn hash(
        &self,
        inputs: &[Fp<ark_bn254::Fr>],
    ) -> Result<Vec<Fp<ark_bn254::Fr>>, super::NativeError> {
        let inputs = inputs.iter().map(|x| x.0).collect::<Vec<_>>();
        let mut last_hash = ark_bn254::Fr::from(0u64);
        for chunk in inputs.chunks(3) {
            let mut input = [&vec![last_hash], chunk].concat();
            input.append(&mut vec![ark_bn254::Fr::from(0u64); 4 - input.len()]);
            last_hash = self
                .inner
                .hash(&input)
                .map_err(|err| NativeError::Poseidon(err))?;
        }
        let output = Fp(last_hash);
        Ok(vec![output])
    }
}
