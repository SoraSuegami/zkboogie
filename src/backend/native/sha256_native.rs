use std::marker::PhantomData;

use self::finite::{FiniteRing, F256};

use super::{NativeError, NativeHasher};
use crate::*;
use itertools::Itertools;
use sha2::{Digest, Sha256, Sha512};

#[derive(Debug, Clone, Default)]
pub struct Sha256Native<F: FiniteRing> {
    _f: PhantomData<F>,
}

impl<F: FiniteRing> NativeHasher<F> for Sha256Native<F> {
    fn new() -> Self {
        Self { _f: PhantomData }
    }

    fn hash(&self, inputs: &[F]) -> Result<Vec<F>, NativeError> {
        let input_bytes = inputs
            .iter()
            .flat_map(|x| x.to_bytes_le())
            .collect::<Vec<_>>();
        let mut hasher = Sha256::new();
        hasher.update(input_bytes);
        let result = hasher.finalize();
        let num_bytes_per_field = (F::modulo_bits_size() / 8) as usize;
        let fields = result
            .chunks(num_bytes_per_field)
            .map(|bytes| F::from_bytes_le(bytes))
            .collect_vec();
        Ok(fields)
    }
}
