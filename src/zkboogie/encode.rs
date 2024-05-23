use std::collections::BTreeMap;

use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::{
    native::{NativeBackend, NativeHasher},
    FiniteRing, GateId, ZKBoogieEachProof, ZKBoogieEachProver, ZKBoogieProof,
};

#[derive(Serialize, Deserialize)]
pub struct EncodedZKBoogieEachProof {
    pub e: u8,
    pub e_input_shares: Vec<Vec<u8>>,
    pub e2_input_commit: Vec<u8>,
    pub e1_wire_shares: BTreeMap<GateId, Vec<u8>>,
    // pub e1_view: PlayerState<F, B>,
    pub e2_view_commit: Vec<u8>,
    pub e_rand: BTreeMap<GateId, Vec<u8>>,
    pub e1_rand: BTreeMap<GateId, Vec<u8>>,
    // pub e1_rand: Vec<B::V>,
    pub e2_rand_commit: Vec<u8>,
    pub e2_output_shares: Vec<Vec<u8>>,
    pub transcript_digest: Vec<u8>,
}

impl EncodedZKBoogieEachProof {
    pub fn from_raw<F: FiniteRing, H: NativeHasher<F>>(
        raw: ZKBoogieEachProof<F, NativeBackend<F, H>>,
    ) -> Self {
        let e_input_shares = raw
            .e_input_shares
            .into_iter()
            .map(|fr| fr.to_bytes_le())
            .collect_vec();
        let e2_input_commit = raw.e2_input_commit.to_bytes_le();
        let e1_wire_shares = BTreeMap::from_iter(
            raw.e1_wire_shares
                .into_iter()
                .map(|(k, v)| (k, v.to_bytes_le())),
        );
        let e2_view_commit = raw.e2_view_commit.to_bytes_le();
        let e_rand = BTreeMap::from_iter(raw.e_rand.into_iter().map(|(k, v)| (k, v.to_bytes_le())));
        let e1_rand =
            BTreeMap::from_iter(raw.e1_rand.into_iter().map(|(k, v)| (k, v.to_bytes_le())));
        let e2_rand_commit = raw.e2_rand_commit.to_bytes_le();
        let e2_output_shares = raw
            .e2_output_shares
            .into_iter()
            .map(|fr| fr.to_bytes_le())
            .collect_vec();
        let transcript_digest = raw.transcript_digest.to_bytes_le();
        Self {
            e: raw.e,
            e_input_shares,
            e2_input_commit,
            e1_wire_shares,
            e2_view_commit,
            e_rand,
            e1_rand,
            e2_rand_commit,
            e2_output_shares,
            transcript_digest,
        }
    }

    pub fn to_raw<F: FiniteRing, H: NativeHasher<F>>(
        self,
    ) -> ZKBoogieEachProof<F, NativeBackend<F, H>> {
        let e_input_shares = self
            .e_input_shares
            .into_iter()
            .map(|bytes| F::from_bytes_le(&bytes))
            .collect_vec();
        let e2_input_commit = F::from_bytes_le(&self.e2_input_commit);
        let e1_wire_shares = BTreeMap::from_iter(
            self.e1_wire_shares
                .into_iter()
                .map(|(k, v)| (k, F::from_bytes_le(&v))),
        );
        let e2_view_commit = F::from_bytes_le(&self.e2_view_commit);
        let e_rand = BTreeMap::from_iter(
            self.e_rand
                .into_iter()
                .map(|(k, v)| (k, F::from_bytes_le(&v))),
        );
        let e1_rand = BTreeMap::from_iter(
            self.e1_rand
                .into_iter()
                .map(|(k, v)| (k, F::from_bytes_le(&v))),
        );
        let e2_rand_commit = F::from_bytes_le(&self.e2_rand_commit);
        let e2_output_shares = self
            .e2_output_shares
            .into_iter()
            .map(|bytes| F::from_bytes_le(&bytes))
            .collect_vec();
        let transcript_digest = F::from_bytes_le(&self.transcript_digest);
        ZKBoogieEachProof {
            e: self.e,
            e_input_shares,
            e2_input_commit,
            e1_wire_shares,
            e2_view_commit,
            e_rand,
            e1_rand,
            e2_rand_commit,
            e2_output_shares,
            transcript_digest,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct EncodedZKBoogieProof {
    pub each_proof: Vec<EncodedZKBoogieEachProof>,
}

impl EncodedZKBoogieProof {
    pub fn from_raw<F: FiniteRing, H: NativeHasher<F>>(
        raw: ZKBoogieProof<F, NativeBackend<F, H>>,
    ) -> Self {
        let each_proof = raw
            .each_proof
            .into_iter()
            .map(EncodedZKBoogieEachProof::from_raw)
            .collect_vec();
        Self { each_proof }
    }

    pub fn to_raw<F: FiniteRing, H: NativeHasher<F>>(
        self,
    ) -> ZKBoogieProof<F, NativeBackend<F, H>> {
        let each_proof = self
            .each_proof
            .into_iter()
            .map(EncodedZKBoogieEachProof::to_raw)
            .collect_vec();
        ZKBoogieProof { each_proof }
    }
}
