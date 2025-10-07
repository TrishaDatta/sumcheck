//! Sumcheck protocol implementation

use crate::ml_sumcheck::data_structures::{BinaryConstraintPolynomial, PolynomialInfo};
use crate::ml_sumcheck::protocol::prover::{ProverMsg, ProverState};
use crate::ml_sumcheck::protocol::verifier::SubClaim;
use crate::ml_sumcheck::protocol::IPForMLSumcheck;
use crate::rng::FeedableRNG;
use crate::Error;
use ark_ff::Field;
use ark_serialize::CanonicalSerialize;
use ark_std::marker::PhantomData;
use ark_std::vec::Vec;

pub mod data_structures;
pub mod protocol;
#[cfg(test)]
mod test;


pub struct MLSumcheck<F: Field>(#[doc(hidden)] PhantomData<F>);

impl<F: Field> MLSumcheck<F> {
    /// Initialize Fiat-Shamir RNG
    fn init_rng() -> impl FeedableRNG<Error = Error> {
        crate::rng::Blake2b512Rng::setup()
    }

    /// Non-interactive prove using Fiat-Shamir
    pub fn prove(
        polynomial: &BinaryConstraintPolynomial<F>,
    ) -> Result<Vec<ProverMsg<F>>, Error> {
        let mut fs_rng = Self::init_rng();
        Self::prove_as_subprotocol(&mut fs_rng, polynomial).map(|(proof, _)| proof)
    }

    /// Non-interactive verify using Fiat-Shamir
    pub fn verify(
        polynomial_info: &PolynomialInfo,
        asserted_sum: F,
        proof: &[ProverMsg<F>],
    ) -> Result<SubClaim<F>, Error> {
        let mut fs_rng = Self::init_rng();
        Self::verify_as_subprotocol(&mut fs_rng, polynomial_info, asserted_sum, proof)
    }

    /// Prove as subprotocol with external RNG
    pub fn prove_as_subprotocol<RNG: FeedableRNG<Error = Error>>(
        fs_rng: &mut RNG,
        polynomial: &BinaryConstraintPolynomial<F>,
    ) -> Result<(Vec<ProverMsg<F>>, ProverState<F>), Error> {
        let mut prover_state = IPForMLSumcheck::prover_init(polynomial);
        let mut prover_messages = Vec::with_capacity(polynomial.num_variables);
        let mut verifier_msg = None;

        for _ in 0..polynomial.num_variables {
            let prover_msg = IPForMLSumcheck::prove_round(&mut prover_state, &verifier_msg);
            
            // Feed prover message to Fiat-Shamir RNG
            Self::feed_prover_msg(fs_rng, &prover_msg)?;
            
            verifier_msg = Some(IPForMLSumcheck::sample_round(fs_rng));
            prover_messages.push(prover_msg);
        }

        Ok((prover_messages, prover_state))
    }

    /// Verify as subprotocol with external RNG
    pub fn verify_as_subprotocol<RNG: FeedableRNG<Error = Error>>(
        fs_rng: &mut RNG,
        polynomial_info: &PolynomialInfo,
        claimed_sum: F,
        proof: &[ProverMsg<F>],
    ) -> Result<SubClaim<F>, Error> {
        let mut verifier_state = IPForMLSumcheck::verifier_init(polynomial_info);

        for prover_msg in proof.iter() {
            // Feed prover message to Fiat-Shamir RNG
            Self::feed_prover_msg(fs_rng, prover_msg)?;
            
            IPForMLSumcheck::verify_round((*prover_msg).clone(), &mut verifier_state, fs_rng);
        }

        IPForMLSumcheck::check_and_generate_subclaim(verifier_state, claimed_sum)
    }

    /// Extract sum from proof (first message contains g(0) + g(1))
    pub fn extract_sum(proof: &[ProverMsg<F>]) -> F {
        proof[0].evaluations[0] + proof[0].evaluations[1]
    }

    /// Helper to feed prover message to RNG
    fn feed_prover_msg<RNG: FeedableRNG<Error = Error>>(
        rng: &mut RNG,
        msg: &ProverMsg<F>,
    ) -> Result<(), Error> {
        let mut buf = Vec::new();
        msg.serialize_compressed(&mut buf)
            .map_err(|_| Error::SerializationError)?;
        rng.feed(&buf)
    }
}