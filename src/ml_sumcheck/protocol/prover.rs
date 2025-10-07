//! Prover
use crate::ml_sumcheck::data_structures::BinaryConstraintPolynomial;
use crate::ml_sumcheck::protocol::verifier::VerifierMsg;
use crate::ml_sumcheck::protocol::IPForMLSumcheck;
use ark_ff::Field;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_serialize::CanonicalSerialize;
use ark_std::{cfg_iter_mut, vec::Vec};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Prover Message
#[derive(Clone, CanonicalSerialize)]
pub struct ProverMsg<F: Field> {
    /// evaluations on P(0), P(1), P(2), ... 
    pub(crate) evaluations: Vec<F>,
}

/// Prover State for binary constraints with eq_t masking
pub struct ProverState<F: Field> {
    /// sampled randomness given by the verifier
    pub randomness: Vec<F>,
    /// List of (coefficient, polynomial) pairs
    pub constraints: Vec<(F, DenseMultilinearExtension<F>)>,
    /// The eq_t point (original, never modified)
    pub eq_point_original: Vec<F>,
    /// Number of variables
    pub num_vars: usize,
    /// The current round number
    pub round: usize,
}

impl<F: Field> IPForMLSumcheck<F> {
    /// Initialize the prover for binary constraint polynomial with eq masking
    pub fn prover_init(polynomial: &BinaryConstraintPolynomial<F>) -> ProverState<F> {
        if polynomial.num_variables == 0 {
            panic!("Attempt to prove a constant.");
        }

        // Clone all polynomials
        let constraints = polynomial
            .constraints
            .iter()
            .map(|(c, p)| (*c, p.clone()))
            .collect();

        ProverState {
            randomness: Vec::with_capacity(polynomial.num_variables),
            constraints,
            eq_point_original: polynomial.eq_point.clone(),
            num_vars: polynomial.num_variables,
            round: 0,
        }
    }

    /// Receive message from verifier, generate prover message, and proceed to next round
    pub fn prove_round(
        prover_state: &mut ProverState<F>,
        v_msg: &Option<VerifierMsg<F>>,
    ) -> ProverMsg<F> {
        if let Some(msg) = v_msg {
            if prover_state.round == 0 {
                panic!("first round should be prover first.");
            }
            prover_state.randomness.push(msg.randomness);

            // Fix variables in all polynomials
            let r = prover_state.randomness[prover_state.round - 1];
            cfg_iter_mut!(prover_state.constraints).for_each(|(_, poly)| {
                *poly = poly.fix_variables(&[r]);
            });
        } else if prover_state.round > 0 {
            panic!("verifier message is empty");
        }

        prover_state.round += 1;

        if prover_state.round > prover_state.num_vars {
            panic!("Prover is not active");
        }

        let i = prover_state.round;
        let nv = prover_state.num_vars;
        
        // Conservative degree upper bound
        let remaining_vars = nv - i + 1;
        // Degree is always 4 for [P(1-P)] * eq_t * (1 - eq_{1,1})
        // - P(1-P): degree 2
        // - eq_t: degree 1  
        // - (1 - eq_{1,1}): degree 1
        // - Product: 2 * 1 * 1 but we need to multiply, so 2 + 1 + 1 = 4
        let degree = 4;
        #[cfg(not(feature = "parallel"))]
        let zeros = vec![F::zero(); degree + 1];
        #[cfg(feature = "parallel")]
        let zeros = || vec![F::zero(); degree + 1];

        let fold_result = ark_std::cfg_into_iter!(0..1 << (nv - i), 1 << 10).fold(
            zeros,
            |mut sum, b| {
                // For each constraint cᵢ·Pᵢ(1-Pᵢ)
                for (coefficient, poly) in &prover_state.constraints {
                    let p0 = poly[b << 1];
                    let p1 = poly[(b << 1) + 1];

                    // P(X) = p0 + X(p1 - p0)
                    let delta = p1 - p0;

                    // P(X)(1-P(X)) coefficients: a0 + a1·X + a2·X²
                    let one = F::one();
                    let two = F::from(2u64);
                    let a0 = p0 * (one - p0);
                    let a1 = delta * (one - two * p0);
                    let a2 = -(delta * delta);

                    // Evaluate at X = 0, 1, 2, ...
                    for x in 0..=degree {
                        let x_field = F::from(x as u64);
                        
                        // Binary constraint at X
                        let binary_val = a0 + a1 * x_field + a2 * x_field * x_field;
                        
                        // Now compute eq_t and eq_{1,...,1} at (r_1,...,r_{i-1}, X, x_{i+1},...,x_n)
                        // where x_{i+1},...,x_n are determined by b
                        
                        // eq_t contribution
                        let mut eq_val = one;
                        
                        // Fixed variables (rounds 1 to i-1)
                        for j in 0..i-1 {
                            let tj = prover_state.eq_point_original[j];
                            let rj = prover_state.randomness[j];
                            eq_val *= (one - tj) + rj * (two * tj - one);
                        }
                        
                        // Current variable X (round i)
                        let ti = prover_state.eq_point_original[i - 1];
                        eq_val *= (one - ti) + x_field * (two * ti - one);
                        
                        // Remaining variables (rounds i+1 to nv)
                        for j in 0..(nv - i) {
                            let tj = prover_state.eq_point_original[i + j];
                            let xj = if (b >> j) & 1 == 1 { one } else { F::zero() };
                            eq_val *= (one - tj) + xj * (two * tj - one);
                        }
                        
                        // eq_{1,...,1} contribution
                        let mut eq_ones_val = one;
                        
                        // Fixed variables
                        for j in 0..i-1 {
                            eq_ones_val *= prover_state.randomness[j];
                        }
                        
                        // Current variable X
                        eq_ones_val *= x_field;
                        
                        // Remaining variables
                        for j in 0..(nv - i) {
                            let xj = if (b >> j) & 1 == 1 { one } else { F::zero() };
                            eq_ones_val *= xj;
                        }
                        
                        // Full product: binary_val * eq_val * (1 - eq_ones_val)
                        let val = binary_val * eq_val * (one - eq_ones_val);
                        
                        sum[x] += *coefficient * val;
                    }
                }
                sum
            },
        );

        #[cfg(not(feature = "parallel"))]
        let products_sum = fold_result;

        #[cfg(feature = "parallel")]
        let products_sum = fold_result.reduce(
            || vec![F::zero(); degree + 1],
            |mut overall, sublist| {
                overall
                    .iter_mut()
                    .zip(sublist.iter())
                    .for_each(|(f, s)| *f += s);
                overall
            },
        );

        ProverMsg {
            evaluations: products_sum,
        }
    }
}