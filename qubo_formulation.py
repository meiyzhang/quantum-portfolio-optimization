"""
qubo_formulation.py

Formulates the discrete "select exactly B of N assets" portfolio problem as
a QUBO (Quadratic Unconstrained Binary Optimization), then converts it to an
Ising Hamiltonian suitable for QAOA.

This is a DIFFERENT problem than continuous Markowitz: here each w_i in {0,1}
is a hard include/exclude decision (equal-weighted among selected assets, or
unweighted binary selection objective), not a continuous capital allocation.
We are not solving the same QP with a quantum trick -- we are solving the
NP-hard discrete subset-selection variant, for which QAOA is a legitimate
(if NISQ-constrained) candidate algorithm.

QUBO objective (binary w in {0,1}^n):

    H(w) = w^T Sigma w  -  q * mu^T w  +  lambda * (sum(w) - B)^2

  - w^T Sigma w        : portfolio variance proxy among selected assets
                         (equal-weight assumption: each selected asset
                         contributes 1/B notionally, but for QUBO/QAOA
                         purposes we keep raw binary indicator variables
                         and rely on the penalty to enforce cardinality;
                         risk/return are then evaluated post-hoc at 1/B
                         equal weights for the selected set)
  - q * mu^T w          : expected-return reward term, q = risk-aversion
                         trade-off parameter (higher q -> more return-seeking)
  - lambda * (sum(w)-B)^2 : penalty enforcing exactly B assets selected.
                         lambda must be large enough that violating the
                         budget constraint is never favorable, but not so
                         large it swamps the risk/return landscape and
                         flattens the optimization surface for QAOA.
"""

import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo


def build_qubo_matrix(mu, sigma, B, q=1.0, lam=None, lam_multiplier=3.0):
    """
    Build the QUBO coefficient matrix Q (n x n, upper-triangular-folded into
    symmetric form is handled by convention: we return the full symmetric
    matrix such that H(w) = w^T Q w + linear terms folded into the diagonal).

    Returns:
        Q : (n,n) numpy array, symmetric, such that objective = w^T Q w
        lam_used : the penalty coefficient actually used
    """
    n = len(mu)

    if lam is None:
        # Heuristic: penalty should dominate the largest plausible swing in
        # the risk/return terms from a single bit flip. Scale off the
        # largest eigenvalue of Sigma and the return spread, with a safety
        # multiplier. This is exactly the "penalty tuning sensitivity"
        # limitation flagged in the project scope -- documented here rather
        # than hidden.
        sigma_scale = np.max(np.abs(sigma))
        mu_scale = np.max(np.abs(mu)) * q
        lam = lam_multiplier * max(sigma_scale, mu_scale, 1e-6) * B

    # Start with the risk term: w^T Sigma w
    Q = sigma.copy().astype(float)

    # Return term: -q * mu^T w  ->  diagonal contribution
    for i in range(n):
        Q[i, i] += -q * mu[i]

    # Penalty term: lambda * (sum(w) - B)^2
    #   = lambda * [ sum_i w_i^2 + 2*sum_{i<j} w_i w_j - 2B*sum_i w_i + B^2 ]
    #   (w_i^2 = w_i since binary)
    # Diagonal: lambda * (1 - 2B)
    # Off-diagonal (i != j): 2*lambda
    # Constant B^2 * lambda is dropped (doesn't affect argmin)
    for i in range(n):
        Q[i, i] += lam * (1 - 2 * B)
        for j in range(n):
            if i != j:
                Q[i, j] += lam  # contributes once per (i,j) and (j,i) -> matches 2*lambda total when symmetrized

    return Q, lam


def qubo_to_quadratic_program(Q, n, B=None):
    """
    Wrap a raw QUBO matrix Q into a Qiskit QuadraticProgram object (binary
    vars x_0..x_{n-1}), objective = x^T Q x. No additional constraints are
    added here -- the cardinality constraint is already baked into Q as a
    penalty term (soft constraint), consistent with the QAOA workflow where
    constraints must be unconstrained-ified before Ising mapping anyway.
    """
    qp = QuadraticProgram(name=f"portfolio_select_B{B}" if B else "portfolio_select")
    n_vars = Q.shape[0]
    for i in range(n_vars):
        qp.binary_var(name=f"x{i}")

    linear = {f"x{i}": float(Q[i, i]) for i in range(n_vars)}
    quadratic = {}
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            coeff = float(Q[i, j] + Q[j, i])  # combine symmetric off-diagonal pair
            if abs(coeff) > 1e-12:
                quadratic[(f"x{i}", f"x{j}")] = coeff

    qp.minimize(linear=linear, quadratic=quadratic)
    return qp


def qp_to_ising(qp):
    """
    Convert a QuadraticProgram (already QUBO-form, unconstrained) into an
    Ising Hamiltonian (SparsePauliOp) + offset, via Qiskit's built-in
    converter. Returns (qubit_op, offset).
    """
    converter = QuadraticProgramToQubo()
    qubo_qp = converter.convert(qp)  # no-op if already QUBO, but normalizes form
    qubit_op, offset = qubo_qp.to_ising()
    return qubit_op, offset, qubo_qp


def evaluate_bitstring(bitstring_or_array, mu, sigma, B, q=1.0):
    """
    Given a binary selection (as a string '01101...' or array), compute the
    EQUAL-WEIGHT portfolio's actual return, risk, and Sharpe for the
    selected subset (post-hoc evaluation, decoupled from QUBO penalty
    artifacts). Also reports whether the cardinality constraint was
    actually satisfied (i.e. whether exactly B assets were selected) --
    a feasibility check independent of the soft-penalty QUBO score.
    """
    if isinstance(bitstring_or_array, str):
        w_bin = np.array([int(c) for c in bitstring_or_array])
    else:
        w_bin = np.array(bitstring_or_array)

    k_selected = int(np.sum(w_bin))
    feasible = (k_selected == B)

    if k_selected == 0:
        return {
            "selected_indices": [],
            "k_selected": 0,
            "feasible": feasible,
            "equal_weight_return": None,
            "equal_weight_risk": None,
            "equal_weight_sharpe": None,
        }

    w_equal = w_bin / k_selected  # equal weight among selected
    ret = mu @ w_equal
    risk = np.sqrt(max(w_equal @ sigma @ w_equal, 0))
    sharpe = ret / risk if risk > 1e-12 else np.nan

    return {
        "selected_indices": list(np.where(w_bin == 1)[0]),
        "k_selected": k_selected,
        "feasible": feasible,
        "equal_weight_return": float(ret),
        "equal_weight_risk": float(risk),
        "equal_weight_sharpe": float(sharpe),
    }


if __name__ == "__main__":
    from data_loader import TICKERS

    mu = np.load("data/mu.npy")
    sigma = np.load("data/sigma.npy")
    n = len(mu)

    print("=" * 70)
    print("QUBO FORMULATION SANITY CHECK")
    print("=" * 70)

    for B in [3, 4, 5, 6, 7]:
        Q, lam = build_qubo_matrix(mu, sigma, B=B, q=1.0)
        print(f"\nB={B}: penalty lambda = {lam:.4f}, Q matrix shape = {Q.shape}, "
              f"Q symmetric = {np.allclose(Q, Q.T)}")

        qp = qubo_to_quadratic_program(Q, n, B=B)
        qubit_op, offset, qubo_qp = qp_to_ising(qp)
        print(f"  Ising Hamiltonian: {qubit_op.num_qubits} qubits, "
              f"{len(qubit_op)} Pauli terms, offset={offset:.4f}")
