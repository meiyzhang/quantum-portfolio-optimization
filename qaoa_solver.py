"""
qaoa_solver.py

QAOA implementation for the portfolio-selection QUBO, built directly on
Qiskit's modern primitives (SamplerV2/EstimatorV2) and QAOAAnsatz, rather
than the legacy `qiskit_algorithms.QAOA` class. The legacy package expects
the older V1 primitive interface and is incompatible with the V2-primitive
qiskit-aer/qiskit versions used here -- this is documented as a known
ecosystem versioning friction point, not silently worked around.

Pipeline per (B, lambda):
    1. Build QUBO -> Ising Hamiltonian (qubo_formulation.py)
    2. Build a QAOAAnsatz circuit for that Hamiltonian, p layers (reps)
    3. Classical outer-loop optimizer (COBYLA) tunes (beta, gamma) angles
       by repeatedly estimating <H> via EstimatorV2 on AerSimulator
    4. Final sampling (SamplerV2, many shots) at the optimized angles to
       extract a bitstring distribution; the modal (most frequent) and
       best-observed-energy bitstrings are both reported, since QAOA is a
       SAMPLING algorithm -- there is no guarantee the most probable
       bitstring is the lowest-energy one observed across all shots.
"""

import numpy as np
import time
from collections import Counter

from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
from scipy.optimize import minimize

from qubo_formulation import (
    build_qubo_matrix,
    qubo_to_quadratic_program,
    qp_to_ising,
    evaluate_bitstring,
)


def build_qaoa_circuit(qubit_op, reps=2):
    """Build a parameterized QAOAAnsatz circuit for the given Ising operator."""
    ansatz = QAOAAnsatz(cost_operator=qubit_op, reps=reps)
    return ansatz


def run_qaoa(
    qubit_op,
    offset,
    reps=2,
    maxiter=150,
    shots=4096,
    seed=42,
    backend=None,
):
    """
    Run QAOA: classical-quantum hybrid loop to find optimal (beta, gamma),
    then sample the final circuit to get a bitstring distribution.

    Returns a dict with optimization trace, final counts, and timing.
    """
    rng = np.random.default_rng(seed)
    n_qubits = qubit_op.num_qubits

    ansatz = build_qaoa_circuit(qubit_op, reps=reps)

    if backend is None:
        backend = AerSimulator(seed_simulator=seed)

    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    isa_ansatz = pm.run(ansatz)
    isa_op = qubit_op.apply_layout(isa_ansatz.layout)

    estimator = AerEstimatorV2()
    sampler = AerSamplerV2()

    n_params = ansatz.num_parameters
    x0 = rng.uniform(-np.pi / 4, np.pi / 4, size=n_params)

    cost_history = []

    def cost_fn(params):
        pub = (isa_ansatz, [isa_op], [params])
        job = estimator.run([pub])
        result = job.result()[0]
        energy = float(result.data.evs[0])
        cost_history.append(energy)
        return energy

    start = time.time()
    opt_result = minimize(
        cost_fn,
        x0,
        method="COBYLA",
        options={"maxiter": maxiter, "rhobeg": 0.3},
    )
    opt_elapsed = time.time() - start

    # Final sampling at optimized parameters
    final_circuit = isa_ansatz.assign_parameters(opt_result.x)
    final_circuit.measure_all()

    sample_job = sampler.run([(final_circuit,)], shots=shots)
    sample_result = sample_job.result()[0]
    counts = sample_result.data.meas.get_counts()

    return {
        "n_qubits": n_qubits,
        "reps": reps,
        "maxiter": maxiter,
        "shots": shots,
        "optimized_params": opt_result.x.tolist(),
        "final_cost_estimate": float(opt_result.fun) + offset,
        "cost_history": [c + offset for c in cost_history],
        "n_cost_evals": len(cost_history),
        "optimization_elapsed_sec": opt_elapsed,
        "counts": counts,
        "circuit_depth": isa_ansatz.depth(),
        "circuit_size": isa_ansatz.size(),
    }


def analyze_qaoa_counts(counts, mu, sigma, B, q, Q, offset, top_k=10):
    """
    Post-process the QAOA bitstring counts:
      - rank by observed frequency
      - rank by actual QUBO energy among observed bitstrings
      - report the best bitstring found (lowest energy among SAMPLED bits,
        not the full 2^n space -- this is what QAOA, as a heuristic, can
        actually promise)
      - report whether the best-observed bitstring is feasible (k==B)
    """
    total_shots = sum(counts.values())

    # Qiskit bitstrings from measure_all() are little-endian relative to
    # qubit index ordering in some conventions -- handle by checking length
    # matches n and using consistent indexing throughout (qubit i -> char
    # position, reversing if Qiskit's default big-endian-string convention
    # applies). Qiskit counts keys are returned MSB-first as 'q_{n-1}...q_0'.
    n = len(mu)

    scored = []
    for bitstr, freq in counts.items():
        # Reverse to map qiskit's MSB-first convention to asset index 0..n-1
        bits = bitstr[::-1]
        bits = bits.zfill(n)
        w = np.array([int(c) for c in bits])
        energy = float(w @ Q @ w)
        eval_result = evaluate_bitstring(bits, mu, sigma, B, q)
        scored.append({
            "bitstring": bits,
            "frequency": freq,
            "probability": freq / total_shots,
            "qubo_energy": energy,
            "feasible": eval_result["feasible"],
            "eval": eval_result,
        })

    # Sort by frequency (most-sampled first)
    by_freq = sorted(scored, key=lambda r: -r["frequency"])
    # Sort by energy (best-found first), restricted to feasible bitstrings
    feasible_scored = [r for r in scored if r["feasible"]]
    by_energy_feasible = sorted(feasible_scored, key=lambda r: r["qubo_energy"])

    return {
        "total_distinct_bitstrings_sampled": len(scored),
        "total_shots": total_shots,
        "n_feasible_sampled": len(feasible_scored),
        "fraction_feasible_shots": sum(r["frequency"] for r in feasible_scored) / total_shots,
        "top_by_frequency": by_freq[:top_k],
        "top_by_energy_feasible": by_energy_feasible[:top_k],
        "best_feasible_found": by_energy_feasible[0] if by_energy_feasible else None,
        "modal_bitstring": by_freq[0] if by_freq else None,
    }


if __name__ == "__main__":
    mu = np.load("data/mu.npy")
    sigma = np.load("data/sigma.npy")
    n = len(mu)
    B = 4

    print("=" * 70)
    print(f"QAOA TEST RUN: B={B}")
    print("=" * 70)

    Q, lam = build_qubo_matrix(mu, sigma, B=B, q=1.0)
    qp = qubo_to_quadratic_program(Q, n, B=B)
    qubit_op, offset, qubo_qp = qp_to_ising(qp)

    print(f"Qubits: {qubit_op.num_qubits}, Pauli terms: {len(qubit_op)}")

    result = run_qaoa(qubit_op, offset, reps=2, maxiter=150, shots=4096)
    print(f"\nOptimization: {result['n_cost_evals']} cost evals, "
          f"{result['optimization_elapsed_sec']:.2f}s")
    print(f"Circuit depth: {result['circuit_depth']}, size: {result['circuit_size']}")
    print(f"Final estimated cost: {result['final_cost_estimate']:.4f}")

    analysis = analyze_qaoa_counts(result["counts"], mu, sigma, B, 1.0, Q, offset)
    print(f"\nDistinct bitstrings sampled: {analysis['total_distinct_bitstrings_sampled']} "
          f"/ {2**n} possible, out of {analysis['total_shots']} shots")
    print(f"Fraction of shots landing on feasible (k=B) bitstrings: "
          f"{analysis['fraction_feasible_shots']:.4f}")

    best = analysis["best_feasible_found"]
    print(f"\nBest FEASIBLE bitstring found by QAOA: {best['bitstring']} "
          f"(energy={best['qubo_energy']:.4f}, prob={best['probability']:.4f})")
    print(f"  Selected: {best['eval']['selected_indices']}")
    print(f"  Sharpe: {best['eval']['equal_weight_sharpe']:.4f}")

    modal = analysis["modal_bitstring"]
    print(f"\nMost FREQUENTLY sampled bitstring: {modal['bitstring']} "
          f"(prob={modal['probability']:.4f}, feasible={modal['feasible']})")
