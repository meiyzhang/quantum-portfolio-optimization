"""
classical_solvers.py

Two classical reference solvers for the discrete B-of-N selection QUBO:

1. Brute force exact enumeration: with N=11, there are 2^11 = 2048 total
   bitstrings -- fully tractable to enumerate exhaustively. This gives the
   TRUE global optimum for every B, which is the ground truth QAOA is
   benchmarked against (not "did QAOA beat classical" but "how close did
   QAOA get to the actually-correct answer").

2. Classical simulated annealing: a standard metaheuristic, included as the
   realistic classical competitor at problem sizes where brute force stops
   being tractable (irrelevant at N=11, but this is the comparison that
   matters at N=30, 50, 100+ -- included here to establish the benchmarking
   methodology even though brute force already solves N=11 exactly).
"""

import numpy as np
import itertools
import time

from qubo_formulation import build_qubo_matrix, evaluate_bitstring
from data_loader import TICKERS


def qubo_energy(bitstring, Q):
    """Compute w^T Q w for a binary vector w (includes the soft penalty)."""
    w = np.array(bitstring, dtype=float)
    return float(w @ Q @ w)


def brute_force_exact(Q, n, B, mu, sigma, q=1.0):
    """
    Exhaustively enumerate all 2^n bitstrings, find the one with lowest QUBO
    energy AMONG those satisfying the cardinality constraint exactly
    (we report both: the unconstrained QUBO argmin, and the best FEASIBLE
    bitstring, since the penalty is soft and could in principle be beaten by
    an infeasible solution if lambda were mistuned -- checking this is
    itself part of validating the penalty weight).
    """
    start = time.time()
    best_energy_overall = np.inf
    best_bits_overall = None

    best_energy_feasible = np.inf
    best_bits_feasible = None

    n_feasible = 0
    for bits in itertools.product([0, 1], repeat=n):
        energy = qubo_energy(bits, Q)
        if energy < best_energy_overall:
            best_energy_overall = energy
            best_bits_overall = bits

        if sum(bits) == B:
            n_feasible += 1
            if energy < best_energy_feasible:
                best_energy_feasible = energy
                best_bits_feasible = bits

    elapsed = time.time() - start

    bitstring_overall = "".join(str(b) for b in best_bits_overall)
    bitstring_feasible = "".join(str(b) for b in best_bits_feasible)

    overall_eval = evaluate_bitstring(bitstring_overall, mu, sigma, B, q)
    feasible_eval = evaluate_bitstring(bitstring_feasible, mu, sigma, B, q)

    return {
        "method": "brute_force_exact",
        "n_enumerated": 2 ** n,
        "n_feasible_found": n_feasible,
        "elapsed_sec": elapsed,
        "unconstrained_argmin": {
            "bitstring": bitstring_overall,
            "qubo_energy": best_energy_overall,
            "is_feasible": overall_eval["feasible"],
            "eval": overall_eval,
        },
        "best_feasible": {
            "bitstring": bitstring_feasible,
            "qubo_energy": best_energy_feasible,
            "eval": feasible_eval,
        },
        "penalty_consistent": (bitstring_overall == bitstring_feasible),
    }


def simulated_annealing(
    Q, n, B, mu, sigma, q=1.0,
    n_restarts=20, n_iters=2000, T_start=10.0, T_end=0.01, seed=42,
):
    """
    Classical simulated annealing over binary strings, minimizing the same
    QUBO energy. Multiple random restarts with geometric cooling schedule.
    Bit-flip neighborhood (flip one random bit per step).
    """
    rng = np.random.default_rng(seed)
    start = time.time()

    best_overall_energy = np.inf
    best_overall_bits = None

    for restart in range(n_restarts):
        # Random initial bitstring with exactly B ones (start feasible)
        bits = np.zeros(n, dtype=int)
        ones_idx = rng.choice(n, size=B, replace=False)
        bits[ones_idx] = 1

        current_energy = qubo_energy(bits, Q)
        best_local_energy = current_energy
        best_local_bits = bits.copy()

        for it in range(n_iters):
            T = T_start * (T_end / T_start) ** (it / n_iters)

            candidate = bits.copy()
            flip_idx = rng.integers(0, n)
            candidate[flip_idx] = 1 - candidate[flip_idx]

            candidate_energy = qubo_energy(candidate, Q)
            delta = candidate_energy - current_energy

            if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-9)):
                bits = candidate
                current_energy = candidate_energy

                if current_energy < best_local_energy:
                    best_local_energy = current_energy
                    best_local_bits = bits.copy()

        if best_local_energy < best_overall_energy:
            best_overall_energy = best_local_energy
            best_overall_bits = best_local_bits

    elapsed = time.time() - start
    bitstring = "".join(str(b) for b in best_overall_bits)
    eval_result = evaluate_bitstring(bitstring, mu, sigma, B, q)

    return {
        "method": "simulated_annealing",
        "n_restarts": n_restarts,
        "n_iters_per_restart": n_iters,
        "elapsed_sec": elapsed,
        "bitstring": bitstring,
        "qubo_energy": best_overall_energy,
        "eval": eval_result,
    }


if __name__ == "__main__":
    mu = np.load("data/mu.npy")
    sigma = np.load("data/sigma.npy")
    n = len(mu)

    print("=" * 70)
    print("CLASSICAL SOLVERS: BRUTE FORCE + SIMULATED ANNEALING")
    print("=" * 70)

    all_results = {}
    for B in [3, 4, 5, 6, 7]:
        print(f"\n--- B = {B} ---")
        Q, lam = build_qubo_matrix(mu, sigma, B=B, q=1.0)

        bf = brute_force_exact(Q, n, B, mu, sigma, q=1.0)
        print(f"Brute force ({bf['n_enumerated']} enumerated, {bf['elapsed_sec']:.2f}s):")
        print(f"  Best feasible bitstring: {bf['best_feasible']['bitstring']} "
              f"(energy={bf['best_feasible']['qubo_energy']:.4f})")
        print(f"  Penalty-consistent (unconstrained argmin == best feasible)? "
              f"{bf['penalty_consistent']}")
        selected_tickers = [TICKERS[i] for i in bf['best_feasible']['eval']['selected_indices']]
        print(f"  Selected: {selected_tickers}")
        print(f"  Equal-weight return={bf['best_feasible']['eval']['equal_weight_return']:.4f}, "
              f"risk={bf['best_feasible']['eval']['equal_weight_risk']:.4f}, "
              f"sharpe={bf['best_feasible']['eval']['equal_weight_sharpe']:.4f}")

        sa = simulated_annealing(Q, n, B, mu, sigma, q=1.0)
        print(f"Simulated annealing ({sa['elapsed_sec']:.3f}s):")
        print(f"  Bitstring: {sa['bitstring']} (energy={sa['qubo_energy']:.4f})")
        print(f"  Matches brute-force optimum? {sa['bitstring'] == bf['best_feasible']['bitstring']}")

        all_results[B] = {"brute_force": bf, "simulated_annealing": sa, "lambda": lam}

    np.save("results/classical_results.npy", all_results, allow_pickle=True)
    print("\nSaved classical benchmark results to results/classical_results.npy")
