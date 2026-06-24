"""
run_full_benchmark.py

Main benchmark driver. For each budget B in {3,4,5,6,7}:
  1. Build QUBO -> Ising
  2. Solve via brute force (exact ground truth, N=11 fully enumerable)
  3. Solve via classical simulated annealing
  4. Solve via QAOA (multi-start: several random initializations, reps=2,
     report the best-converged run) on Aer simulator
  5. Compare: QUBO energy gap to optimum, Sharpe ratio of selected portfolio,
     feasible-shot fraction, whether QAOA's sampled optimum matches brute
     force, runtime

Saves a full results dictionary + prints a summary table.
"""

import numpy as np
import time
import json

from data_loader import TICKERS
from qubo_formulation import build_qubo_matrix, qubo_to_quadratic_program, qp_to_ising
from classical_solvers import brute_force_exact, simulated_annealing
from qaoa_solver import run_qaoa, analyze_qaoa_counts


def run_qaoa_multistart(qubit_op, offset, Q, mu, sigma, B, q=1.0,
                          reps=2, maxiter=200, shots=4096, n_starts=5, base_seed=100):
    """
    Run QAOA from n_starts random initializations, keep the run with the
    lowest converged estimator cost, and return its full analysis alongside
    aggregate statistics across all starts (for reporting variance, not just
    a single cherry-picked run).
    """
    runs = []
    start_time = time.time()
    for i in range(n_starts):
        seed = base_seed + i
        result = run_qaoa(qubit_op, offset, reps=reps, maxiter=maxiter, shots=shots, seed=seed)
        analysis = analyze_qaoa_counts(result["counts"], mu, sigma, B, q, Q, offset)
        runs.append({"seed": seed, "result": result, "analysis": analysis})
    elapsed = time.time() - start_time

    best_run = min(runs, key=lambda r: r["result"]["final_cost_estimate"])

    feasible_fracs = [r["analysis"]["fraction_feasible_shots"] for r in runs]
    final_costs = [r["result"]["final_cost_estimate"] for r in runs]

    return {
        "n_starts": n_starts,
        "total_elapsed_sec": elapsed,
        "best_run": best_run,
        "all_final_costs": final_costs,
        "all_feasible_fracs": feasible_fracs,
        "mean_feasible_frac": float(np.mean(feasible_fracs)),
        "std_feasible_frac": float(np.std(feasible_fracs)),
        "best_final_cost": float(min(final_costs)),
        "circuit_depth": best_run["result"]["circuit_depth"],
        "circuit_size": best_run["result"]["circuit_size"],
    }


def main():
    mu = np.load("data/mu.npy")
    sigma = np.load("data/sigma.npy")
    n = len(mu)
    q_riskaversion = 1.0
    budgets = [3, 4, 5, 6, 7]

    all_results = {}

    print("=" * 100)
    print("FULL BENCHMARK: BRUTE FORCE vs SIMULATED ANNEALING vs QAOA (multi-start)")
    print("=" * 100)

    for B in budgets:
        print(f"\n{'='*100}\nB = {B}\n{'='*100}")

        Q, lam = build_qubo_matrix(mu, sigma, B=B, q=q_riskaversion)
        qp = qubo_to_quadratic_program(Q, n, B=B)
        qubit_op, offset, qubo_qp = qp_to_ising(qp)

        # --- Brute force ---
        bf = brute_force_exact(Q, n, B, mu, sigma, q=q_riskaversion)
        optimal_bitstring = bf["best_feasible"]["bitstring"]
        optimal_energy = bf["best_feasible"]["qubo_energy"]
        optimal_sharpe = bf["best_feasible"]["eval"]["equal_weight_sharpe"]
        optimal_tickers = [TICKERS[i] for i in bf["best_feasible"]["eval"]["selected_indices"]]

        print(f"[Brute force] optimum: {optimal_bitstring} energy={optimal_energy:.4f} "
              f"sharpe={optimal_sharpe:.4f} tickers={optimal_tickers} "
              f"({bf['elapsed_sec']:.3f}s)")

        # --- Simulated annealing ---
        sa = simulated_annealing(Q, n, B, mu, sigma, q=q_riskaversion)
        sa_matches_optimal = (sa["bitstring"] == optimal_bitstring)
        print(f"[Sim. annealing] {sa['bitstring']} energy={sa['qubo_energy']:.4f} "
              f"matches_optimal={sa_matches_optimal} ({sa['elapsed_sec']:.3f}s)")

        # --- QAOA (multi-start) ---
        qaoa_multi = run_qaoa_multistart(
            qubit_op, offset, Q, mu, sigma, B, q=q_riskaversion,
            reps=2, maxiter=200, shots=4096, n_starts=5, base_seed=100 + B * 10,
        )
        best_qaoa_run = qaoa_multi["best_run"]
        best_qaoa_feasible = best_qaoa_run["analysis"]["best_feasible_found"]
        qaoa_matches_optimal = (best_qaoa_feasible["bitstring"] == optimal_bitstring) if best_qaoa_feasible else False

        print(f"[QAOA multi-start, n={qaoa_multi['n_starts']}] "
              f"best converged cost={qaoa_multi['best_final_cost']:.4f}  "
              f"mean_feasible_frac={qaoa_multi['mean_feasible_frac']:.4f}±{qaoa_multi['std_feasible_frac']:.4f}  "
              f"depth={qaoa_multi['circuit_depth']}  "
              f"({qaoa_multi['total_elapsed_sec']:.2f}s total)")
        if best_qaoa_feasible:
            print(f"    Best feasible bitstring found: {best_qaoa_feasible['bitstring']} "
                  f"energy={best_qaoa_feasible['qubo_energy']:.4f} "
                  f"prob={best_qaoa_feasible['probability']:.4f}  "
                  f"matches_optimal={qaoa_matches_optimal}")
        else:
            print("    No feasible bitstring sampled in best run.")

        energy_gap = (
            (best_qaoa_feasible["qubo_energy"] - optimal_energy) if best_qaoa_feasible else None
        )

        all_results[B] = {
            "optimal_bitstring": optimal_bitstring,
            "optimal_energy": optimal_energy,
            "optimal_sharpe": optimal_sharpe,
            "optimal_tickers": optimal_tickers,
            "lambda": lam,
            "brute_force": bf,
            "simulated_annealing": {
                "bitstring": sa["bitstring"],
                "energy": sa["qubo_energy"],
                "matches_optimal": sa_matches_optimal,
                "elapsed_sec": sa["elapsed_sec"],
            },
            "qaoa": {
                "n_starts": qaoa_multi["n_starts"],
                "best_final_cost": qaoa_multi["best_final_cost"],
                "mean_feasible_frac": qaoa_multi["mean_feasible_frac"],
                "std_feasible_frac": qaoa_multi["std_feasible_frac"],
                "circuit_depth": qaoa_multi["circuit_depth"],
                "circuit_size": qaoa_multi["circuit_size"],
                "best_feasible_bitstring": best_qaoa_feasible["bitstring"] if best_qaoa_feasible else None,
                "best_feasible_energy": best_qaoa_feasible["qubo_energy"] if best_qaoa_feasible else None,
                "best_feasible_probability": best_qaoa_feasible["probability"] if best_qaoa_feasible else None,
                "matches_optimal": qaoa_matches_optimal,
                "energy_gap_to_optimal": energy_gap,
                "elapsed_sec": qaoa_multi["total_elapsed_sec"],
            },
        }

    # --- Summary table ---
    print("\n\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    header = f"{'B':>3} | {'Optimal':>13} | {'SA match':>9} | {'QAOA match':>10} | {'QAOA gap':>10} | {'QAOA feas%':>10} | {'Sharpe*':>8}"
    print(header)
    print("-" * len(header))
    for B in budgets:
        r = all_results[B]
        gap_str = f"{r['qaoa']['energy_gap_to_optimal']:.3f}" if r['qaoa']['energy_gap_to_optimal'] is not None else "N/A"
        print(f"{B:>3} | {r['optimal_bitstring']:>13} | "
              f"{str(r['simulated_annealing']['matches_optimal']):>9} | "
              f"{str(r['qaoa']['matches_optimal']):>10} | "
              f"{gap_str:>10} | "
              f"{r['qaoa']['mean_feasible_frac']*100:>9.1f}% | "
              f"{r['optimal_sharpe']:>8.3f}")
    print("\n(*Sharpe = equal-weight Sharpe ratio of the BRUTE-FORCE optimal subset, i.e. the")
    print(" best achievable Sharpe for that B under the equal-weight assumption.)")

    np.save("results/full_benchmark_results.npy", all_results, allow_pickle=True)

    json_summary = {}
    for B in budgets:
        r = all_results[B]
        json_summary[str(B)] = {
            "optimal_bitstring": r["optimal_bitstring"],
            "optimal_tickers": r["optimal_tickers"],
            "optimal_sharpe": r["optimal_sharpe"],
            "lambda": r["lambda"],
            "sa_matches_optimal": r["simulated_annealing"]["matches_optimal"],
            "qaoa_matches_optimal": r["qaoa"]["matches_optimal"],
            "qaoa_energy_gap": r["qaoa"]["energy_gap_to_optimal"],
            "qaoa_mean_feasible_frac": r["qaoa"]["mean_feasible_frac"],
            "qaoa_circuit_depth": r["qaoa"]["circuit_depth"],
        }
    with open("results/summary.json", "w") as f:
        json.dump(json_summary, f, indent=2)

    print("\nSaved full results to results/full_benchmark_results.npy and results/summary.json")


if __name__ == "__main__":
    main()
