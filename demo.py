#!/usr/bin/env python3
"""QuantumGrid demo — run all solvers on a small UC instance and compare."""
from __future__ import annotations

import sys
import numpy as np

from data.generators import build_small_fleet
from formulation.unit_commitment import solve_unit_commitment
from formulation.qubo_encoding import uc_to_qubo, qubo_to_ising
from quantum.qaoa_solver import solve_qaoa
from quantum.vqe_solver import solve_vqe
from classical.baselines import solve_greedy, solve_simulated_annealing
from benchmarks.penalty_tuning import check_constraints
from visualization.plots import plot_convergence, plot_dispatch, plot_schedule_heatmap


def main():
    print("=" * 60)
    print("  QuantumGrid — Quantum-Classical UC Benchmark")
    print("=" * 60)

    # Problem setup
    N, T = 4, 6
    fleet = build_small_fleet(n=N, seed=42)
    demand = np.array([300.0, 350.0, 400.0, 380.0, 320.0, 280.0])
    print(f"\nProblem: {N} generators × {T} timesteps = {N * T} qubits")
    print(f"Demand: {demand.tolist()}")
    print(f"Fleet capacity: {fleet.total_capacity():.0f} MW")

    results = {}

    # 1. MILP (optimal reference)
    print("\n--- MILP (OR-Tools SCIP) ---")
    milp = solve_unit_commitment(fleet, demand, reserve_fraction=0.10, time_limit_s=30)
    print(f"  Status: {milp.status}")
    print(f"  Cost: ${milp.total_cost:,.0f}")
    print(f"  Time: {milp.solve_time_s:.3f}s")
    results["MILP"] = milp.total_cost

    # 2. Greedy
    print("\n--- Greedy Priority List ---")
    greedy = solve_greedy(fleet, demand)
    chk = check_constraints(fleet, demand, greedy.schedule, greedy.dispatch)
    print(f"  Cost: ${greedy.total_cost:,.0f}")
    print(f"  Demand met: {chk['demand_met_fraction']:.0%}")
    print(f"  Time: {greedy.solve_time_s:.4f}s")
    results["Greedy"] = greedy.total_cost

    # 3. Simulated Annealing
    print("\n--- Simulated Annealing ---")
    sa = solve_simulated_annealing(fleet, demand, max_iterations=2000, seed=42)
    chk = check_constraints(fleet, demand, sa.schedule, sa.dispatch)
    print(f"  Cost: ${sa.total_cost:,.0f}")
    print(f"  Demand met: {chk['demand_met_fraction']:.0%}")
    print(f"  Time: {sa.solve_time_s:.3f}s")
    results["SA"] = sa.total_cost

    # 4. QAOA
    print("\n--- QAOA (p=1, 40 iterations) ---")
    qaoa = solve_qaoa(fleet, demand, n_layers=1, max_iterations=40, seed=42)
    chk = check_constraints(fleet, demand, qaoa.schedule, qaoa.dispatch)
    print(f"  QUBO cost: {qaoa.best_cost:,.0f}")
    print(f"  Demand met: {chk['demand_met_fraction']:.0%}")
    print(f"  Time: {qaoa.solve_time_s:.1f}s")
    results["QAOA"] = qaoa.best_cost

    # 5. VQE (tiny instance — 2 gen × 3 timesteps)
    print("\n--- VQE (2 gen × 3 t, 1 layer, 20 iterations) ---")
    fleet_v = build_small_fleet(n=2, seed=0)
    demand_v = np.array([100.0, 150.0, 120.0])
    vqe = solve_vqe(fleet_v, demand_v, n_layers=1, max_iterations=20, seed=42)
    chk = check_constraints(fleet_v, demand_v, vqe.schedule, vqe.dispatch)
    print(f"  QUBO cost: {vqe.best_cost:,.0f}")
    print(f"  Demand met: {chk['demand_met_fraction']:.0%}")
    print(f"  Time: {vqe.solve_time_s:.2f}s")

    # Summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    for method, cost in results.items():
        print(f"  {method:>10s}: ${cost:>12,.0f}")

    # Save plots
    print("\nSaving plots to results/ ...")
    import os
    os.makedirs("results", exist_ok=True)

    plot_convergence(
        {"QAOA (p=1)": qaoa.convergence},
        title="QAOA Convergence",
        save_path="results/convergence.png",
    )
    plot_dispatch(
        greedy.dispatch, demand,
        title="Greedy Dispatch",
        save_path="results/dispatch_greedy.png",
    )
    plot_schedule_heatmap(
        qaoa.schedule.astype(int),
        title="QAOA Schedule (p=1)",
        save_path="results/schedule_qaoa.png",
    )
    print("Done. See results/ directory for plots.\n")


if __name__ == "__main__":
    main()
