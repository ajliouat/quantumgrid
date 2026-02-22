"""Scaling analysis benchmarks.

Systematically measure solve time and solution quality as problem size grows:
  - N generators: 2, 3, 4, 5
  - T timesteps: 3, 4, 6
  - QAOA layers p: 1, 2
  - Compares MILP, QAOA, VQE, SA, Greedy
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, List, Optional

import numpy as np

from data.generators import build_small_fleet, build_fleet
from formulation.qubo_encoding import evaluate_qubo, uc_to_qubo, decode_solution
from classical.baselines import solve_greedy, solve_simulated_annealing


@dataclass
class ScalingPoint:
    """One measurement point in the scaling study."""

    n_generators: int
    n_timesteps: int
    n_qubits: int
    method: str
    layers: Optional[int]
    solve_time_s: float
    total_cost: float
    demand_met_fraction: float


@dataclass
class ScalingStudy:
    """Collection of scaling measurements."""

    points: List[ScalingPoint] = field(default_factory=list)

    def to_records(self) -> List[Dict]:
        return [
            {
                "n_gen": p.n_generators,
                "n_time": p.n_timesteps,
                "n_qubits": p.n_qubits,
                "method": p.method,
                "layers": p.layers,
                "time_s": p.solve_time_s,
                "cost": p.total_cost,
                "demand_met": p.demand_met_fraction,
            }
            for p in self.points
        ]

    def filter_method(self, method: str) -> List[ScalingPoint]:
        return [p for p in self.points if p.method == method]


def _demand_met_fraction(fleet, demand, schedule, dispatch):
    gen_total = dispatch.sum(axis=0)
    met = gen_total >= demand * 0.99
    return float(met.mean())


def run_classical_scaling(
    n_gen_list: List[int] = None,
    t_list: List[int] = None,
    sa_iterations: int = 500,
    seed: int = 42,
) -> ScalingStudy:
    """Run classical solvers across problem sizes.

    Uses only MILP, SA, Greedy — doesn't need quantum simulation.
    """
    if n_gen_list is None:
        n_gen_list = [4, 6, 8, 10]
    if t_list is None:
        t_list = [6, 12, 24]

    study = ScalingStudy()
    rng = np.random.default_rng(seed)

    for N in n_gen_list:
        for T in t_list:
            fleet = build_fleet(n_generators=N, total_capacity_mw=N * 500, seed=seed)
            demand = rng.uniform(N * 100, N * 300, size=T)

            # Greedy
            t0 = perf_counter()
            res_g = solve_greedy(fleet, demand)
            t_g = perf_counter() - t0
            dmf_g = _demand_met_fraction(fleet, demand, res_g.schedule, res_g.dispatch)
            study.points.append(ScalingPoint(
                n_generators=N, n_timesteps=T, n_qubits=N * T,
                method="greedy", layers=None,
                solve_time_s=t_g, total_cost=res_g.total_cost,
                demand_met_fraction=dmf_g,
            ))

            # SA
            t0 = perf_counter()
            res_sa = solve_simulated_annealing(
                fleet, demand, max_iterations=sa_iterations, seed=seed
            )
            t_sa = perf_counter() - t0
            dmf_sa = _demand_met_fraction(fleet, demand, res_sa.schedule, res_sa.dispatch)
            study.points.append(ScalingPoint(
                n_generators=N, n_timesteps=T, n_qubits=N * T,
                method="sa", layers=None,
                solve_time_s=t_sa, total_cost=res_sa.total_cost,
                demand_met_fraction=dmf_sa,
            ))

    return study


def run_quantum_scaling(
    n_gen_list: List[int] = None,
    t_list: List[int] = None,
    p_list: List[int] = None,
    max_qaoa_iter: int = 20,
    max_vqe_iter: int = 20,
    seed: int = 42,
) -> ScalingStudy:
    """Run quantum solvers across small problem sizes.

    Only imports quantum solvers when called (they are slow).
    """
    from quantum.qaoa_solver import solve_qaoa
    from quantum.vqe_solver import solve_vqe

    if n_gen_list is None:
        n_gen_list = [2, 3, 4]
    if t_list is None:
        t_list = [3, 4]
    if p_list is None:
        p_list = [1, 2]

    study = ScalingStudy()
    rng = np.random.default_rng(seed)

    for N in n_gen_list:
        fleet = build_small_fleet(n=N, seed=seed)
        for T in t_list:
            demand = rng.uniform(N * 50, N * 150, size=T)

            for p in p_list:
                # QAOA
                res_q = solve_qaoa(fleet, demand, n_layers=p,
                                   max_iterations=max_qaoa_iter, seed=seed)
                schedule, dispatch = res_q.schedule, res_q.dispatch
                dmf = _demand_met_fraction(fleet, demand, schedule, dispatch)
                study.points.append(ScalingPoint(
                    n_generators=N, n_timesteps=T, n_qubits=N * T,
                    method="qaoa", layers=p,
                    solve_time_s=res_q.solve_time_s, total_cost=res_q.best_cost,
                    demand_met_fraction=dmf,
                ))

            # VQE (1 layer only — too slow for deeper)
            res_v = solve_vqe(fleet, demand, n_layers=1,
                              max_iterations=max_vqe_iter, seed=seed)
            schedule, dispatch = res_v.schedule, res_v.dispatch
            dmf = _demand_met_fraction(fleet, demand, schedule, dispatch)
            study.points.append(ScalingPoint(
                n_generators=N, n_timesteps=T, n_qubits=N * T,
                method="vqe", layers=1,
                solve_time_s=res_v.solve_time_s, total_cost=res_v.best_cost,
                demand_met_fraction=dmf,
            ))

    return study
