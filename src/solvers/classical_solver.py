from typing import Dict, Tuple, List, Any
from dwave.samplers import SimulatedAnnealingSampler
from .base_solver import BaseSolver

class ClassicalSolver(BaseSolver):
    """Solves the QUBO using a classical simulated annealer for benchmarking."""
    def solve(self, qubo: Dict, offset: float, var_list: List[str]) -> Tuple[Dict[str, int], Dict[str, Any]]:
        sampler = SimulatedAnnealingSampler()
        response = sampler.sample_qubo(qubo, num_reads=100)
        solution = response.first.sample
        print(f"Classical solver found solution with energy: {response.first.energy + offset:.4f}")
        return solution, {}
