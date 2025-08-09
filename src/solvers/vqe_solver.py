import cirq
import numpy as np
from scipy.optimize import minimize
from typing import Dict, Tuple, List, Any
from .base_solver import BaseSolver
from ..analysis.utils import qubo_to_ising_hamiltonian # Helper function

class VQESolver(BaseSolver):
    """Solves the problem using the Variational Quantum Eigensolver (VQE)."""
    def __init__(self, max_iter: int = 150):
        self.max_iter = max_iter

    def solve(self, qubo: Dict, offset: float, var_list: List[str]) -> Tuple[Dict[str, int], Dict[str, Any]]:
        hamiltonian, ising_offset, qubits = qubo_to_ising_hamiltonian(qubo, offset, var_list)
        num_qubits = len(qubits)
        
        # Hardware-Efficient Ansatz
        def ansatz(params):
            circuit = cirq.Circuit()
            for i in range(num_qubits):
                circuit.append(cirq.Ry(rads=params[2*i])(qubits[i]))
                circuit.append(cirq.Rz(rads=params[2*i+1])(qubits[i]))
            for i in range(num_qubits - 1):
                circuit.append(cirq.CZ(qubits[i], qubits[i+1]))
            return circuit

        simulator = cirq.Simulator()
        history = {"energies": [], "params": []}

        def cost_function(params):
            circuit = ansatz(params)
            expectation = simulator.simulate_expectation_values(circuit, hamiltonian)
            energy = np.real(expectation[0])
            history["energies"].append(energy + ising_offset)
            history["params"].append(params)
            return energy

        initial_params = np.random.uniform(0, 2 * np.pi, 2 * num_qubits)
        result = minimize(cost_function, initial_params, method='COBYLA', options={'maxiter': self.max_iter})

        final_circuit = ansatz(result.x)
        final_circuit.append(cirq.measure(*qubits, key='result'))
        samples = simulator.run(final_circuit, repetitions=1000)
        counts = samples.histogram(key='result')
        most_common_outcome = counts.most_common(1)[0][0]
        solution_bitstring = f"{most_common_outcome:0{num_qubits}b}"
        solution = {var: int(bit) for var, bit in zip(var_list, solution_bitstring)}

        print(f"VQE found solution with energy: {min(history['energies']):.4f}")
        return solution, history
