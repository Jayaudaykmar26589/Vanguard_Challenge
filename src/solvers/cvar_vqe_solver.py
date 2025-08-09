import cirq
import numpy as np
from scipy.optimize import minimize
from typing import Dict, Tuple, List, Any
from .base_solver import BaseSolver
from ..analysis.utils import qubo_to_ising_hamiltonian, get_bitstring_energy


class CVaRVQESolver(BaseSolver):
    """Solves with CVaR-VQE, optimizing for the average of the worst-case results."""
    def __init__(self, alpha: float = 0.2, max_iter: int = 150):
        self.alpha = alpha
        self.max_iter = max_iter

    def solve(self, qubo: Dict, offset: float, var_list: List[str]) -> Tuple[Dict[str, int], Dict[str, Any]]:
        hamiltonian, ising_offset, qubits = qubo_to_ising_hamiltonian(qubo, offset, var_list)
        num_qubits = len(qubits)

        def ansatz(params):
            # Same ansatz as VQE
            circuit = cirq.Circuit()
            for i in range(num_qubits):
                circuit.append(cirq.Ry(rads=params[2*i])(qubits[i]))
                circuit.append(cirq.Rz(rads=params[2*i+1])(qubits[i]))
            for i in range(num_qubits - 1):
                circuit.append(cirq.CZ(qubits[i], qubits[i+1]))
            return circuit

        simulator = cirq.Simulator()
        history = {"energies": [], "params": []}
        
        def cvar_cost_function(params):
            # 1. Sample from the circuit
            circuit = ansatz(params)
            circuit.append(cirq.measure(*qubits, key='result'))
            samples = simulator.run(circuit, repetitions=200).measurements['result']
            
            # 2. Calculate energy for each sample
            energies = [get_bitstring_energy(s, qubo, offset) for s in samples]
            
            # 3. Sort and calculate CVaR
            energies.sort(reverse=True) # Sort descending to get worst cases
            num_worst = int(self.alpha * len(energies))
            cvar_energy = np.mean(energies[:num_worst])
            
            history["energies"].append(cvar_energy)
            history["params"].append(params)
            return cvar_energy

        initial_params = np.random.uniform(0, 2 * np.pi, 2 * num_qubits)
        result = minimize(cvar_cost_function, initial_params, method='COBYLA', options={'maxiter': self.max_iter})

        # Final analysis is still based on the best single outcome found
        final_circuit = ansatz(result.x)
        final_circuit.append(cirq.measure(*qubits, key='result'))
        samples = simulator.run(final_circuit, repetitions=1000)
        counts = samples.histogram(key='result')
        most_common_outcome = counts.most_common(1)[0][0]
        solution_bitstring = f"{most_common_outcome:0{num_qubits}b}"
        solution = {var: int(bit) for var, bit in zip(var_list, solution_bitstring)}

        print(f"CVaR-VQE found solution with CVaR energy: {min(history['energies']):.4f}")
        return solution, history
