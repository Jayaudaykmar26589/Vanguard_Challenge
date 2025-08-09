import cirq
import numpy as np
from scipy.optimize import minimize
from typing import Dict, Tuple, List, Any
from .base_solver import BaseSolver
from ..analysis.utils import qubo_to_ising_hamiltonian

class QAOASolver(BaseSolver):
    """Solves the problem using the Quantum Approximate Optimization Algorithm (QAOA)."""
    def __init__(self, layers: int = 2, max_iter: int = 100):
        self.p = layers # Number of QAOA layers

    def solve(self, qubo: Dict, offset: float, var_list: List[str]) -> Tuple[Dict[str, int], Dict[str, Any]]:
        problem_hamiltonian, ising_offset, qubits = qubo_to_ising_hamiltonian(qubo, offset, var_list)
        num_qubits = len(qubits)

        # Mixer Hamiltonian
        mixer_hamiltonian = cirq.PauliSum.from_pauli_strings([cirq.X(q) for q in qubits])
        
        # QAOA Ansatz
        def qaoa_ansatz(params):
            gammas = params[:self.p]
            betas = params[self.p:]
            circuit = cirq.Circuit(cirq.H.on_each(*qubits))
            for i in range(self.p):
                # Problem Hamiltonian Evolution
                circuit += cirq.PauliSumExponential(problem_hamiltonian, gammas[i])
                # Mixer Hamiltonian Evolution
                circuit += cirq.PauliSumExponential(mixer_hamiltonian, betas[i])
            return circuit

        simulator = cirq.Simulator()
        history = {"energies": [], "params": []}

        def cost_function(params):
            circuit = qaoa_ansatz(params)
            expectation = simulator.simulate_expectation_values(circuit, problem_hamiltonian)
            energy = np.real(expectation[0])
            history["energies"].append(energy + ising_offset)
            history["params"].append(params)
            return energy

        initial_params = np.random.uniform(0, np.pi, 2 * self.p)
        result = minimize(cost_function, initial_params, method='COBYLA', options={'maxiter': self.max_iter})

        final_circuit = qaoa_ansatz(result.x)
        final_circuit.append(cirq.measure(*qubits, key='result'))
        samples = simulator.run(final_circuit, repetitions=1000)
        counts = samples.histogram(key='result')
        most_common_outcome = counts.most_common(1)[0][0]
        solution_bitstring = f"{most_common_outcome:0{num_qubits}b}"
        solution = {var: int(bit) for var, bit in zip(var_list, solution_bitstring)}
        
        print(f"QAOA found solution with energy: {min(history['energies']):.4f}")
        return solution, history
