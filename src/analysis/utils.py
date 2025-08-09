import cirq
import numpy as np
from typing import Dict, List, Tuple

def qubo_to_ising_hamiltonian(
    qubo: Dict[Tuple[str, str], float],
    offset: float,
    var_list: List[str]
) -> Tuple[cirq.PauliSum, float, List[cirq.GridQubit]]:
    """
    Converts a QUBO dictionary to a Cirq Ising Hamiltonian.

    The conversion uses the identity x = (1 - Z) / 2, where x is a binary
    variable (0, 1) and Z is a Pauli Z operator with eigenvalues (+1, -1).

    Args:
        qubo: The QUBO problem, formatted as a dictionary mapping variable
              tuples to their quadratic coefficients.
        offset: The constant energy offset in the QUBO formulation.
        var_list: An ordered list of variable names. The order determines
                  the mapping from variables to qubits.

    Returns:
        A tuple containing:
        - The Cirq PauliSum representing the Ising Hamiltonian.
        - The new constant energy offset for the Ising model.
        - A list of Cirq qubits corresponding to the variables.
    """
    num_vars = len(var_list)
    qubits = cirq.GridQubit.rect(1, num_vars)
    var_map = {var_name: i for i, var_name in enumerate(var_list)}

    # Initialize Ising model parameters
    linear_coeffs = np.zeros(num_vars)      # h_i for Z_i terms
    quadratic_coeffs = np.zeros((num_vars, num_vars)) # J_ij for Z_i Z_j terms
    ising_offset = offset

    # Decompose QUBO coefficients into linear and quadratic parts
    for (i_str, j_str), coeff in qubo.items():
        i, j = var_map[i_str], var_map[j_str]
        if i == j:
            # This is a linear QUBO term: Q_ii * x_i
            # x_i -> (1 - Z_i) / 2
            # Q_ii * x_i -> Q_ii * (1 - Z_i) / 2 = Q_ii/2 - (Q_ii/2) * Z_i
            ising_offset += coeff / 2.0
            linear_coeffs[i] -= coeff / 2.0
        else:
            # This is a quadratic QUBO term: Q_ij * x_i * x_j
            # x_i * x_j -> (1 - Z_i)/2 * (1 - Z_j)/2 = (1 - Z_i - Z_j + Z_i*Z_j)/4
            # Q_ij * x_i*x_j -> Q_ij/4 * (1 - Z_i - Z_j + Z_i*Z_j)
            ising_offset += coeff / 4.0
            linear_coeffs[i] -= coeff / 4.0
            linear_coeffs[j] -= coeff / 4.0
            quadratic_coeffs[i, j] += coeff / 4.0

    # Build the PauliSum from the Ising coefficients
    hamiltonian_terms = []

    # Add linear (h_i * Z_i) terms
    for i in range(num_vars):
        if not np.isclose(linear_coeffs[i], 0):
            hamiltonian_terms.append(linear_coeffs[i] * cirq.Z(qubits[i]))

    # Add quadratic (J_ij * Z_i * Z_j) terms
    for i in range(num_vars):
        for j in range(i + 1, num_vars):
            if not np.isclose(quadratic_coeffs[i, j], 0):
                hamiltonian_terms.append(quadratic_coeffs[i, j] * cirq.Z(qubits[i]) * cirq.Z(qubits[j]))

    return cirq.PauliSum.from_pauli_strings(hamiltonian_terms), ising_offset, qubits


def get_bitstring_energy(
    bitstring_int: int,
    qubo: Dict[Tuple[str, str], float],
    offset: float,
    var_list: List[str]
) -> float:
    """
    Calculates the energy of a specific bitstring outcome for a given QUBO.

    This is used by the CVaR-VQE solver to evaluate the cost of individual
    measurement samples.

    Args:
        bitstring_int: The integer representation of the measurement outcome
                       (as returned by Cirq).
        qubo: The QUBO problem dictionary.
        offset: The constant energy offset.
        var_list: The ordered list of variable names.

    Returns:
        The classical energy of the given bitstring according to the QUBO.
    """
    num_vars = len(var_list)
    # Convert integer to a padded binary string, then to a solution dictionary
    binary_string = f"{bitstring_int:0{num_vars}b}"
    solution = {var: int(bit) for var, bit in zip(var_list, binary_string)}
    
    energy = offset
    for (i_str, j_str), coeff in qubo.items():
        energy += coeff * solution[i_str] * solution[j_str]
        
    return energy
