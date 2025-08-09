import numpy as np
from pyqubo import Array, Constraint, Placeholder, CompileInfo
from typing import Tuple, List, Dict, Any

def define_problem_parameters(num_securities: int) -> Dict[str, Any]:
    """Generates a sample problem instance with random but plausible data."""
    np.random.seed(42)
    params = {}
    params['C'] = list(range(num_securities))
    params['p_c'] = np.random.uniform(90, 110, num_securities)
    params['m_c'] = np.random.uniform(1, 5, num_securities)
    params['M_c'] = params['m_c'] + np.random.uniform(10, 20, num_securities)
    params['i_c'] = np.random.uniform(5, 15, num_securities)
    params['delta_c'] = np.ones(num_securities)
    params['L'] = [0]
    params['J'] = [0]
    params['K_l'] = {0: params['C']}
    params['beta_c_j'] = np.random.uniform(-0.5, 1.5, (num_securities, 1))
    params['K_target_l_j'] = np.random.uniform(5, 10, (1, 1))
    params['rho_j'] = np.array([1.0])
    params['N'] = int(num_securities / 2)
    params['A_c'] = (params['m_c'] + np.minimum(params['M_c'], params['i_c'])) / (2 * params['delta_c'])
    return params

def build_qubo_model(params: Dict[str, Any]) -> Tuple[Dict, float, List[str], CompileInfo]:
    """Builds the QUBO model from the problem parameters."""
    num_securities = len(params['C'])
    N = params['N']
    y = Array.create('y', shape=num_securities, vartype='BINARY')

    # Objective Function
    H_obj = 0
    for l in params['L']:
        for j in params['J']:
            inner_sum = sum(params['beta_c_j'][c, j] * params['A_c'][c] * y[c] for c in params['K_l'][l])
            target = params['K_target_l_j'][l, j]
            H_obj += params['rho_j'][j] * (inner_sum - target)**2

    # Constraint: sum(y_c) <= N
    num_slack_bits = (N).bit_length()
    s = Array.create('s', shape=num_slack_bits, vartype='BINARY')
    slack_expression = sum(2**k * s[k] for k in range(num_slack_bits))
    P = Placeholder('P')
    H_constraint = Constraint((sum(y) + slack_expression - N)**2, label="max_bonds")

    # Final Hamiltonian and Model Compilation
    H = H_obj + P * H_constraint
    model = H.compile()

    # Heuristic for penalty value
    penalty_value = 2 * np.max(np.abs(model.to_qubo(feed_dict={'P': 0})[0].values()))
    
    qubo, offset = model.to_qubo(feed_dict={'P': penalty_value})
    print(f"QUBO built with {len(model.variables)} variables. Penalty P={penalty_value:.2f}")
    
    return qubo, offset, model.variables, model
