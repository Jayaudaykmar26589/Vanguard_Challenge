import time
from typing import Dict, Any

def analyze_solution(solution: Dict[str, int], params: Dict, compiled_model, runtime: float) -> Dict:
    """Analyzes a solution vector and returns key metrics."""
    
    # Evaluate expressions using the compiled model
    decoded_solution = compiled_model.decode_sample(solution, vartype='BINARY')
    
    num_bonds_selected = sum(v for k, v in solution.items() if k.startswith('y['))
    constraint_violation = decoded_solution.constraints(only_broken=True)
    
    print(f"Solver Runtime: {runtime:.2f} seconds")
    print(f"Number of bonds selected: {num_bonds_selected} (Constraint: <= {params['N']})")

    if not constraint_violation:
        print("Constraint: Max bonds constraint SATISFIED.")
    else:
        print(f"!! WARNING: CONSTRAINTS VIOLATED: {constraint_violation}")

    obj_val = decoded_solution.energy
    print(f"Original Objective Function Value: {obj_val:.4f}")
    
    return {
        "objective": obj_val,
        "num_bonds": num_bonds_selected,
        "constraint_ok": not constraint_violation
    }
