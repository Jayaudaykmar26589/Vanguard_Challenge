import argparse
import os
import time
from typing import Dict

from src.problem_builder import define_problem_parameters, build_qubo_model
from src.solvers.classical_solver import ClassicalSolver
from src.solvers.vqe_solver import VQESolver
from src.solvers.cvar_vqe_solver import CVaRVQESolver
from src.solvers.qaoa_solver import QAOASolver
from src.analysis.analyzer import analyze_solution
from src.analysis.plotter import plot_convergence, plot_solution_comparison

def main(args):
    """Main execution script controlled by CLI arguments."""
    # --- 1. Define Problem & Build QUBO ---
    print("--- Vanguard Portfolio Optimization Challenge ---")
    print(f"Problem Size: {args.num_securities} securities, Max Bonds N={int(args.num_securities/2)}\n")

    params = define_problem_parameters(args.num_securities)
    qubo, offset, var_list, compiled_model = build_qubo_model(params)

    solvers = {
        "classical": ClassicalSolver(),
        "vqe": VQESolver(max_iter=150),
        "cvar": CVaRVQESolver(alpha=args.cvar_alpha, max_iter=150),
        "qaoa": QAOASolver(layers=args.qaoa_layers, max_iter=100)
    }

    solvers_to_run = solvers.keys() if args.run_all else [args.solver]
    results: Dict[str, Dict] = {}

    # --- 2. Run Solvers ---
    for solver_name in solvers_to_run:
        solver = solvers[solver_name]
        print("\n" + "="*50)
        print(f"               RUNNING: {solver_name.upper()} SOLVER")
        print("="*50)

        start_time = time.time()
        solution_dict, history = solver.solve(qubo, offset, var_list)
        end_time = time.time()

        results[solver_name] = {
            "solution": solution_dict,
            "history": history,
            "runtime": end_time - start_time
        }

    # --- 3. Analyze and Plot Results ---
    print("\n" + "="*50)
    print("               SOLUTION ANALYSIS")
    print("="*50)

    for solver_name, result in results.items():
        print(f"\n--- {solver_name.upper()} RESULTS ---")
        metrics = analyze_solution(
            result["solution"], params, compiled_model, result["runtime"]
        )
        results[solver_name]["metrics"] = metrics

        # Plot convergence for variational algorithms
        if result["history"] and "energies" in result["history"]:
            plot_convergence(
                result["history"],
                f"{solver_name.upper()} Convergence",
                f"results/{solver_name}_convergence.png"
            )

    # Plot final comparison of all solvers run
    if len(results) > 1:
        plot_solution_comparison(results, "results/final_comparison.png")

    print("\nScript finished. Plots are saved in the 'results/' directory.")

if __name__ == "__main__":
    if not os.path.exists('results'):
        os.makedirs('results')

    parser = argparse.ArgumentParser(description="Vanguard Quantum Portfolio Optimization Solver")
    parser.add_argument("--num-securities", type=int, default=4, help="Number of securities in the portfolio.")
    parser.add_argument("--run-all", action="store_true", help="Run all available solvers.")
    parser.add_argument("--solver", type=str, default="classical", choices=["classical", "vqe", "cvar", "qaoa"], help="Select the solver to run.")
    parser.add_argument("--cvar-alpha", type=float, default=0.2, help="Alpha parameter for CVaR-VQE (fraction of worst cases to average).")
    parser.add_argument("--qaoa-layers", type=int, default=2, help="Number of layers (p) for the QAOA algorithm.")
    
    args = parser.parse_args()
    main(args)
