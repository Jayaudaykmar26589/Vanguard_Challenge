import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any

def plot_convergence(history: Dict, title: str, save_path: str):
    """Plots the energy convergence from a history dictionary."""
    plt.figure(figsize=(10, 6))
    plt.plot(history['energies'], marker='o', linestyle='-')
    plt.xlabel("Optimizer Iteration")
    plt.ylabel("Energy / Cost")
    plt.title(title)
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Convergence plot saved to {save_path}")

def plot_solution_comparison(results: Dict[str, Dict], save_path: str):
    """Plots a bar chart comparing the objective values of different solvers."""
    df = pd.DataFrame({
        solver: res["metrics"] for solver, res in results.items()
    }).T

    df['objective'].plot(kind='bar', figsize=(12, 7), color='skyblue')
    plt.ylabel("Final Objective Value")
    plt.title("Comparison of Final Objective Value by Solver")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Comparison plot saved to {save_path}")
