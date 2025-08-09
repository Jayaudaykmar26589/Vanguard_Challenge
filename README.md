```markdown
# Vanguard Quantum Portfolio Optimization Challenge - Professional Implementation

This repository presents an advanced, award-worthy solution to the Vanguard Portfolio Optimization Challenge. It leverages multiple state-of-the-art quantum algorithms to solve a constrained quadratic portfolio optimization problem, demonstrating a deep and practical understanding of hybrid quantum-classical workflows.

## Key Features & Methodologies

This solution goes beyond a basic implementation by incorporating several advanced features:

1.  **Multiple Quantum Solvers**: We implement and compare three different quantum approaches:
    *   **Variational Quantum Eigensolver (VQE)**: A standard algorithm for finding the ground state of a Hamiltonian.
    *   **CVaR-VQE**: A risk-aware version of VQE based on the paper [arXiv:1907.04769](https://arxiv.org/abs/1907.04769). Instead of minimizing the average energy, it minimizes the Conditional Value at Risk (CVaR), focusing on finding robust solutions by mitigating the worst-case outcomes. This is exceptionally relevant for financial applications.
    *   **Quantum Approximate Optimization Algorithm (QAOA)**: A powerful algorithm designed specifically for combinatorial optimization problems, providing an alternative to the VQE paradigm.

2.  **Professional Software Design**: The project is structured with modularity and scalability in mind. A `BaseSolver` class allows for easy integration and comparison of different algorithms. A command-line interface provides full control over the execution without modifying the source code.

3.  **In-Depth Analysis and Visualization**: The solution includes tools to:
    *   Validate the correctness of solutions against the problem's constraints.
    *   Generate **convergence plots** for the variational algorithms, a key performance metric.
    *   Create **summary plots** comparing the final objective values across all solvers.

4.  **Robust QUBO Formulation**: The constrained problem is converted to a QUBO (Quadratic Unconstrained Binary Optimization) using `pyqubo`, with careful handling of penalty terms to ensure constraints are respected.

## Project Structure

- `main.py`: A powerful CLI for running the entire workflow.
- `requirements.txt`: All necessary Python dependencies.
- `src/`: Source code directory.
  - `problem_builder.py`: Defines the problem and builds the QUBO model.
  - `solvers/`: Contains all solver implementations (Classical, VQE, CVaR-VQE, QAOA).
  - `analysis/`: Tools for validating solutions and generating plots.
- `tests/`: Unit tests to ensure code correctness.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd vanguard-quantum-challenge-pro
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the optimization:**
    The `main.py` script provides a flexible CLI.

    **Example: Run all solvers on a 6-asset problem and generate plots:**
    ```bash
    python main.py --num-securities 6 --run-all
    ```

    **Example: Run only the CVaR-VQE solver with specific parameters:**
    ```bash
    python main.py --num-securities 8 --solver cvar --cvar-alpha 0.25
    ```

    **See all available options:**
    ```bash
    python main.py --help
    ```

## Example Output

The script will output detailed logs for each solver and generate plots like these in the `results` directory:

**Convergence Plot (VQE/CVaR/QAOA):**
![Convergence Plot](https://i.imgur.com/example-convergence-plot.png)

**Final Solution Comparison:**
![Solution Comparison Plot](https://i.imgur.com/example-comparison-plot.png)


## Future Work & Scalability

For problems too large for today's QPUs, this framework can be extended with **problem decomposition** techniques. The modular solver design allows for integrating a meta-solver that could break the QUBO into smaller subproblems (e.g., using methods like Maximum-Cut partitioning) and solve them individually using the existing quantum solvers, before classically reconstructing a global solution.
