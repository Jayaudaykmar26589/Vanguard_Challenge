from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Any

class BaseSolver(ABC):
    """Abstract base class for all solvers."""

    @abstractmethod
    def solve(self, qubo: Dict, offset: float, var_list: List[str]) -> Tuple[Dict[str, int], Dict[str, Any]]:
        """
        Solves the given QUBO problem.

        Args:
            qubo (Dict): The QUBO dictionary.
            offset (float): The energy offset.
            var_list (List[str]): The list of variable names.

        Returns:
            A tuple containing:
            - The best solution found as a dictionary.
            - A history dictionary which may contain convergence data.
        """
        pass
