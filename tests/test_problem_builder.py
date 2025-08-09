import unittest
from src.problem_builder import define_problem_parameters, build_qubo_model

class TestProblemBuilder(unittest.TestCase):

    def test_qubo_creation(self):
        """Test that the QUBO is created without errors for a small problem."""
        params = define_problem_parameters(num_securities=4)
        qubo, offset, var_list, _ = build_qubo_model(params)
        
        self.assertIsInstance(qubo, dict)
        self.assertIsInstance(offset, float)
        self.assertGreater(len(var_list), 0)
        
        # Check that both asset and slack variables are present
        has_y_var = any(v.startswith('y[') for v in var_list)
        has_s_var = any(v.startswith('s[') for v in var_list)
        self.assertTrue(has_y_var)
        self.assertTrue(has_s_var)

if __name__ == '__main__':
    unittest.main()
