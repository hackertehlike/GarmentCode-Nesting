import unittest
import tempfile
import nesting.config as config
from nesting.evolution import Evolution
from nesting.layout import Container
from unittest.mock import MagicMock

class CompareDesignParamsTest(unittest.TestCase):
    def test_compare_design_params(self):
        params1 = {'a': {'b': 1, 'c': [1,2]}}
        params2 = {'a': {'b': 2, 'c': [1,3]}}
        
        # Create mock objects for body_params and design_sampler
        mock_body_params = MagicMock()
        mock_design_sampler = MagicMock()
        
        with tempfile.TemporaryDirectory() as tmp:
            config.SAVE_LOGS_PATH = tmp
            evo = Evolution(
                {}, 
                Container(10,10), 
                design_params=params1,
                body_params=mock_body_params,
                design_sampler=mock_design_sampler
            )
            diffs = evo._compare_design_params(params1, params2)
            self.assertEqual(len(diffs), 2)
            self.assertIn(('a.b', 1, 2), diffs)
            self.assertIn(('a.c.1', 2, 3), diffs)

if __name__ == '__main__':
    unittest.main()