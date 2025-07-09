import yaml
import unittest
import pygarment as pyg
from assets.garment_programs.meta_garment import MetaGarment
from assets.bodies.body_params import BodyParameters

class TestGetPanelByName(unittest.TestCase):
    def setUp(self):
        with open('assets/design_params/default.yaml') as f:
            self.design = yaml.safe_load(f)['design']
        self.design['meta']['bottom']['v'] = 'SkirtCircle'
        self.design['meta']['upper']['v'] = None
        self.design['meta']['wb']['v'] = None
        self.body = BodyParameters('assets/bodies/mean_all.yaml')

    def test_nested_panel_retrieval(self):
        garment = MetaGarment('test', self.body, self.design)
        panel = garment.get_panel_by_name('skirt_front')
        self.assertIsNotNone(panel)
        self.assertIsInstance(panel, pyg.Panel)

if __name__ == '__main__':
    unittest.main()
