import unittest
import pygarment as pyg
from assets.garment_programs.circle_skirt import CircleArcPanel

class TestEdgeLabels(unittest.TestCase):
    def test_propagate_label_append(self):
        panel = CircleArcPanel.from_w_length_suns('test_panel', length=50, top_width=30, sun_fraction=0.5)
        top_seq = panel.interfaces['top'].edges
        original_label = top_seq[0].label
        top_seq.propagate_label('lower_interface', append=True)
        self.assertEqual(top_seq[0].label, original_label)
        self.assertIn('lower_interface', top_seq[0].extra_labels)
        # ensure split still works
        panel1, panel2 = panel.split()
        self.assertIsNotNone(panel1)
        self.assertIsNotNone(panel2)
        self.assertIsInstance(panel1, pyg.Panel)
        self.assertIsInstance(panel2, pyg.Panel)

if __name__ == '__main__':
    unittest.main()
