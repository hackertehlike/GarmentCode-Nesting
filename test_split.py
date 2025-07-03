import unittest
import numpy as np

from assets.garment_programs.circle_skirt import CircleArcPanel
import pygarment as pyg

class TestSplit(unittest.TestCase):

    def test_split_circle_arc_panel(self):
        # Create a panel to be split
        panel = CircleArcPanel.from_w_length_suns(
            'test_panel',
            length=50,
            top_width=30,
            sun_fraction=0.5
        )

        # Split the panel
        panel1, panel2 = panel.split()

        # 1. Check if two panels are returned
        self.assertIsNotNone(panel1)
        self.assertIsNotNone(panel2)
        self.assertIsInstance(panel1, pyg.Panel)
        self.assertIsInstance(panel2, pyg.Panel)

        # 2. Check for the new 'split' edge
        split_edge1 = panel1.get_edge_by_label('split_left')
        split_edge2 = panel2.get_edge_by_label('split_right')
        self.assertIsNotNone(split_edge1)
        self.assertIsNotNone(split_edge2)
        self.assertEqual(split_edge1.label, 'split_left')
        self.assertEqual(split_edge2.label, 'split_right')

        # 3. Check that the split edges are opposites
        self.assertTrue(np.allclose(split_edge1.start, split_edge2.end))
        self.assertTrue(np.allclose(split_edge1.end, split_edge2.start))

        # 4. Check if the lengths of the split edges add up
        original_top_len = panel.get_edge_by_label('top').length()
        new_top_len = panel1.get_edge_by_label('top').length() + panel2.get_edge_by_label('top').length()
        self.assertAlmostEqual(original_top_len, new_top_len, places=5)

        original_bottom_len = panel.get_edge_by_label('bottom').length()
        new_bottom_len = panel1.get_edge_by_label('bottom').length() + panel2.get_edge_by_label('bottom').length()
        self.assertAlmostEqual(original_bottom_len, new_bottom_len, places=5)

        # 5. Check that the side edges are preserved
        orig_left = panel.get_edge_by_label('left')
        orig_right = panel.get_edge_by_label('right')
        left1 = panel1.get_edge_by_label('left')
        right2 = panel2.get_edge_by_label('right')
        self.assertIsNotNone(left1)
        self.assertIsNotNone(right2)
        self.assertTrue(np.allclose(left1.start, orig_left.start))
        self.assertTrue(np.allclose(left1.end, orig_left.end))
        self.assertTrue(np.allclose(right2.start, orig_right.start))
        self.assertTrue(np.allclose(right2.end, orig_right.end))

if __name__ == '__main__':
    unittest.main()
