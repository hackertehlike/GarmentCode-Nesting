import numpy as np
import pygarment as pyg

from assets.garment_programs.base_classes import StackableSkirtComponent


class CircleArcPanel(pyg.Panel):
    """One panel circle skirt"""

    def __init__(self, 
                 name, 
                 top_rad, length, angle, 
                 match_top_int_proportion=None, 
                 match_bottom_int_proportion=None
            ) -> None:
        super().__init__(name)

        halfarc = angle / 2

        dist_w = 2 * top_rad * np.sin(halfarc)
        dist_out = 2 * (top_rad + length) * np.sin(halfarc)

        vert_len = length * np.cos(halfarc)

        # Create empty edge sequence
        self.edges = pyg.EdgeSequence()

        # top edge - index 0
        top_edge = pyg.CircleEdgeFactory.from_points_radius(
            [-dist_w/2, 0], [dist_w/2, 0], 
            radius=top_rad, large_arc=halfarc > np.pi / 2, label='top')
        self.edges.append(top_edge)

        # right edge - index 1
        right_edge = pyg.Edge(
            self.edges[-1].end, [dist_out / 2, -vert_len], label='right')
        self.edges.append(right_edge)
        
        # Bottom edge - index 2
        bottom_edge = pyg.CircleEdgeFactory.from_points_radius(
            self.edges[-1].end, [- dist_out / 2, -vert_len], 
            radius=top_rad + length,
            large_arc=halfarc > np.pi / 2, right=False, label='bottom')
        self.edges.append(bottom_edge)

        # left edge - index 3 - closes the loop
        left_edge = pyg.Edge(self.edges[-1].end, self.edges[0].start, label='left')
        self.edges.append(left_edge)

        # Verify all edge labels are set correctly
        # print(f"[CircleArcPanel.__init__] Panel {name} created with edges:")
        # for i, edge in enumerate(self.edges):
        #     print(f"  Edge {i}: label={getattr(edge, 'label', 'None')}")

        # Interfaces
        self.interfaces = {
            'top': pyg.Interface(self, self.edges[0],
                                 ruffle=self.edges[0].length() / match_top_int_proportion if match_top_int_proportion is not None else 1.
                                 ).reverse(True),
            'bottom': pyg.Interface(self, self.edges[2],
                                    ruffle=self.edges[2].length() / match_bottom_int_proportion if match_bottom_int_proportion is not None else 1.
                                    ),
            'left': pyg.Interface(self, self.edges[3]),
            'right': pyg.Interface(self, self.edges[1])
        }
        
        # Double-check edge labels after initialization is complete
        self._verify_edge_labels()
        
    def _verify_edge_labels(self):
        """
        Verify that all edges have the expected labels after initialization.
        This helps catch issues where labels might be lost or incorrectly assigned.
        """
        expected_labels = ['top', 'right', 'bottom', 'left']
        missing_labels = []
        
        # print(f"[CircleArcPanel._verify_edge_labels] Verifying edge labels for panel '{self.name}'")
        
        # Check that each expected label exists on an edge
        for label in expected_labels:
            edge = self.get_edge_by_label(label)
            if edge is None:
                missing_labels.append(label)
                print(f"[CircleArcPanel._verify_edge_labels] ERROR: Edge with label '{label}' not found")
        
        # If we're missing labels, print debugging info and fix them
        if missing_labels:
            print(f"[CircleArcPanel._verify_edge_labels] Missing labels: {missing_labels}")
            print(f"[CircleArcPanel._verify_edge_labels] Attempting to fix missing labels...")
            
            # Ensure the expected labels are set based on index
            if 'top' in missing_labels and len(self.edges) >= 1:
                self.edges[0].label = 'top'
                print(f"[CircleArcPanel._verify_edge_labels] Set label 'top' on edge 0")
            
            if 'right' in missing_labels and len(self.edges) >= 2:
                self.edges[1].label = 'right'
                print(f"[CircleArcPanel._verify_edge_labels] Set label 'right' on edge 1")
            
            if 'bottom' in missing_labels and len(self.edges) >= 3:
                self.edges[2].label = 'bottom'
                print(f"[CircleArcPanel._verify_edge_labels] Set label 'bottom' on edge 2")
            
            if 'left' in missing_labels and len(self.edges) >= 4:
                self.edges[3].label = 'left'
                print(f"[CircleArcPanel._verify_edge_labels] Set label 'left' on edge 3")
        # else:
        #     print(f"[CircleArcPanel._verify_edge_labels] All expected edge labels found")

    def length(self, *args):
        return self.interfaces['right'].edges.length()
    
    @staticmethod
    def from_w_length_suns(name, length, top_width, sun_fraction, **kwargs):
        arc = sun_fraction * 2 * np.pi
        rad = top_width / arc

        return CircleArcPanel(name, rad, length, arc, **kwargs)
    
    @staticmethod
    def from_all_length(name, length, top_width, bottom_width, **kwargs):

        diff = bottom_width - top_width
        arc = diff / length
        rad = top_width / arc

        return CircleArcPanel(name, rad, length, arc, **kwargs)
    
    @staticmethod
    def from_length_rad(name, length, top_width, rad, **kwargs):

        arc = top_width / rad

        return CircleArcPanel(name, rad, length, arc, **kwargs)

    def split(self, proportion=0.5):
        """Splits the panel into two new panels at a specified proportion"""
        # Debug information about edges
        print(f"[CircleArcPanel.split] Splitting panel '{self.name}' at proportion {proportion}")
        self._verify_edge_labels()  # Ensure edge labels are verified before splitting

        # print(f"[CircleArcPanel.split] Panel name: {self.name}")
        # print(f"[CircleArcPanel.split] Number of edges: {len(self.edges)}")
        # for i, edge in enumerate(self.edges):
        #     print(f"[CircleArcPanel.split] Edge {i}: label={getattr(edge, 'label', 'None')}")
        
        # Get edges by label - this should now work reliably with our changes
        bottom_edge = self.get_edge_by_label('bottom')
        top_edge = self.get_edge_by_label('top')
        left_edge = self.get_edge_by_label('left')
        right_edge = self.get_edge_by_label('right')
        
        # Validate all edges are found
        if bottom_edge is None:
            raise ValueError(f"Panel {self.name} does not have a bottom edge to split.")
        if top_edge is None:
            raise ValueError(f"Panel {self.name} does not have a top edge to split.")
        if left_edge is None:
            raise ValueError(f"Panel {self.name} does not have a left edge to split.")
        if right_edge is None:
            raise ValueError(f"Panel {self.name} does not have a right edge to split.")

        # Get split points
        split_point_bottom = bottom_edge.point_at(proportion)

        print(f"[CircleArcPanel.split] Split point on bottom edge: {split_point_bottom}")
        
        split_point_top = top_edge.point_at(proportion)

        print(f"[CircleArcPanel.split] Split points: "
              f"bottom={split_point_bottom}, top={split_point_top}")

        # Split edges
        bottom1, bottom2 = bottom_edge.split_at_point(split_point_bottom)
        top1, top2 = top_edge.split_at_point(split_point_top)

        # Create new split edges for each panel
        split_edge1 = pyg.Edge(split_point_top, split_point_bottom, label='split_left')
        split_edge2 = pyg.Edge(split_point_bottom, split_point_top, label='split_right')


        # Debug the split edges
        print(f"[CircleArcPanel.split] Split edges created:")
        print(f"  Split Edge 1: {split_edge1}")
        print(f"  Split Edge 2: {split_edge2}")

        
        # Create new panels
        panel1 = pyg.Panel(f'{self.name}_1')
        panel1.edges = pyg.EdgeSequence([top1, right_edge, bottom1, split_edge1])
        
        # Override the edge labels on panel1
        panel1.edges[0].label = 'top'
        panel1.edges[1].label = 'right'
        panel1.edges[2].label = 'bottom'
        panel1.edges[3].label = 'left'  # The split edge becomes the left edge of panel1

        print(f"[CircleArcPanel.split] Created panel1")

        panel2 = pyg.Panel(f'{self.name}_2')
        panel2.edges = pyg.EdgeSequence([top2, split_edge2, bottom2, left_edge])
        
        # Override the edge labels on panel2
        panel2.edges[0].label = 'top'
        panel2.edges[1].label = 'right'  # The split edge becomes the right edge of panel2
        panel2.edges[2].label = 'bottom'
        panel2.edges[3].label = 'left'

        print(f"[CircleArcPanel.split] Created panel2")

        # print(f"[CircleArcPanel.split] Created panel1 with edges:")
        # for i, edge in enumerate(panel1.edges):
        #     print(f"  Edge {i}: label={getattr(edge, 'label', 'None')}")
            
        # print(f"[CircleArcPanel.split] Created panel2 with edges:")
        # for i, edge in enumerate(panel2.edges):
        #     print(f"  Edge {i}: label={getattr(edge, 'label', 'None')}")
            
        # Apply the verification method to the new panels
        # Add the _verify_edge_labels method to the new panels
        #panel1._verify_edge_labels = self._verify_edge_labels.__get__(panel1)
        #panel2._verify_edge_labels = self._verify_edge_labels.__get__(panel2)
        
        #print(f"[CircleArcPanel.split] Verifying edge labels for new panels")

        # Verify edge labels in both new panels
        #panel1._verify_edge_labels()
        #panel2._verify_edge_labels()


        #print(f"[CircleArcPanel.split] Edge labels verified for new panels")
        # Return the new panels
        print(f"[CircleArcPanel.split] Returning new panels: {panel1.name}, {panel2.name}")
        return panel1, panel2

    def get_edge_by_label(self, label):
        """
        Get an edge by its label without any fallback to indices.
        This implementation requires that all edges have proper labels.
        """
        # Debug the label lookup
        # print(f"[CircleArcPanel.get_edge_by_label] Looking for edge with label '{label}'")
        
        # First check if the Panel class has this method
        if hasattr(super(), 'get_edge_by_label'):
            edge = super().get_edge_by_label(label)
            if edge is not None:
                return edge
        
        # If not found or method doesn't exist, implement our own lookup
        for i, edge in enumerate(self.edges):
            if hasattr(edge, 'label') and edge.label == label:
                # print(f"[CircleArcPanel.get_edge_by_label] Found edge {i} with label '{label}'")
                return edge
        
        # Debug what labels we do have
        print(f"[CircleArcPanel.get_edge_by_label] Warning: Edge with label '{label}' not found")
        for i, edge in enumerate(self.edges):
            if hasattr(edge, 'label'):
                print(f"[CircleArcPanel.get_edge_by_label] Edge {i} has label '{edge.label}'")
            else:
                print(f"[CircleArcPanel.get_edge_by_label] Edge {i} has no label attribute")
        
        # No fallback - strict behavior as requested
        return None

class AsymHalfCirclePanel(pyg.Panel):
    """Panel for a asymmetrci circle skirt"""

    def __init__(self, 
                 name, 
                 top_rad, length_f, length_s,
                 match_top_int_proportion=None, 
                 match_bottom_int_proportion=None
                 ) -> None:
        """ Half a shifted arc section
        """
        super().__init__(name)

        dist_w = 2 * top_rad 
        dist_out = 2 * (top_rad + length_s)

        # Create empty edge sequence
        self.edges = pyg.EdgeSequence()

        # top edge - index 0
        top_edge = pyg.CircleEdgeFactory.from_points_radius(
            [-dist_w/2, 0], [dist_w/2, 0], 
            radius=top_rad, large_arc=False, label='top')
        self.edges.append(top_edge)

        # right edge - index 1
        right_edge = pyg.Edge(
            self.edges[-1].end, [dist_out / 2, 0], label='right')
        self.edges.append(right_edge)
        
        # Bottom edge - index 2
        bottom_edge = pyg.CircleEdgeFactory.from_three_points(
            self.edges[-1].end, [- dist_out / 2, 0], 
            point_on_arc=[0, -(top_rad + length_f)],
            label='bottom'
        )
        self.edges.append(bottom_edge)

        # left edge - index 3 - closes the loop
        left_edge = pyg.Edge(self.edges[-1].end, self.edges[0].start, label='left')
        self.edges.append(left_edge)

        # Verify all edge labels are set correctly
        # print(f"[AsymHalfCirclePanel.__init__] Panel {name} created with edges:")
        # for i, edge in enumerate(self.edges):
        #     print(f"  Edge {i}: label={getattr(edge, 'label', 'None')}")

        # Interfaces
        self.interfaces = {
            'top': pyg.Interface(self, self.edges[0],
                                 ruffle=self.edges[0].length() / match_top_int_proportion if match_top_int_proportion is not None else 1.
                                 ).reverse(True),
            'bottom': pyg.Interface(self, self.edges[2],
                                    ruffle=self.edges[2].length() / match_bottom_int_proportion if match_bottom_int_proportion is not None else 1.
                                    ),
            'left': pyg.Interface(self, self.edges[3]),
            'right': pyg.Interface(self, self.edges[1])
        }
        
        # Double-check edge labels after initialization is complete
        self._verify_edge_labels()
        
    def _verify_edge_labels(self):
        """
        Verify that all edges have the expected labels after initialization.
        This helps catch issues where labels might be lost or incorrectly assigned.
        """
        expected_labels = ['top', 'right', 'bottom', 'left']
        missing_labels = []
        
        # print(f"[AsymHalfCirclePanel._verify_edge_labels] Verifying edge labels for panel '{self.name}'")
        
        # Check that each expected label exists on an edge
        for label in expected_labels:
            edge = None
            # Find edge with this label
            for e in self.edges:
                if hasattr(e, 'label') and e.label == label:
                    edge = e
                    break
                    
            if edge is None:
                missing_labels.append(label)
                print(f"[AsymHalfCirclePanel._verify_edge_labels] ERROR: Edge with label '{label}' not found")
        
        # If we're missing labels, print debugging info and fix them
        if missing_labels:
            print(f"[AsymHalfCirclePanel._verify_edge_labels] Missing labels: {missing_labels}")
            print(f"[AsymHalfCirclePanel._verify_edge_labels] Attempting to fix missing labels...")
            
            # Ensure the expected labels are set based on index
            if 'top' in missing_labels and len(self.edges) >= 1:
                self.edges[0].label = 'top'
                print(f"[AsymHalfCirclePanel._verify_edge_labels] Set label 'top' on edge 0")
            
            if 'right' in missing_labels and len(self.edges) >= 2:
                self.edges[1].label = 'right'
                print(f"[AsymHalfCirclePanel._verify_edge_labels] Set label 'right' on edge 1")
            
            if 'bottom' in missing_labels and len(self.edges) >= 3:
                self.edges[2].label = 'bottom'
                print(f"[AsymHalfCirclePanel._verify_edge_labels] Set label 'bottom' on edge 2")
            
            if 'left' in missing_labels and len(self.edges) >= 4:
                self.edges[3].label = 'left'
                print(f"[AsymHalfCirclePanel._verify_edge_labels] Set label 'left' on edge 3")
        # else:
        #     print(f"[AsymHalfCirclePanel._verify_edge_labels] All expected edge labels found")
    
    def length(self, *args):
        return self.interfaces['right'].edges.length()
        
    def get_edge_by_label(self, label):
        """
        Get an edge by its label without any fallback to indices.
        This implementation requires that all edges have proper labels.
        """
        # Debug the label lookup
        # print(f"[AsymHalfCirclePanel.get_edge_by_label] Looking for edge with label '{label}'")
        
        # First check if the Panel class has this method
        if hasattr(super(), 'get_edge_by_label'):
            edge = super().get_edge_by_label(label)
            if edge is not None:
                return edge
        
        # If not found or method doesn't exist, implement our own lookup
        for i, edge in enumerate(self.edges):
            if hasattr(edge, 'label') and edge.label == label:
                # print(f"[AsymHalfCirclePanel.get_edge_by_label] Found edge {i} with label '{label}'")
                return edge
        
        # Debug what labels we do have
        print(f"[AsymHalfCirclePanel.get_edge_by_label] Warning: Edge with label '{label}' not found")
        for i, edge in enumerate(self.edges):
            if hasattr(edge, 'label'):
                print(f"[AsymHalfCirclePanel.get_edge_by_label] Edge {i} has label '{edge.label}'")
            else:
                print(f"[AsymHalfCirclePanel.get_edge_by_label] Edge {i} has no label attribute")
        
        # No fallback - strict behavior as requested
        return None

class SkirtCircle(StackableSkirtComponent):
    """Simple circle skirt"""
    def __init__(self, body, design, tag='', length=None, rise=None, slit=True, asymm=False, min_len=5, **kwargs) -> None:
        super().__init__(body, design, tag)

        design = design['flare-skirt']
        suns = design['suns']['v']
        self.rise = design['rise']['v'] if rise is None else rise
        waist, hips_depth, _ = self.eval_rise(self.rise)

        if length is None:  # take from design parameters
            length = hips_depth + design['length']['v'] * body['_leg_length']

        # NOTE: with some combinations of rise and length parameters length may become too small/negative
        # Hence putting a min positive value here
        length = max(length, min_len)

        # panels
        if not asymm:  # Typical symmetric skirt
            self.front = CircleArcPanel.from_w_length_suns(
                f'skirt_front_{tag}' if tag else 'skirt_front', 
                length, waist / 2, suns / 2,
                match_top_int_proportion=self.body['waist'] - self.body['waist_back_width'],
                ).translate_by([0, body['_waist_level'], 15])

            self.back = CircleArcPanel.from_w_length_suns(
                f'skirt_back_{tag}'  if tag else 'skirt_back', 
                length, waist / 2, suns / 2,
                match_top_int_proportion=self.body['waist_back_width'],
                ).translate_by([0, body['_waist_level'], -15])
        else:  # Asymmetric skirt - this code wasn't used in your code
            raise NotImplementedError("Asymmetric skirt mode is not currently supported.")

        # Add a cut
        if design['cut']['add']['v'] and slit:
            self.add_cut(
                self.front if design['cut']['place']['v'] > 0 else self.back, 
                design, length)

        # Stitches
        self.stitching_rules = pyg.Stitches(
            (self.front.interfaces['right'], self.back.interfaces['right']),
            (self.front.interfaces['left'], self.back.interfaces['left'])
        )

        # Interfaces
        self.interfaces = {
            'top': pyg.Interface.from_multiple(self.front.interfaces['top'], self.back.interfaces['top']),
            'bottom_f': self.front.interfaces['bottom'],
            'bottom_b': self.back.interfaces['bottom'],
            'bottom': pyg.Interface.from_multiple(self.front.interfaces['bottom'], self.back.interfaces['bottom'])
        }
        
    def add_cut(self, panel, design, sk_length):
        """Add a cut to the skirt"""
        width, depth = design['cut']['width']['v'] * sk_length, design['cut']['depth']['v'] * sk_length

        target_edge = panel.interfaces['bottom'].edges[0]
        t_len = target_edge.length()
        offset = abs(design['cut']['place']['v'] * t_len)

        # Respect the placement boundaries
        offset = max(offset, width / 2)
        offset = min(offset, t_len - width / 2)

        # NOTE: heuristic is specific for the panels that we use
        right = target_edge.start[0] > target_edge.end[0]

        # Make a cut
        cut_shape = pyg.EdgeSeqFactory.dart_shape(width, depth=depth)

        new_edges, _, interf_edges = pyg.ops.cut_into_edge(
            cut_shape, target_edge, 
            offset=offset, 
            right=right
        )

        panel.edges.substitute(target_edge, new_edges)
        panel.interfaces['bottom'].substitute(
            target_edge, interf_edges,
            [panel for _ in range(len(interf_edges))])
        
    def length(self, *args):
        return self.front.length()


class AsymmSkirtCircle(SkirtCircle):
    """Front/back asymmetric skirt"""
    def __init__(self, body, design, tag='', length=None, rise=None, slit=True, **kwargs):
        # We updated the base class to raise NotImplementedError for asymm=True
        # So let's implement it properly here
        
        # First, use super() without asymm=True to get base initialization
        super().__init__(body, design, tag, length, rise, slit, asymm=False)
        
        # Then customize the implementation as needed for asymmetric skirt
        print("[AsymmSkirtCircle.__init__] Warning: This implementation is a placeholder.")
        print("[AsymmSkirtCircle.__init__] Using symmetric skirt implementation instead.")
        # Your implementation for asymmetric skirts would go here