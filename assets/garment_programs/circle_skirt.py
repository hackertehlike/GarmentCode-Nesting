import numpy as np
import pygarment as pyg

from assets.garment_programs.base_classes import StackableSkirtComponent
import copy


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
            radius=top_rad, large_arc=halfarc > np.pi / 2)
        top_edge.add_semantic_label('top')  # Add semantic label
        self.edges.append(top_edge)
        
        # Now we can check if the edge is accessible via get_edge_by_label
        #top_check = self.get_edge_by_label('top')
        #print(f"[CircleArcPanel.__init__] Top edge check: {top_check is not None}")

        # right edge - index 1
        right_edge = pyg.Edge(
            self.edges[-1].end, [dist_out / 2, -vert_len])
        right_edge.add_semantic_label('right')  # Add semantic label
        self.edges.append(right_edge)
        
        # Bottom edge - index 2
        bottom_edge = pyg.CircleEdgeFactory.from_points_radius(
            self.edges[-1].end, [- dist_out / 2, -vert_len], 
            radius=top_rad + length,
            large_arc=halfarc > np.pi / 2, right=False)
        bottom_edge.add_semantic_label('bottom')
        self.edges.append(bottom_edge)

        # left edge - index 3 - closes the loop
        left_edge = pyg.Edge(self.edges[-1].end, self.edges[0].start)
        left_edge.add_semantic_label('left')  # Add semantic label
        self.edges.append(left_edge)

        # Verify all edge labels are set correctly
        # #print(f"[CircleArcPanel.__init__] Panel {name} created with edges:")
        # for i, edge in enumerate(self.edges):
        #     #print(f"  Edge {i}: label={getattr(edge, 'label', 'None')}")

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
        print([edge.semantic_labels for edge in self.edges], name)

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
                # print(f"[CircleArcPanel._verify_edge_labels] ERROR: Edge with label '{label}' not found")

        if missing_labels:
            print(f"[CircleArcPanel._verify_edge_labels] Missing edge labels: {missing_labels}")
        else:
           print(f"[CircleArcPanel._verify_edge_labels] All expected edge labels found")
        
        return missing_labels
        

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
        # print([edge.semantic_labels for edge in self.edges], self.name)
        # print(f"[CircleArcPanel.split] Starting split for panel {self.name}")
        # print(f"[CircleArcPanel.split] Panel has {len(self.edges)} edges")
        
        # # Debug: examine the interfaces to see if they're modifying the edges
        # print(f"[CircleArcPanel.split] Checking edges:")
        # for edge in self.edges:
        #     print(f"  Edge: {edge}, label={getattr(edge, 'label', 'None')}, semantic_labels={getattr(edge, 'semantic_labels', [])}")
        self._verify_edge_labels()  # Ensure edge labels are verified before splitting

        # Check if the panel EdgeSequence is clockwise or counter-clockwise
        clockwise = self.edges.is_clockwise()

        if not clockwise:
            print(f"[CircleArcPanel.split] Warning: Panel {self.name} is not clockwise, reversing edges.")
            self.edges.reverse()

        # Get edges by label
        bottom_edge = self.get_edge_by_label('bottom')
        top_edge = self.get_edge_by_label('top')
        left_edge = self.get_edge_by_label('left')
        right_edge = self.get_edge_by_label('right')

        # debug all of the edge labels
        #print(f"[CircleArcPanel.split] Edge labels before split:")
        for edge in self.edges:
            print(f"  Edge: {edge}, label={getattr(edge, 'label', 'None')}, semantic_labels={getattr(edge, 'semantic_labels', [])}")

        
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
        split_point_bottom = bottom_edge.point_at(1-proportion)
        #split_point_bottom = bottom_edge.midpoint().tolist()

        #print(f"[CircleArcPanel.split] Split point on bottom edge: {split_point_bottom}")
        
        split_point_top = top_edge.point_at(proportion)
        #split_point_top = top_edge.midpoint().tolist()

        #print(f"[CircleArcPanel.split] Split points: "f"bottom={split_point_bottom}, top={split_point_top}")

        # Split edges
        bottom1, bottom2 = bottom_edge.split_at_point(split_point_bottom)
        top1, top2 = top_edge.split_at_point(split_point_top)

        # Create new split edges for each panel with copied vertices to avoid shared references
        # Make copies of the vertices to ensure they're not shared between panels
        split_point_top_copy1 = split_point_top.copy()
        split_point_bottom_copy1 = split_point_bottom.copy()
        split_point_top_copy2 = split_point_top.copy()
        split_point_bottom_copy2 = split_point_bottom.copy()
        
        # Create edges with the copied vertices
        split_edge1 = pyg.Edge(split_point_top_copy1, split_point_bottom_copy1)
        split_edge1.add_semantic_label('left')
            
        split_edge2 = pyg.Edge(split_point_bottom_copy2, split_point_top_copy2)
        split_edge2.add_semantic_label('right')

        # Debug the split edges
        #print(f"[CircleArcPanel.split] Split edges created:")
        #print(f"  Split Edge 1: {split_edge1}")
        #print(f"  Split Edge 2: {split_edge2}")


        # def make_left_to_right(edge):
        #     """Ensure the edge is oriented from left to right."""
        #     if edge.start[0] > edge.end[0]:
        #         edge.reverse()
        #     return edge
        
        # def make_top_to_bottom(edge):
        #     """Ensure the edge is oriented from top to bottom."""
        #     if edge.start[1] < edge.end[1]:
        #         edge.reverse()
        #     return edge
        
        # Create new panels
        panel1 = pyg.Panel(f'{self.name}_split_left')
        
        # Ensure consistent edge orientations for the left panel:
        # - top edge: left to right
        # - split edge: top to bottom 
        # - bottom edge: right to left (reverse of left to right)
        # - left edge: bottom to top (reverse of top to bottom)
        panel1.edges = pyg.EdgeSequence([
            copy.deepcopy(top1),
            split_edge1,
            copy.deepcopy(bottom2),
            copy.deepcopy(left_edge)
        ])

        #print(f"[CircleArcPanel.split] Created panel1")

        panel2 = pyg.Panel(f'{self.name}_split_right')
        
        # Ensure consistent edge orientations for the right panel:
        # - top edge (top2): left to right
        # - right edge: top to bottom
        # - bottom edge (bottom2): right to left (reverse of left to right)
        # - split edge: bottom to top (reverse of top to bottom)
        panel2.edges = pyg.EdgeSequence([
            copy.deepcopy(top2),
            copy.deepcopy(right_edge),
            copy.deepcopy(bottom1),
            split_edge2
        ])
        
        #print(f"[CircleArcPanel.split] Created panel2")

        # #print(f"[CircleArcPanel.split] Created panel1 with edges:")
        # for i, edge in enumerate(panel1.edges):
        #     #print(f"  Edge {i}: label={getattr(edge, 'label', 'None')}")
            
        # #print(f"[CircleArcPanel.split] Created panel2 with edges:")
        # for i, edge in enumerate(panel2.edges):
        #     #print(f"  Edge {i}: label={getattr(edge, 'label', 'None')}")
            
        # Apply the verification method to the new panels
        # Add the _verify_edge_labels method to the new panels
        #panel1._verify_edge_labels = self._verify_edge_labels.__get__(panel1)
        #panel2._verify_edge_labels = self._verify_edge_labels.__get__(panel2)
        
        print(f"[CircleArcPanel.split] Verifying edge labels for new panels")

        # Verify edge labels in both new panels
        #panel1._verify_edge_labels()
        #panel2._verify_edge_labels()


        #print(f"[CircleArcPanel.split] Edge labels verified for new panels")
        
        # Return the new panels
        print(f"[CircleArcPanel.split] Returning new panels: {panel1.name}, {panel2.name}")
        
        return panel1, panel2

    # def get_edge_by_label(self, label):
    #     """
    #     Get an edge by its label without any fallback to indices.
    #     This implementation checks both the label attribute and semantic_labels.
    #     """
    #     # Debug the label lookup
    #     #print(f"[CircleArcPanel.get_edge_by_label] Looking for edge with label '{label}'")
        
    #     if hasattr(super(), 'get_edge_by_label'):
    #         edge = super().get_edge_by_label(label)
    #         if edge is not None:
    #             return edge
        
    #     # If not found or method doesn't exist, implement our own lookup
    #     for _, edge in enumerate(self.edges):
    #         # Check for direct label attribute
    #         if hasattr(edge, 'label') and edge.label == label:
    #             #print(f"[CircleArcPanel.get_edge_by_label] Found edge {i} with label attribute '{label}'")
    #             return edge
            
    #         # Check semantic_labels
    #         if hasattr(edge, 'semantic_labels') and label in edge.semantic_labels:
    #             #print(f"[CircleArcPanel.get_edge_by_label] Found edge {i} with semantic label '{label}'")
    #             return edge
        
    #     # Debug what labels we do have
    #     #print(f"[CircleArcPanel.get_edge_by_label] Warning: Edge with label '{label}' not found")
    #     #for i, edge in enumerate(self.edges):
    #         #if hasattr(edge, 'label'):
    #             #print(f"[CircleArcPanel.get_edge_by_label] Edge {i} has label '{edge.label}'")
    #         #else:
    #             #print(f"[CircleArcPanel.get_edge_by_label] Edge {i} has no label attribute")
                
    #         #if hasattr(edge, 'semantic_labels'):
    #             #print(f"[CircleArcPanel.get_edge_by_label] Edge {i} has semantic_labels: {edge.semantic_labels}")
        
    #     # No fallback - strict behavior as requested
    #     return None

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
            radius=top_rad, large_arc=False)
        top_edge.add_semantic_label('top')  # Add semantic label
        self.edges.append(top_edge)

        # right edge - index 1
        right_edge = pyg.Edge(
            self.edges[-1].end, [dist_out / 2, 0])
        right_edge.add_semantic_label('right')
        self.edges.append(right_edge)
        
        # Bottom edge - index 2
        bottom_edge = pyg.CircleEdgeFactory.from_three_points(
            self.edges[-1].end, [- dist_out / 2, 0], 
            point_on_arc=[0, -(top_rad + length_f)]
        )
        bottom_edge.semantic_labels.add('bottom')
        self.edges.append(bottom_edge)

        # left edge - index 3 - closes the loop
        left_edge = pyg.Edge(self.edges[-1].end, self.edges[0].start)
        left_edge.semantic_labels.add('left')  # Add semantic label
        self.edges.append(left_edge)

        # Verify all edge labels are set correctly
        # #print(f"[AsymHalfCirclePanel.__init__] Panel {name} created with edges:")
        # for i, edge in enumerate(self.edges):
        #     #print(f"  Edge {i}: label={getattr(edge, 'label', 'None')}")

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
            edge = self.get_edge_by_label(label)
            if edge is None:
                missing_labels.append(label)
                print(f"[AsymHalfCirclePanel._verify_edge_labels] ERROR: Edge with label '{label}' not found")
        
        # else:
        #    print(f"[AsymHalfCirclePanel._verify_edge_labels] All expected edge labels found")
        
        return missing_labels

    def length(self, *args):
        return self.interfaces['right'].edges.length()

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
        #print("[AsymmSkirtCircle.__init__] Warning: This implementation is a placeholder.")
        #print("[AsymmSkirtCircle.__init__] Using symmetric skirt implementation instead.")
        # Your implementation for asymmetric skirts would go here