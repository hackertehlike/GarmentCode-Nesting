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

        #split_history = {}

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
        # print([edge.semantic_labels for edge in self.edges], name)

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

        # if missing_labels:
        #     print(f"[CircleArcPanel._verify_edge_labels] Missing edge labels: {missing_labels}")
        # else:
        #    print(f"[CircleArcPanel._verify_edge_labels] All expected edge labels found")
        
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

    def get_all_edges_by_label(self, label):
        """Find all edges with a given label"""
        edges = []
        for edge in self.edges:
            if edge.label == label or label in edge.semantic_labels:
                edges.append(edge)
        return edges
    
    def get_appropriate_edge(self, edges, proportion):
        """
        Select the appropriate edge from multiple edges with the same label.
        This works by treating the edges as if they were one long edge and finding
        the edge that contains the point at the given proportion of the total length.
        
        Args:
            edges: List of edges with the same label
            proportion: Proportion along the combined edge length (0-1)
            
        Returns:
            (edge, local_proportion): The selected edge and the proportion within that edge
        """
        if not edges:
            return None, None
        
        if len(edges) == 1:
            return edges[0], proportion
        
        # Calculate total length of all edges
        total_length = sum(edge.length() for edge in edges)
        target_length = total_length * proportion
        
        # Find which edge contains the target point
        current_length = 0
        for edge in edges:
            edge_length = edge.length()
            if current_length + edge_length >= target_length:
                # This edge contains our target point
                local_proportion = (target_length - current_length) / edge_length
                return edge, local_proportion
            current_length += edge_length
            
        # If we get here, return the last edge
        return edges[-1], 1.0
    
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
        #    print(f"[CircleArcPanel.split] Warning: Panel {self.name} is not clockwise, reversing edges.")
            self.edges.reverse()

        # Get all edges by label
        bottom_edges = self.get_all_edges_by_label('bottom')
        top_edges = self.get_all_edges_by_label('top')
        left_edge = self.get_edge_by_label('left')
        right_edge = self.get_edge_by_label('right')
        
        # Get the appropriate edges and proportions for top and bottom
        bottom_edge, bottom_prop = self.get_appropriate_edge(bottom_edges, 1-proportion)
        top_edge, top_prop = self.get_appropriate_edge(top_edges, proportion)

        #debug all of the edge labels
        print(f"[CircleArcPanel.split] Edge labels before split:")
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

        # Get split points using the local proportion for each edge
        split_point_bottom = bottom_edge.point_at(bottom_prop)
        split_point_top = top_edge.point_at(top_prop)

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
        
        # print(f"[CircleArcPanel.split] Verifying edge labels for new panels")

        # Verify edge labels in both new panels
        #panel1._verify_edge_labels()
        #panel2._verify_edge_labels()


        #print(f"[CircleArcPanel.split] Edge labels verified for new panels")

        # add split history
        # self.split_history.append({
        #     #'panel_name': self.name,
        #     'proportion': proportion,
        #     'new_panels': [panel1, panel2]
        # })
        
        # Return the new panels
        print(f"[CircleArcPanel.split] Returning new panels: {panel1.name}, {panel2.name}")
        
        return panel1, panel2
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
        bottom_edge.add_semantic_label('bottom')
        self.edges.append(bottom_edge)

        # left edge - index 3 - closes the loop
        left_edge = pyg.Edge(self.edges[-1].end, self.edges[0].start)
        left_edge.add_semantic_label('left')  # Add semantic label
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
        
    def get_all_edges_by_label(self, label):
        """Find all edges with a given label"""
        edges = []
        for edge in self.edges:
            if edge.label == label or label in edge.semantic_labels:
                edges.append(edge)
        return edges
    
    def get_appropriate_edge(self, edges, proportion):
        """
        Select the appropriate edge from multiple edges with the same label.
        This works by treating the edges as if they were one long edge and finding
        the edge that contains the point at the given proportion of the total length.
        
        Args:
            edges: List of edges with the same label
            proportion: Proportion along the combined edge length (0-1)
            
        Returns:
            (edge, local_proportion): The selected edge and the proportion within that edge
        """
        if not edges:
            return None, None
        
        if len(edges) == 1:
            return edges[0], proportion
        
        # Calculate total length of all edges
        total_length = sum(edge.length() for edge in edges)
        target_length = total_length * proportion
        
        # Find which edge contains the target point
        current_length = 0
        for edge in edges:
            edge_length = edge.length()
            if current_length + edge_length >= target_length:
                # This edge contains our target point
                local_proportion = (target_length - current_length) / edge_length
                return edge, local_proportion
            current_length += edge_length
            
        # If we get here, return the last edge
        return edges[-1], 1.0

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
            self.back.edges[0].add_semantic_label('top')
            #print(f"Back top edge labels immediately after adding: {self.back.edges[0].semantic_labels}")

        else:  # Asymmetric skirt - this code wasn't used in your code
            raise NotImplementedError("Asymmetric skirt mode is not currently supported.")

        # Add a cut
        if design['cut']['add']['v'] and slit:
            #print(f"Back top edge labels before cut: {self.back.edges[0].semantic_labels}")
            self.add_cut(
                self.front if design['cut']['place']['v'] > 0 else self.back, 
                design, length)
            #print(f"Back top edge labels after cut: {self.back.edges[0].semantic_labels}")

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

        #new_edges.add_semantic_label('top')
        #interf_edges.add_semantic_label('top')


        for edge in new_edges:
            edge.add_semantic_label('top')
        for edge in interf_edges:
            edge.add_semantic_label('top')
        

        panel.edges.substitute(target_edge, new_edges)
        panel.interfaces['bottom'].substitute(
            target_edge, interf_edges,
            [panel for _ in range(len(interf_edges))])
        

        
    def length(self, *args):
        return self.front.length()


class AsymmSkirtCircle(SkirtCircle):
    """Front/back asymmetric skirt"""
    def __init__(self, body, design, tag='', length=None, rise=None, slit=True, **kwargs):
        
        # First, use super() without asymm=True to get base initialization
        super().__init__(body, design, tag, length, rise, slit, asymm=False)