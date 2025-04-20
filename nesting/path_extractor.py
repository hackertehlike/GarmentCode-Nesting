import numpy as np
import svgpathtools as svgpath
import os
import json
import copy

from pygarment.pattern.core import BasicPattern


class PatternPathExtractor(BasicPattern):
    """
    Loads a garment pattern specification from a JSON file (via BasicPattern)
    and provides methods to extract an approximate outline (path) for each panel
    as a list of [x, y] vertices.
    """

    def __init__(self, pattern_file):
        # Load and normalize the pattern via BasicPattern.
        super().__init__(pattern_file)
        
    def _flip_y(self, p):
        return p

    def get_panel_outline(self, panel_name, samples_per_edge=10):
        """
        Returns the outline (path) of the given panel as a list of [x, y] points.
        For each edge, the associated curve is sampled to approximate curves.
        
        Args:
            panel_name: The key/name of the panel in the pattern spec.
            samples_per_edge: Number of points sampled per edge (including start and end).
            
        Returns:
            A list of [x, y] coordinates approximating the panel's boundary.
        """
        panel = self.pattern['panels'][panel_name]
        vertices = panel['vertices']
        outline = []

        # Assume that panel['edges'] is ordered as the panel's edge loop.
        for edge in panel['edges']:
            # Use the internal method _edge_as_curve to get an svgpath object for this edge.
            curve = self._edge_as_curve(vertices, edge)

            # For the first edge, add the starting point.
            if not outline:
                pt_start = curve.point(0)
                outline.append([pt_start.real, pt_start.imag])
            
            # Generate intermediate samples.
            ts = np.linspace(0, 1, samples_per_edge, endpoint=False)[1:]
            for t in ts:
                pt = curve.point(t)
                outline.append([pt.real, pt.imag])
            
            # Ensure the curve's end point is added.
            pt_end = curve.point(1)
            outline.append([pt_end.real, pt_end.imag])
        
        return outline

    def get_all_panel_outlines(self, samples_per_edge=10):
        """
        Returns a dictionary mapping each panel name to its outline (list of [x, y] points).
        
        Args:
            samples_per_edge: Number of sampling points per edge.
        
        Returns:
            A dict where keys are panel names and values are lists of [x, y] points.
        """
        outlines = {}
        for panel_name in self.pattern['panels']:
            outlines[panel_name] = self.get_panel_outline(panel_name, samples_per_edge)
        return outlines


if __name__ == '__main__':
    # Replace with the path to your pattern JSON file.
    pattern_file = "/Users/aysegulbarlas/codestuff/GarmentCode/nesting-assets/Configured_design_specification.json"
    extractor = PatternPathExtractor(pattern_file)
    all_outlines = extractor.get_all_panel_outlines(samples_per_edge=20)
    for name, outline in all_outlines.items():
        print(f"Panel: {name}")
        for pt in outline:
            print(pt)