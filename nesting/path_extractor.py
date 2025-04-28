import numpy as np
import svgpathtools as svgpath
import os
import json
import copy

from pygarment.pattern.core import BasicPattern
from nesting import utils

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
        Returns the outline (path) of the given panel as a list of [x, y] points,
        shifted so that its top-left vertex becomes (0,0).
        """
        panel = self.pattern['panels'][panel_name]
        vertices = panel['vertices']
        outline = []

        for edge in panel['edges']:
            curve = self._edge_as_curve(vertices, edge)

            if not outline:
                p0 = curve.point(0)
                outline.append([p0.real, -p0.imag])

            for t in np.linspace(0, 1, samples_per_edge, endpoint=False)[1:]:
                p = curve.point(t)
                outline.append([p.real, -p.imag])

            p1 = curve.point(1)
            outline.append([p1.real, -p1.imag])

        # shift the outline so that its TOP LEFT vertex of the bounding box is at (0,0)
        # consistent with the canvas coordinate system
        xs = [pt[0] for pt in outline]
        ys = [pt[1] for pt in outline]
        min_x = min(xs)
        min_y = min(ys)
        shifted_outline = [[x - min_x, y - min_y] for x, y in outline]

        if utils._signed_area(shifted_outline) < 0:        # currently CCW → reverse
            shifted_outline.reverse()

        return shifted_outline


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


# if __name__ == '__main__':
#     pattern_file = "/Users/aysegulbarlas/codestuff/GarmentCode/nesting-assets/Configured_design_specification.json"
#     extractor = PatternPathExtractor(pattern_file)
#     all_outlines = extractor.get_all_panel_outlines(samples_per_edge=20)
#     for name, outline in all_outlines.items():
#         print(f"Panel: {name}")
#         for pt in outline:
#             print(pt)


