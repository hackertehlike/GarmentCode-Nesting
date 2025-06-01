import numpy as np
import svgpathtools as svgpath
import os
import json
import copy

from pygarment.pattern.core import BasicPattern
from nesting import utils
from nesting import layout
from nesting.layout import Piece
from svgpathtools import Line

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
    
    def _edge_as_curve(self, vertices: list, edge: dict):
        base_curve = super()._edge_as_curve(vertices, edge)

        if isinstance(base_curve, svgpath.Arc):
            return svgpath.Arc(
                base_curve.start,
                base_curve.radius,
                rotation=base_curve.rotation,
                large_arc=base_curve.large_arc,
                sweep=not base_curve.sweep,
                end=base_curve.end
            )

        # All other segment types (Line, CubicBezier, QuadraticBezier) are
        # unaffected by the reflection, so keep them as they are.
        return base_curve

    
    def _get_panel_outline(self, panel_name, samples_per_edge=10):
        panel = self.pattern['panels'][panel_name]
        vertices = panel['vertices']
        outline = []

        for edge in panel['edges']:
            curve = self._edge_as_curve(vertices, edge)

            # If it's a straight‐line segment, only add start and end:
            if isinstance(curve, Line):
                p0 = curve.point(0)
                p1 = curve.point(1)
                outline.append([p0.real, -p0.imag])
                outline.append([p1.real, -p1.imag])
                continue

            # Otherwise (Arc, CubicBezier, QuadraticBezier), sample normally:
            p0 = curve.point(0)
            outline.append([p0.real, -p0.imag])

            for t in np.linspace(0, 1, samples_per_edge, endpoint=False)[1:]:
                p = curve.point(t)
                outline.append([p.real, -p.imag])

            p1 = curve.point(1)
            outline.append([p1.real, -p1.imag])

        shifted_outline = utils.shift_coordinates(outline)
        if utils.signed_area(shifted_outline) < 0:
            shifted_outline.reverse()

        return Piece(shifted_outline, id=panel_name)



    def get_all_panel_pieces(self, samples_per_edge=10) -> dict[str, Piece]:
        """
        Returns a dictionary mapping each panel name to its outline (list of [x, y] points).
        
        Args:
            samples_per_edge: Number of sampling points per edge.
        
        Returns:
            A dict of Piece objects keyed by their name/id, each representing a panel.
        """
        all_pieces = {}
        for panel_name in self.pattern['panels']:
            piece = self._get_panel_outline(panel_name, samples_per_edge)
            all_pieces[panel_name] = piece
        return all_pieces


# if __name__ == '__main__':
#     pattern_file = "/Users/aysegulbarlas/codestuff/GarmentCode/nesting-assets/Configured_design_specification.json"
#     extractor = PatternPathExtractor(pattern_file)
#     all_outlines = extractor.get_all_panel_outlines(samples_per_edge=20)
#     for name, outline in all_outlines.items():
#         print(f"Panel: {name}")
#         for pt in outline:
#             print(pt)


