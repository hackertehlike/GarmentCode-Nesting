import numpy as np
import svgpathtools as svgpath
import os
import json
import copy

from pygarment.pattern.core import BasicPattern
from nesting import utils
from nesting import layout
from nesting.layout import Piece


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
        """
        Returns the Piece object representing the given panel as a list of [x, y] points,
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
        # xs = [pt[0] for pt in outline]
        # ys = [pt[1] for pt in outline]
        # min_x = float(min(xs))
        # min_y = float(min(ys))
        
        # shifted_outline = [(x - min_x, y - min_y) for x, y in outline]

        shifted_outline = utils.shift_coordinates(outline)

        if utils.signed_area(shifted_outline) < 0:        # currently CCW → reverse
            # print(panel_name, "is CCW, reversing it")
            shifted_outline.reverse()
        # else:
            # print(panel_name, "is CW, keeping it as is")

        # Create a Piece object for the panel outline
        piece = Piece(shifted_outline, id=panel_name)
        return piece


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


