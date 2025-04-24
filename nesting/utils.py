from __future__ import annotations
from typing import List, Tuple, Iterable
import math
import pyclipper

# clipper uses int coordinates, so we need to scale our floats
# for nesting purposes 3 decimal precision is sufficient
_SCALE = 1_000  # three‑decimal precision

def to_clipper(path):
    """Convert a float path to IntPoint (int) for Pyclipper."""
    return [(int(round(x * _SCALE)), int(round(y * _SCALE))) for x, y in path]

def from_clipper(path):
    """Convert an IntPoint path back to floats."""
    inv = 1.0 / _SCALE
    return [(x * inv, y * inv) for x, y in path]


def add_seam_allowance(contour, allowance = 1.0, join_type = pyclipper.JT_MITER, miter_limit: float = 2.0):
    """
    Offset *contour* outward by *allowance* (in the same units as the input).
    Negative allowance contracts the shape.
    """
    subj = to_clipper(contour)
    if allowance == 0:
        return [from_clipper(subj)]

    pco   = pyclipper.PyclipperOffset(miter_limit = miter_limit)
    pco.AddPath(subj, join_type, pyclipper.ET_CLOSEDPOLYGON)
    delta = int(round(allowance * _SCALE))
    solution = pco.Execute(delta)
    return [from_clipper(p) for p in solution]


def polygons_overlap(poly_a, poly_b, area_tol = 1e-12) -> bool:
    """
    True iff poly_a and poly_b overlap with **positive area**.
    Touching at an edge or point returns False.
    """
    a = to_clipper(poly_a)
    b = to_clipper(poly_b)
    pc = pyclipper.Pyclipper()
    pc.AddPath(a, pyclipper.PT_SUBJECT, True)
    pc.AddPath(b, pyclipper.PT_CLIP,    True)
    inter = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_NONZERO,
                       pyclipper.PFT_NONZERO)
    for path in inter:
        if abs(pyclipper.Area(path)) > area_tol * _SCALE * _SCALE:
            return True
    return False

def _translate_polygon(poly, dx, dy):
    """Return a *new* polygon translated by (dx, dy)."""
    return [(x + dx, y + dy) for (x, y) in poly]


def no_fit_polygon(stationary, moving):
    """
    Compute the No‑Fit Polygon of *moving* about *stationary*.
    Returned coordinates are floats in the original units.
    """
    A = to_clipper(stationary)
    # Reflect B through the origin for Minkowski difference
    B = [(-x, -y) for x, y in moving]
    B = to_clipper(B)

    # The boolean flag 'True' tells Clipper the paths are closed polygons
    nfp_paths = pyclipper.MinkowskiSum(A, B, True)
    return [from_clipper(p) for p in nfp_paths]

def polygon_area(poly):
    """
    Calculate the area of a polygon using pyclipper.
    The polygon is represented as a list of (x, y) tuples.
    The area is returned as a float.
    """
    if len(poly) < 3:
        return 0.0  # Not a polygon
    area = pyclipper.Area(to_clipper(poly))
    return area * _SCALE * _SCALE

def _signed_area(poly):
    """
    Shoelace formula.  Positive ⇒ clockwise winding,
    negative ⇒ counter-clockwise, zero ⇒ degenerate.
    """
    area = 0.0
    for (x0, y0), (x1, y1) in zip(poly, poly[1:] + [poly[0]]):
        area += x0 * y1 - x1 * y0
    return area * 0.5