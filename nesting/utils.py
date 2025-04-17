from __future__ import annotations
from typing import List, Tuple, Iterable
import math
import pyclipper

_SCALE = 1_000  # three‑decimal precision; adjust as required

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
    subj = to_clipper(contour, _SCALE)
    if allowance == 0:
        return [from_clipper(subj, _SCALE)]

    pco   = pyclipper.PyclipperOffset(miter_limit = miter_limit)
    pco.AddPath(subj, join_type, pyclipper.ET_CLOSEDPOLYGON)
    delta = int(round(allowance * _SCALE))
    solution = pco.Execute(delta)
    return [from_clipper(p, _SCALE) for p in solution]


def polygons_overlap(poly_a, poly_b, area_tol = 1e-12) -> bool:
    """
    True iff poly_a and poly_b overlap with **positive area**.
    Touching at an edge or point returns False.
    """
    a = to_clipper(poly_a, _SCALE)
    b = to_clipper(poly_b, _SCALE)
    pc = pyclipper.Pyclipper()
    pc.AddPath(a, pyclipper.PT_SUBJECT, True)
    pc.AddPath(b, pyclipper.PT_CLIP,    True)
    inter = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_NONZERO,
                       pyclipper.PFT_NONZERO)
    for path in inter:
        if abs(pyclipper.Area(path)) > area_tol * _SCALE * _SCALE:
            return True
    return False


def no_fit_polygon(stationary, moving):
    """
    Compute the No‑Fit Polygon of *moving* about *stationary*.
    Returned coordinates are floats in the original units.
    """
    A = to_clipper(stationary, _SCALE)
    # Reflect B through the origin for Minkowski difference
    B = [(-x, -y) for x, y in moving]
    B = to_clipper(B, _SCALE)

    # The boolean flag 'True' tells Clipper the paths are closed polygons
    nfp_paths = pyclipper.MinkowskiSum(A, B, True)
    return [from_clipper(p, _SCALE) for p in nfp_paths]
