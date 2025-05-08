from __future__ import annotations
from typing import List, Tuple, Iterable
import math
import pyclipper
# from .layout import Layout, Container, Piece
# from .layout import Container, Piece
# from CGAL.CGAL_Polygon_2 import Polygon_2
from CGAL.CGAL_Alpha_shape_2 import (Alpha_shape_2, Alpha_shape_2_Edge,
                                     REGULARIZED, GENERAL)
from CGAL.CGAL_Alpha_shape_2 import Alpha_shape_2, REGULAR, SINGULAR 
from CGAL import CGAL_Kernel

Point_2 = CGAL_Kernel.Point_2

# clipper uses int coordinates, so we need to scale our floats
# for nesting purposes 3 decimal precision is sufficient
_SCALE = 1000  # three‑decimal precision
CW     = pyclipper.Orientation   # True for clockwise in pyclipper

def to_clipper(path):
    """Convert a float path to IntPoint (int) for Pyclipper."""
    return [(int(round(x * _SCALE)), int(round(y * _SCALE))) for x, y in path]

def from_clipper(path):
    """Convert an IntPoint path back to floats."""
    inv = 1.0 / _SCALE
    return [(x * inv, y * inv) for x, y in path]


# moved to layout.py as a method of Piece
# def add_seam_allowance(piece, allowance = 1.0, join_type = pyclipper.JT_MITER, miter_limit: float = 2.0) -> None:
#     """
#     Updates the piece's outer path with the seam allowance.
#     The piece's inner path is unchanged.
#     """
#     contour = piece.get_inner_path()
#     if not contour or len(contour) < 3:
#         raise ValueError("Piece has no inner path to offset.")
    
#     print(f"Adding seam allowance of {allowance} to piece {piece.id}")

#     subj = to_clipper(contour)
#     if allowance == 0:
#         return [from_clipper(subj)]

#     pco   = pyclipper.PyclipperOffset(miter_limit = miter_limit)
#     pco.AddPath(subj, join_type, pyclipper.ET_CLOSEDPOLYGON)
#     delta = int(round(allowance * _SCALE))
#     solution = pco.Execute(delta)

#     offset_paths = []
#     for path in solution:
#         outline = from_clipper(path)

#         xs = [pt[0] for pt in outline]
#         ys = [pt[1] for pt in outline]
#         min_x = min(xs)
#         min_y = min(ys)
#         shifted_outline = [(x - min_x, y - min_y) for x, y in outline]

#         offset_paths+=shifted_outline

#     # Update the piece's outer path with the offset paths
#     piece.outer_path = offset_paths

#     # print("Updating bounding box of the piece")
#     piece.update_bbox()

#     print (f"Piece {piece.id} outer path updated with seam allowance")


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
    No-Fit Polygon of *moving* about *stationary*.
    Coordinates returned as floats in the original units.
    """
    
    A = to_clipper(stationary)
    B = [(-x, -y) for x, y in moving]          # reflect the moving part
    B = to_clipper(B)
    nfp = pyclipper.MinkowskiSum(B, A, True)

    # 4. Back to floating-point
    # return [entry for p in nfp for entry in from_clipper(p)] #[from_clipper(p) for p in nfp] 
    return flatten(nfp)

# def flatten_piece_list(pieces: List[Piece]) -> List[Tuple[float, float]]:
#     return [(x + piece.translation[0], y + piece.translation[1]) for piece in pieces for x, y in piece.get_outer_path()]

def flatten(xss):
    """
    Flatten a list of lists into a single list.
    """
    return [x for xs in xss for x in from_clipper(xs)]

def polygon_area(poly):
    """
    Calculate the area of a polygon using pyclipper.
    The polygon is represented as a list of (x, y) tuples.
    The area is returned as a float.
    """
    if len(poly) < 3:
        return 0.0  # Not a polygon
    area = pyclipper.Area(to_clipper(poly))
    return area / _SCALE**2  # Convert to float


def signed_area(poly):
    """Shoelace – positive ⇒ clockwise, negative ⇒ counter-clockwise."""
    return 0.5 * sum(x0*y1 - x1*y0 for (x0, y0), (x1, y1)
                     in zip(poly, poly[1:]+[poly[0]]))

def px_to_cm(self, dx_px: float, dy_px: float) -> tuple[float, float]:
    scale = self.effective_scale
    return dx_px / scale, dy_px / scale

def cm_to_px(self, dx_cm: float, dy_cm: float) -> tuple[float, float]:
    scale = self.effective_scale
    return dx_cm * scale, dy_cm * scale

# def inner_fit_rectangle(container: Container, piece: Piece):
#     """
#     IFR (CW) in container coordinates when the *piece* anchor is its
#     top-left corner and y grows **down**.
#     """
#     Wc, Hc = container.width, container.height          # container
#     Wp = max(x for x, _ in piece.get_outer_path())              # width
#     Hp = max(y for _, y in piece.get_outer_path())            # height

#     if Wp > Wc or Hp > Hc:               # piece larger than container
#         return []

#     # TL ➜ TR ➜ BR ➜ BL  (CW with y-down)
#     return [
#         (0.0,        0.0),               # top-left  (same as anchor)
#         (Wc - Wp,    0.0),               # top-right
#         (Wc - Wp, Hc - Hp),              # bottom-right
#         (0.0,     Hc - Hp),              # bottom-left
#     ]

def shift_coordinates(outline: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    xs = [pt[0] for pt in outline]
    ys = [pt[1] for pt in outline]
    min_x = float(min(xs))
    min_y = float(min(ys))
        
    shifted_outline = [(x - min_x, y - min_y) for x, y in outline]
    return shifted_outline

# def fits(self, poly, dx, dy):
#         """
#         Check if the polygon *poly* fits in the container at (dx, dy).
#         The polygon is translated by (dx, dy) and checked against the
#         container boundaries and all previously placed pieces.
#         """
#         xs, ys = zip(*_translate_polygon(poly, dx, dy))
#         if (min(xs) < 0 or max(xs) > self.container.width or
#             min(ys) < 0 or max(ys) > self.container.height):
#             return False

#         for other, ox, oy in self.placed:
#             if polygons_overlap(
#                     _translate_polygon(poly, dx, dy),
#                     _translate_polygon(other.vertices, ox, oy)):
#                 return False
#         return True

def scale(vertices:List[Tuple[float, float]], factor: float) -> List[Tuple[float, float]]:
    """Scale the piece *in place* by *factor*."""
    import copy
    vertices = copy.deepcopy(vertices)
    for i, (x, y) in enumerate(vertices):
        vertices[i] = (x * factor, y * factor)

    return vertices

def concave_hull(points, alpha2):
    pts    = [Point_2(x, y) for x, y in points]
    ashape = Alpha_shape_2(pts, alpha2, REGULARIZED)

    # boundary edges are classified REGULAR; SINGULAR can also lie on the boundary
    boundary_classes = (REGULAR, SINGULAR)

    edges = [e for e in ashape.alpha_shape_edges()
             if ashape.classify(e) in boundary_classes]

    if not edges:
        raise ValueError(
            f"α‑shape produced no boundary edges for α²={alpha2:.4g} – "
            "choose a larger radius or verify the point cloud."
        )

    # CGAL returns edges as pairs of vertex handles
    # walk the edges to get the boundary polygon
    from collections import defaultdict, deque

    adj = defaultdict(list)
    for e in edges:
        v1, v2 = e[0], e[1]
        adj[v1].append(v2)
        adj[v2].append(v1)

    # pick an arbitrary start and trace the boundary
    start = edges[0][0]
    hull = deque([start])
    prev = None
    while True:
        cur = hull[-1]
        nxt_candidates = [v for v in adj[cur] if v is not prev]
        if not nxt_candidates:          # closed loop?
            break
        nxt = nxt_candidates[0]         # deterministic but not necessarily CCW
        if nxt == start:
            break
        hull.append(nxt)
        prev = cur

    # make sure the polygon is closed and CCW
    hull.append(start)
    if signed_area(hull) < 0:  # currently CCW → reverse
        hull.reverse()

    return [(p.x(), p.y()) for p in hull]

def compute_offset_path(contour: list[tuple[float, float]],
                        allowance: float = 1.0,
                        join_type = None,
                        miter_limit: float = 2.0) -> list[tuple[float, float]]:
    """
    Compute and return the offset path for a given contour using Pyclipper.
    All pyclipper-specific constants are kept here.
    """

    # Set default join type if not provided
    if join_type is None:
        join_type = pyclipper.JT_MITER

    subj = to_clipper(contour)
    if allowance == 0:
        return from_clipper(subj)

    pco = pyclipper.PyclipperOffset(miter_limit=miter_limit)
    pco.AddPath(subj, join_type, pyclipper.ET_CLOSEDPOLYGON)
    delta = int(round(allowance * _SCALE))
    solution = pco.Execute(delta)

    offset_paths = []
    for path in solution:
        outline = from_clipper(path)
        xs = [pt[0] for pt in outline]
        ys = [pt[1] for pt in outline]
        min_x = min(xs)
        min_y = min(ys)
        shifted_outline = [(x - min_x, y - min_y) for x, y in outline]
        offset_paths += shifted_outline

    return offset_paths