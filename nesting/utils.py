from __future__ import annotations
from typing import List, Tuple, Iterable
import math
import pyclipper
# from .layout import Layout, Container, Piece
# from .layout import Container, Piece
# from CGAL.CGAL_Polygon_2 import Polygon_2
# from CGAL.CGAL_Alpha_shape_2 import (Alpha_shape_2, Alpha_shape_2_Edge,
#                                      REGULARIZED, GENERAL)
# from CGAL.CGAL_Alpha_shape_2 import Alpha_shape_2, REGULAR, SINGULAR 
# from CGAL import CGAL_Kernel

# Point_2 = CGAL_Kernel.Point_2

from shapely.geometry import MultiPoint
from shapely.ops import unary_union


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


def polygons_overlap(poly_a, poly_b, area_tol = 1e-4) -> bool:
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

def sample_polygon_edges(polygon: List[Tuple[float, float]], n_per_edge: int) -> List[Tuple[float, float]]:
    """
    Sample interior points along each edge of a polygon, distributing
    n_per_edge points proportionally to edge length across all edges.
    Vertices themselves are excluded (already included as NFP candidates).
    """
    if n_per_edge <= 0 or len(polygon) < 2:
        return []

    # Compute edge lengths (polygon is closed: last == first, so skip last segment)
    coords = polygon[:-1] if (
        len(polygon) > 1
        and abs(polygon[0][0] - polygon[-1][0]) < 1e-9
        and abs(polygon[0][1] - polygon[-1][1]) < 1e-9
    ) else polygon

    edges = []
    n = len(coords)
    for i in range(n):
        x0, y0 = coords[i]
        x1, y1 = coords[(i + 1) % n]
        length = math.hypot(x1 - x0, y1 - y0)
        if length > 0:
            edges.append(((x0, y0), (x1, y1), length))

    total_length = sum(e[2] for e in edges)
    if total_length == 0:
        return []

    sampled: List[Tuple[float, float]] = []
    for (x0, y0), (x1, y1), length in edges:
        k = max(1, round(length / total_length * n_per_edge * len(edges)))
        for j in range(1, k + 1):
            t = j / (k + 1)
            sampled.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))

    return sampled


def no_fit_polygon(stationary, moving, n_edge_samples: int = 0):
    """
    No-Fit Polygon of *moving* about *stationary*.
    Coordinates returned as floats in the original units.
    Uses the top-leftmost vertex of the moving polygon as reference point.
    (In this coordinate system, top means minimum y since y grows downward)

    n_edge_samples: if > 0, also sample interior points along each NFP edge,
                    distributed proportionally to edge length.
    """

    # Find the top-leftmost vertex to use as reference
    # topleft_idx = find_topleft_vertex(moving)
    # ref_x, ref_y = moving[topleft_idx]

    # # Translate moving polygon so that reference vertex is at origin
    # moving_centered = [(x - ref_x, y - ref_y) for x, y in moving]
    moving_centered =[(x, y) for x, y in moving]  # no translation, use origin as reference
    A = to_clipper(stationary)
    B = [(-x, -y) for x, y in moving_centered]  # reflect the centered moving part
    B = to_clipper(B)
    nfp = pyclipper.MinkowskiSum(B, A, True)

    vertices = flatten(nfp)

    if n_edge_samples <= 0:
        return vertices

    edge_pts: List[Tuple[float, float]] = []
    for polygon_clipper in nfp:
        polygon_float = from_clipper(polygon_clipper)
        edge_pts.extend(sample_polygon_edges(polygon_float, n_edge_samples))

    return vertices + edge_pts

def find_topleft_vertex(polygon):
    """
    Find the index of the top-leftmost vertex in a polygon.
    Since y-axis grows downward, the top vertex has the minimum y coordinate.
    """
    if not polygon:
        return 0
    
    min_y = float('inf')
    min_x = float('inf')
    min_idx = 0
    
    for i, (x, y) in enumerate(polygon):
        # First prioritize topmost (minimum y in this coordinate system)
        if y < min_y:
            min_y = y
            min_x = x
            min_idx = i
        # If y-coordinates are equal, choose the leftmost
        elif y == min_y and x < min_x:
            min_x = x
            min_idx = i
    
    return min_idx

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
    The area is returned as a float (always non-negative).
    """
    if len(poly) < 3:
        return 0.0  # Not a polygon
    signed = pyclipper.Area(to_clipper(poly))
    return abs(signed) / _SCALE**2



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


def clean_polygon_coordinates(coordinates: List[Tuple[float, float]], close=True) -> List[Tuple[float, float]]:
    """
    Clean polygon coordinates by removing duplicate consecutive points and ensuring closure.
    
    Args:
        coordinates: List of (x, y) coordinates defining the polygon
        
    Returns:
        Cleaned list of coordinates with duplicates removed and polygon closed
    """
    from garmentcode.utils import close_enough

    tol = 1e-9
    cleaned_coords: List[Tuple[float, float]] = []
    for i, coord in enumerate(coordinates):
        if i == 0:
            cleaned_coords.append(coord)
            continue

        prev = coordinates[i - 1]
        if not (close_enough(coord[0] - prev[0], tol=tol) and close_enough(coord[1] - prev[1], tol=tol)):
            cleaned_coords.append(coord)
    
    # Ensure polygon is closed (first == last) using tolerance
    if len(cleaned_coords) > 0 and close:
        first = cleaned_coords[0]
        last = cleaned_coords[-1]
        if not (close_enough(first[0] - last[0], tol=tol) and close_enough(first[1] - last[1], tol=tol)):
            cleaned_coords.append(first)
        
    return cleaned_coords


def polygon_split(
    coordinates: List[Tuple[float, float]],
    object_name: str,
    use_centroid: bool = True,
    proportion: float = 0.5,
    epsilon: float = 1e-7
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Split a polygon into two parts by finding the optimal vertical split line
    that minimizes area difference between the resulting parts.
    
    This is a utility function shared between Piece.split() and Panel.split()
    to avoid code duplication.
    
    Args:
        coordinates: List of (x, y) coordinates defining the polygon
        object_name: Name of the object being split (for error messages)
        use_centroid: If True, use polygon centroid x-coordinate; if False, use proportion of bounding box
        proportion: Position of the split line as proportion of bounding box width (ignored if use_centroid=True)
        epsilon: Geometric tolerance for split line extension
        
    Returns:
        Tuple of (left_polygon_coords, right_polygon_coords) where each is a list of (x, y) tuples
        
    Raises:
        ValueError: If the polygon is invalid or split fails
        RuntimeError: If no valid split is found
    """
    from shapely.geometry import Polygon, LineString, MultiPoint
    from shapely.ops import split as shapely_split
    from shapely.errors import GEOSException
    
    # Clean the coordinates
    cleaned_coords = clean_polygon_coordinates(coordinates)
    
    poly = Polygon(cleaned_coords)
    
    # If invalid, try to fix with buffer operation
    if not poly.is_valid:
        try:
            poly = poly.buffer(0)
            if not poly.is_valid:
                raise ValueError(f"{object_name} produces an invalid polygon for split")
        except Exception:
            raise ValueError(f"{object_name} produces an invalid polygon for split")
    
    if poly.is_empty:
        raise ValueError(f"{object_name}: empty polygon given")

    # Determine split line position
    minx, miny, maxx, maxy = poly.bounds
    x0 = poly.centroid.x if use_centroid else minx + proportion * (maxx - minx)
    
    # Create vertical line that crosses the polygon
    vertical = LineString([(x0, miny - 1.0), (x0, maxy + 1.0)])

    # Find boundary intersections with the vertical line
    hits = poly.boundary.intersection(vertical)
    
    if isinstance(hits, MultiPoint):
        pts = sorted(hits.geoms, key=lambda p: p.y)
    else:
        pts = [hits] if hits else []
        
    if len(pts) < 2 or len(pts) % 2:
        raise ValueError(f"{object_name}: expected even # of intersections, got {len(pts)}")

    # Test every inside segment to find the one with minimal area difference
    best_split = None
    best_delta = float("inf")

    for i in range(0, len(pts), 2):
        p, q = pts[i], pts[i + 1]
        
        # Create a slightly extended cut line to ensure clean splitting
        dy = q.y - p.y
        p0 = (p.x, p.y - epsilon * max(1.0, abs(dy)))
        q0 = (q.x, q.y + epsilon * max(1.0, abs(dy)))
        knife = LineString([p0, q0])

        try:
            parts_gc = shapely_split(poly, knife)
            parts = [g for g in parts_gc.geoms if g.geom_type == "Polygon"]
            
            if len(parts) >= 2:
                area_left, area_right = parts[0].area, parts[1].area
                delta = abs(area_left - area_right)
                if delta < best_delta:
                    best_delta = delta
                    best_split = (parts[0], parts[1])
        except GEOSException:
            continue

    if best_split is None:
        raise RuntimeError(f"{object_name}: no valid split produced")

    poly_left, poly_right = best_split
    
    # Validate that the total area is conserved (within numerical tolerance)
    original_area = poly.area
    split_total_area = poly_left.area + poly_right.area
    area_tolerance = max(1e-6, original_area * 1e-9)  # Adaptive tolerance based on polygon size
    area_diff = abs(original_area - split_total_area)
    
    if area_diff > area_tolerance:
        raise ValueError(
            f"{object_name}: area conservation violated after split. "
            f"Original area: {original_area:.9f}, "
            f"Split total area: {split_total_area:.9f}, "
            f"Difference: {area_diff:.9f}, "
            f"Tolerance: {area_tolerance:.9f}"
        )
    
    # Convert back to coordinate lists (excluding the closing coordinate)
    left_coords = list(poly_left.exterior.coords)[:-1]
    right_coords = list(poly_right.exterior.coords)[:-1]
    
    return left_coords, right_coords