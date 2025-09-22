from collections import OrderedDict
from typing import Literal, Optional, List, Tuple, Union
from enum import Enum

import random
import numpy as np
from scipy.spatial import Delaunay

from shapely.geometry import Point, Polygon, MultiPoint, MultiLineString, MultiPolygon
from shapely.ops import unary_union, polygonize

from .layout import Layout, Container, Piece
import nesting.config as config
import nesting.utils as utils

import math
import random
from typing import Iterable, List, Optional, Tuple

from shapely import affinity
from shapely.geometry import Polygon

from .layout import Layout, Container, Piece


# ── SIMPLE SORTING FUNCTIONS ──────────────────────────────────────────────────

def sort_pieces_by_bbox_area(pieces: List[Piece], reverse: bool = True) -> List[Piece]:
    """Sort pieces by bounding box area"""
    return sorted(pieces, key=lambda p: p.bbox_area, reverse=reverse)

def sort_pieces_by_area(pieces: List[Piece], reverse: bool = True) -> List[Piece]:
    """Sort pieces by actual polygon area"""
    return sorted(pieces, key=lambda p: utils.polygon_area(p.get_outer_path()), reverse=reverse)

def sort_pieces_by_length(pieces: List[Piece], reverse: bool = True) -> List[Piece]:
    """Sort pieces by width"""
    return sorted(pieces, key=lambda p: p.width, reverse=reverse)

def get_piece_order_by_criteria(layout: Layout, criteria: str, reverse: bool = True) -> List[str]:
    """Get piece IDs sorted by criteria"""
    pieces = list(layout.order.values())
    
    if criteria == 'bbox_area':
        sorted_pieces = sort_pieces_by_bbox_area(pieces, reverse)
    elif criteria == 'area':
        sorted_pieces = sort_pieces_by_area(pieces, reverse)
    elif criteria == 'length':
        sorted_pieces = sort_pieces_by_length(pieces, reverse)
    else:
        raise ValueError(f"Unknown criteria: {criteria}")
    
    return [piece.id for piece in sorted_pieces]


class PlacementMode(Enum):
    """Placement mode for selecting the best position from NFP candidates."""
    BOTTOM_LEFT = "bottom_left"          # Traditional bottom-left fill
    MAX_OVERLAP = "max_overlap"          # Maximize overlap with existing pieces
    MIN_BBOX_LENGTH = "min_bbox_length"  # Minimize bounding box length
    MIN_BBOX_AREA = "min_bbox_area"      # Minimize bounding box area


DECODER_REGISTRY: dict[str, type] = {}

def register_decoder(name: str):
    def deco(cls):
        DECODER_REGISTRY[name] = cls
        return cls
    return deco


class PlacementEngine:
    """
    Base class for all placement strategies.
    Provides:
      - _fits(...) container‐bounds/overlap testing
      - anchor/gravitate for BL logic
      - unified alpha_shape(...) to compute concave hull
      - _update_exterior_contour(...) that reuses alpha_shape
      - usage_BB(), rest_length(), layout_is_valid() for metrics
    """

    def __init__(self, layout: Layout, container: Container, gravitate_once: bool = False, **kwargs):
        self.layout = layout
        self.container = container
        self.placed: List[Piece] = []
        self.gravitate_once = gravitate_once
        # Will hold either a Polygon or a MultiPolygon representing
        # the “exterior contour” of all placed pieces
        self._exterior_contour: Optional[Union[Polygon, MultiPolygon]] = None

        # Last computed concave hull (for utilization, via alpha_shape)
        self._last_hull: Optional[Polygon] = None

    @property
    def last_hull(self) -> Optional[Polygon]:
        """Return the last computed concave hull, or None if not computed yet."""
        return self._last_hull

    # ── EXISTING UTILITY METHODS (unchanged) ──────────────────────────────────────

    def _fits(self, piece: Piece, dx: float, dy: float) -> bool:
        """Return True if piece at ``(dx, dy)`` lies inside container and does not overlap."""

        # ---- bounding box check against container ----
        cand_min_x = piece.min_x + dx
        cand_max_x = piece.max_x + dx
        cand_min_y = piece.min_y + dy
        cand_max_y = piece.max_y + dy

        if (
            cand_min_x < 0 or cand_max_x > self.container.width
            or cand_min_y < 0 or cand_max_y > self.container.height
        ):
            return False

        poly = utils._translate_polygon(piece.get_outer_path(), dx, dy)

        # ---- overlap checks ----
        for other in self.placed:
            ox, oy = other.translation

            other_min_x = other.min_x + ox
            other_max_x = other.max_x + ox
            other_min_y = other.min_y + oy
            other_max_y = other.max_y + oy

            if (
                cand_max_x <= other_min_x or cand_min_x >= other_max_x
                or cand_max_y <= other_min_y or cand_min_y >= other_max_y
            ):
                # Bounding boxes do not intersect → cannot overlap.
                continue

            other_poly = utils._translate_polygon(other.get_outer_path(), ox, oy)
            if utils.polygons_overlap(poly, other_poly):
                return False

        return True

    def anchor(self, piece: Piece) -> Tuple[float, float]:
        """Push piece against the container’s top‐right corner."""
        vertices = piece.get_outer_path()
        xs = [v[0] for v in vertices]
        min_x, max_x = min(xs), max(xs)
        start_x = self.container.width - (max_x - min_x)
        start_y = 0.0
        return start_x, start_y

    def gravitate(self, piece: Piece, x: float, y: float, step: float = config.GRAVITATE_STEP) -> Tuple[float, float]:
        """Slide piece left until jammed, then down, repeating until no movement."""
        moved = True
        while moved:
            moved = False
            while x - step >= 0 and self._fits(piece, x - step, y):
                x -= step
                moved = True
            while y + step <= self.container.height and self._fits(piece, x, y + step):
                y += step
                moved = True
        return x, y

    def BLCompact(self, compact_order: List[str]): # -> List[Tuple[float, float]]:
        """
        Given the ids of pieces in `compact_order`, shift the pieces in self.placed
        """
        for piece_id in compact_order:
            piece = next((p for p in self.placed if p.id == piece_id), None)
            if piece is None:
                raise ValueError(f"BLCompact: piece id '{piece_id}' not found in placed pieces")
            x0, y0 = piece.translation
            x_new, y_new = self.gravitate(piece, x0, y0)
            piece.translation = (x_new, y_new)

    def decode_in_order(self, piece_order: List[str]) -> list[Tuple[str, float, float, float]]:
        """
        Interface method - subclasses must implement this with their specific placement logic
        """
        raise NotImplementedError("Subclasses must implement decode_in_order")
            
    

    def _gravitate_once(self, piece: Piece, x: float, y: float, step: float = config.GRAVITATE_STEP) -> Tuple[float, float]:
        """
        “Traditional” fully-left then fully-down:
        1. From (x,y), move left in discrete steps until you can't go further.
        2. From that leftmost location, move down in discrete steps until you can’t go further.
        No iteration back to left after moving down.
        """
        # 1) Slide left as far as possible
        while x - step >= 0 and self._fits(piece, x - step, y):
            x -= step

        # 2) Slide down as far as possible
        while y + step <= self.container.height and self._fits(piece, x, y + step):
            y += step

        return x, y


    def _flatten_piece_list(self) -> List[Tuple[float, float]]:
        return [
            (x + p.translation[0], y + p.translation[1])
            for p in self.placed
            for x, y in p.get_outer_path()
        ]

    def layout_is_valid(self) -> bool:
        """
        Check if all pieces have been placed and the layout is valid.
        A valid layout means:
        1. All pieces from the original layout have been placed
        2. All placed pieces are within container bounds
        3. No pieces overlap
        """
        # Check if all pieces from the original layout have been placed
        if len(self.placed) != len(self.layout.order):
            return False
            
        # Check if all pieces fit within container and don't overlap
        n = len(self.placed)
        for i in range(n):
            pi = self.placed[i]
            poly_i = utils._translate_polygon(pi.get_outer_path(), *pi.translation)
            xs_i, ys_i = zip(*poly_i)
            if (
                min(xs_i) < 0 or max(xs_i) > self.container.width
                or min(ys_i) < 0 or max(ys_i) > self.container.height
            ):
                return False
            for j in range(i + 1, n):
                pj = self.placed[j]
                poly_j = utils._translate_polygon(pj.get_outer_path(), *pj.translation)
                if utils.polygons_overlap(poly_i, poly_j):
                    return False
        return True

    def bbox_area(self) -> float:
        """
        Calculate the utilization ratio based on the bounding box.
        Returns 0.0 if not all pieces have been placed or if the layout is invalid.
        """
        # First check if the layout is valid (includes check that all pieces are placed)
        if not self.layout_is_valid():
            return 0.0
            
        flattened = self._flatten_piece_list()
        if not flattened:
            return 0.0
        xs = [v[0] for v in flattened]
        ys = [v[1] for v in flattened]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        return (max_x - min_x) * (max_y - min_y)

    def usage_BB(self) -> float:
        """
        Calculate the utilization ratio based on the bounding box.
        Returns 0.0 if not all pieces have been placed or if the layout is invalid.
        """
        bbox_area = self.bbox_area()
        total_area = sum(utils.polygon_area(p.get_outer_path()) for p in self.placed)
        ratio = total_area / bbox_area if bbox_area > 0 else 0.0
        return ratio

    def rest_length(self) -> float:
        flattened = self._flatten_piece_list()
        if not flattened:
            return self.container.width
        xs = [v[0] for v in flattened]
        return self.container.width - max(xs)
    
    def rest_height(self) -> float:
        flattened = self._flatten_piece_list()
        if not flattened:
            return self.container.height
        ys = [v[1] for v in flattened]
        return min(ys)

    # ──  α-SHAPE (CONCAVE HULL) HELPER ────────────────────────────────────

    def alpha_shape(self,
                    points: List[Tuple[float, float]],
                    *,
                    trim_ratio: float = 7.0,
                    interior_spacing: float = 5.0,
                    boundary_spacing: float = 1.0
                   ) -> Polygon:
        """
        Compute a concave hull of `points` via alpha-shape
        Stores result in self._last_hull.
        """
        interior_spacing = float(interior_spacing)
        boundary_spacing = float(boundary_spacing)

        def sample_interior(poly: Polygon) -> List[Tuple[float, float]]:
            minx, miny, maxx, maxy = poly.bounds
            xs = np.arange(minx + interior_spacing/2.0, maxx, interior_spacing)
            ys = np.arange(miny + interior_spacing/2.0, maxy, interior_spacing)
            pts = []
            for x in xs:
                for y in ys:
                    if poly.contains(Point(x, y)):
                        pts.append((float(x), float(y)))
            return pts

        def sample_boundary(poly: Polygon) -> List[Tuple[float, float]]:
            coords = list(poly.exterior.coords)
            pts = []
            for (x0, y0), (x1, y1) in zip(coords, coords[1:]):
                edge_len = float(np.hypot(x1 - x0, y1 - y0))
                n_samples = max(int(edge_len / boundary_spacing), 1)
                for t in np.linspace(0.0, 1.0, n_samples, endpoint=False):
                    pts.append((float(x0 + t * (x1 - x0)),
                                float(y0 + t * (y1 - y0))))
            return pts

        # Build sampled point cloud over all placed pieces
        all_pts: List[Tuple[float, float]] = []
        centroids = []
        for piece in self.placed:
            raw = piece.get_outer_path()
            # Clean coordinates and translate
            translated_coords = [(x + piece.translation[0], y + piece.translation[1]) for x, y in raw]
            cleaned_coords = utils.clean_polygon_coordinates(translated_coords)
            poly = Polygon(cleaned_coords)
            
            # Only proceed if polygon is valid
            if poly.is_valid and not poly.is_empty:
                all_pts.extend(sample_boundary(poly))
                all_pts.extend(sample_interior(poly))
                centroids.append(poly.centroid.coords[0])

        pts = np.array(all_pts, dtype=float)
        if pts.shape[0] < 4:
            hull = MultiPoint(pts).convex_hull
            self._last_hull = hull
            return hull

        tri = Delaunay(pts)

        # Edge length computation for all simplices
        simplices = tri.simplices
        tri_pts = pts[simplices]  # (n, 3, 2)
        edges = np.stack([
            tri_pts[:, 0] - tri_pts[:, 1],
            tri_pts[:, 1] - tri_pts[:, 2],
            tri_pts[:, 2] - tri_pts[:, 0],
        ], axis=1)
        edge_lengths = np.linalg.norm(edges, axis=2)

        median_len = float(np.median(edge_lengths))
        max_edge = median_len * trim_ratio

        mask = np.all(edge_lengths <= max_edge, axis=1)
        kept_simplices = simplices[mask]

        if kept_simplices.size == 0:
            hull = MultiPoint(pts).convex_hull
            self._last_hull = hull
            return hull

        # Collect unique edges from kept simplices
        edge_pairs = np.stack([
            kept_simplices[:, [0, 1]],
            kept_simplices[:, [1, 2]],
            kept_simplices[:, [2, 0]],
        ], axis=1).reshape(-1, 2)
        edge_pairs = np.sort(edge_pairs, axis=1)
        kept = {tuple(e) for e in np.unique(edge_pairs, axis=0)}
        if not kept:
            hull = MultiPoint(pts).convex_hull
            self._last_hull = hull
            return hull

        mls = MultiLineString([ (tuple(pts[i]), tuple(pts[j])) for i, j in kept ])
        regions = list(polygonize(unary_union(mls)))
        if not regions:
            hull = MultiPoint(pts).convex_hull
            self._last_hull = hull
            return hull

        merged = unary_union(regions)
        if hasattr(merged, "geoms"):
            hull = max(merged.geoms, key=lambda p: p.area)
        else:
            hull = merged

        missing = [c for c in centroids if not hull.contains(Point(c))]
        if missing:
            full_hull = MultiPoint(pts).convex_hull
            hull = unary_union([hull, full_hull])

        if config.SNAP:
            w, h = self.container.width, self.container.height
            tol_x = config.SNAP_TOLERANCE * w
            tol_y = config.SNAP_TOLERANCE * h
            coords = list(hull.exterior.coords)
            snapped: List[Tuple[float, float]] = []
            n = len(coords)

            def is_on_horiz(pt: Tuple[float, float]) -> bool:
                return abs(pt[1]) < tol_y or abs(pt[1] - h) < tol_y

            def is_on_vert(pt: Tuple[float, float]) -> bool:
                return abs(pt[0]) < tol_x or abs(pt[0] - w) < tol_x

            for i, (x, y) in enumerate(coords):
                prev = coords[i - 1]
                nxt = coords[(i + 1) % n]

                if abs(x) < tol_x or (abs(prev[0]) < tol_x and abs(nxt[0]) < tol_x):
                    x = 0.0
                elif abs(x - w) < tol_x or (abs(prev[0] - w) < tol_x and abs(nxt[0] - w) < tol_x):
                    x = w
                # snap if adjacent to vertical boundary segment
                elif is_on_vert(prev) and is_on_vert(nxt):
                    x = 0.0 if abs(x) < abs(x - w) else w

                if abs(y) < tol_y or (abs(prev[1]) < tol_y and abs(nxt[1]) < tol_y):
                    y = 0.0
                elif abs(y - h) < tol_y or (abs(prev[1] - h) < tol_y and abs(nxt[1] - h) < tol_y):
                    y = h
                # snap if adjacent to horizontal boundary segment
                elif is_on_horiz(prev) and is_on_horiz(nxt):
                    y = 0.0 if abs(y) < abs(y - h) else h

                curr = (x, y)
                snapped.append(curr)
                x2, y2 = coords[(i + 1) % n]
                if abs(x2) < tol_x:
                    x2 = 0.0
                elif abs(x2 - w) < tol_x:
                    x2 = w
                if abs(y2) < tol_y:
                    y2 = 0.0
                elif abs(y2 - h) < tol_y:
                    y2 = h
                nxt_snapped = (x2, y2)

                if (
                    (is_on_horiz(curr) and is_on_vert(nxt_snapped)) or
                    (is_on_vert(curr) and is_on_horiz(nxt_snapped))
                ):
                    corner = (
                        curr[0] if is_on_vert(curr) else nxt_snapped[0],
                        curr[1] if is_on_horiz(curr) else nxt_snapped[1]
                    )
                    if corner not in snapped:
                        snapped.append(corner)

            # Clean the snapped coordinates before creating the polygon
            cleaned_snapped = utils.clean_polygon_coordinates(snapped)
            hull = Polygon(cleaned_snapped)

        # Check if the resulting hull is valid before returning it
        if not hull.is_valid:
            print("[alpha_shape] Generated an invalid geometry - this chromosome will be discarded")
            # Create an empty polygon as fallback
            empty_hull = Polygon()
            self._last_hull = empty_hull
            return empty_hull
    
        self._last_hull = hull
        return hull

    def _update_exterior_contour(self, new_piece: Piece) -> None:
        """
        Merge `new_piece` into the running exterior contour.  
        If config.PRESERVE_HOLES is True, keep the full union (Polygon or MultiPolygon).
        Otherwise, build one concave α‐shape over all exterior vertices of the union.
        """
        raw = new_piece.get_outer_path()
        x0, y0 = new_piece.translation
        shifted = [(x + x0, y + y0) for x, y in raw]
        
        # Clean coordinates before creating polygon
        cleaned_shifted = utils.clean_polygon_coordinates(shifted)
        piece_poly = Polygon(cleaned_shifted)

        if self._exterior_contour is None:
            self._exterior_contour = piece_poly
            return

        merged = self._exterior_contour.union(piece_poly)
        self._exterior_contour = merged
        return

    def concave_hull_utilization(self) -> float:
        """
        Calculate the utilization ratio based on the concave hull.
        Returns 0.0 if not all pieces have been placed or if the layout is invalid.
        """
        # First check if the layout is valid (includes check that all pieces are placed)
        if not self.layout_is_valid():
            return 0.0
            
        if not self.placed:
            return 0.0

        pts = self._flatten_piece_list()
        hull = self.alpha_shape(pts,
                                trim_ratio=config.HULL_TRIM_RATIO,
                                interior_spacing=config.INTERIOR_SAMPLE_SPACING,
                                boundary_spacing=config.BOUNDARY_SAMPLE_SPACING)
        hull_area = hull.area if not hull.is_empty else 0.0
        total_area = sum(utils.polygon_area(p.get_outer_path()) for p in self.placed)
        return (total_area / hull_area) if hull_area > 0 else 0.0
    
    def concave_hull_area(self) -> float:
        """
        Calculate the area of the concave hull formed by all placed pieces.
        Returns 0.0 if not all pieces have been placed or if the layout is invalid.
        """
        # First check if the layout is valid (includes check that all pieces are placed)
        if not self.layout_is_valid():
            return 0.0
            
        if not self.placed:
            return 0.0

        pts = self._flatten_piece_list()
        hull = self.alpha_shape(pts,
                                trim_ratio=config.HULL_TRIM_RATIO,
                                interior_spacing=config.INTERIOR_SAMPLE_SPACING,
                                boundary_spacing=config.BOUNDARY_SAMPLE_SPACING)
        return hull.area if not hull.is_empty else 0.0

@register_decoder("BL")
class BottomLeftDecoder(PlacementEngine):
    def __init__(self, layout: Layout, container: Container, *, step: Optional[float] = None, **kwargs):
        super().__init__(layout, container)
        self.step = step if step is not None else config.GRAVITATE_STEP

    def decode(self) -> list[Tuple[str, float, float, float]]:
        """Place pieces using BL heuristic"""
        return self.decode_in_order(list(self.layout.order.keys()))

    def decode_in_order(self, piece_order: List[str]) -> list[Tuple[str, float, float, float]]:
        """Decode pieces in specified order using BL placement"""
        for piece_id in piece_order:
            if piece_id not in self.layout.order:
                continue
            piece = self.layout.order[piece_id]
            x0, y0 = self.anchor(piece)
            if self.gravitate_once:
                dx, dy = self._gravitate_once(piece, x0, y0, step=self.step)
            else:
                dx, dy = self.gravitate(piece, x0, y0, step=self.step)
            piece.translation = (dx, dy)
            self.placed.append(piece)
        return [(p.id, p.translation[0], p.translation[1], p.rotation) for p in self.placed]


@register_decoder("Random")


@register_decoder("Greedy")
class GreedyBLDecoder(BottomLeftDecoder):
    """
    A Greedy BL decoder that can sort by different metrics:
      - bbox_area       (piece.min_x→piece.max_x × piece.min_y→piece.max_y)
      - hull_area       (area of convex hull of piece’s polygon)
      - aspect_ratio    (max(width/height, height/width))
    """
    def __init__(self,
                 layout: Layout,
                 container: Container,
                 *,
                 step: Optional[float] = None,
                 traditional: bool = False,
                 sort_key: Literal["bbox_area", "hull_area", "aspect_ratio"] = "bbox_area",
                 **kwargs):
        """
        :param sort_key: One of "bbox_area", "hull_area", or "aspect_ratio".
                         Determines how pieces are ordered before placement.
        """
        self.sort_key = sort_key
        # 1) Compute the metric for each piece
        metrics = {}
        for pid, piece in layout.order.items():
            if sort_key == "bbox_area":
                # piece.bbox_area is already available
                metrics[pid] = piece.bbox_area
            elif sort_key == "hull_area":
                # Use your existing utils.polygon_area(...) on the convex hull of piece.get_outer_path()
                from shapely.geometry import Polygon
                raw_path = piece.get_outer_path()
                convex_hull_coords = Polygon(raw_path).convex_hull.exterior.coords[:]  
                # convex_hull_coords is a sequence of (x, y) tuples
                metrics[pid] = utils.polygon_area(list(convex_hull_coords))
            elif sort_key == "aspect_ratio":
                w, h = piece.width, piece.height
                # Use max(w/h, h/w) so that extremely elongated shapes rank higher
                metrics[pid] = max(w / h, h / w) if (h > 0 and w > 0) else 1.0
            else:
                raise ValueError(f"Unrecognized sort_key: {sort_key}")

        # 2) Sort piece IDs in descending order of the chosen metric
        sorted_ids = sorted(metrics.keys(), key=lambda pid: metrics[pid], reverse=True)

        # 3) Rebuild layout.order as an OrderedDict in that sorted order
        sorted_pieces = [(pid, layout.order[pid]) for pid in sorted_ids]
        new_order = OrderedDict(sorted_pieces)

        # 4) Initialize the parent BL decoder with the newly ordered layout
        super().__init__(Layout(new_order), container, step=step, traditional=traditional, **kwargs)



@register_decoder("Random")
class RandomDecoder(PlacementEngine):
    def __init__(self,
                 layout: Layout,
                 container: Container,
                 *,
                 decoder: str = "BL",  # choose 'BL' or 'NFP'
                 rotations_on: bool = config.ENABLE_ROTATIONS,
                 **kwargs):
        # Shuffle piece order
        ids = list(layout.order)
        random.shuffle(ids)
        shuffled = OrderedDict((i, layout.order[i]) for i in ids)
        super().__init__(Layout(shuffled), container, **kwargs)
        self.decoder = decoder
        self.rotations_on = rotations_on

    def decode(self):
        # Apply random rotations if enabled
        if self.rotations_on:
            for p in self.layout.order.values():
                p.rotate(random.choice(config.ALLOWED_ROTATIONS))
        # Delegate to chosen decode strategy
        if self.decoder == "BL":
            dec = BottomLeftDecoder(self.layout, self.container, step=config.GRAVITATE_STEP)
        elif self.decoder == "NFP":
            dec = NFPDecoder(self.layout, self.container)
        else:
            raise ValueError(f"RandomDecoder: unsupported decoder '{self.decoder}'")
        placements = dec.decode()
        self.placed = dec.placed
        return placements


@register_decoder("NFP")
class NFPDecoder(PlacementEngine):
    """
    NFP-based placement engine with configurable placement modes.
    Supports bottom_left, max_overlap, min_bbox_length, and min_bbox_area placement strategies.
    """

    def __init__(self, layout: Layout, container: Container, *,
                 placement_mode: Union[PlacementMode, str] = PlacementMode.MAX_OVERLAP,
                 step=None, **kwargs):
        super().__init__(layout, container)
        self._nfp_cache = {}

        # Handle string or enum input for placement mode
        if isinstance(placement_mode, str):
            try:
                self.placement_mode = PlacementMode(placement_mode)
            except ValueError:
                raise ValueError(f"Invalid placement mode: {placement_mode}. Valid modes: {[m.value for m in PlacementMode]}")
        else:
            self.placement_mode = placement_mode

    def decode(self):
        return self.decode_in_order(list(self.layout.order.keys()))

    def decode_in_order(self, piece_order: List[str]) -> list[Tuple[str, float, float, float]]:
        """Decode pieces in specified order using NFP placement"""
        for piece_id in piece_order:
            if piece_id not in self.layout.order:
                continue
            piece = self.layout.order[piece_id]
            
            best_x, best_y = self._find_best_position(piece)
            if best_x is None:
                continue

            piece.translation = (best_x, best_y)
            self.placed.append(piece)
            self._update_exterior_contour(piece)

        return [(p.id, *p.translation, p.rotation) for p in self.placed]

    def _find_best_position(self, piece: Piece, gravitate_on: bool = config.NFP_GRAVITATE_ON) -> Tuple[Optional[float], Optional[float]]:
        """
        Place `piece` by generating NFP candidates against each exterior boundary
        ring in self._exterior_contour (Polygon or MultiPolygon).
        Score by bounding‐box overlap, return the best (x,y).
        """
        if not self.placed:
            return 0.0, self.container.height - piece.height

        contour = self._exterior_contour
        if contour is None or contour.is_empty:
            ax, ay = self.anchor(piece)
            return self.gravitate(piece, ax, ay)

        # 1) Gather every exterior ring
        rings: List[List[Tuple[float, float]]] = []
        if isinstance(contour, Polygon):
            rings.append(list(contour.exterior.coords))
        else:
            for geom in getattr(contour, "geoms", []):
                if isinstance(geom, Polygon):
                    rings.append(list(geom.exterior.coords))

        # 2) Compute all NFP candidates
        moving_path = piece.get_outer_path()
        cleaned_moving = utils.clean_polygon_coordinates(moving_path)
        all_nfp_pts: List[Tuple[float, float]] = []
        for ring_coords in rings:
            cleaned_ring = utils.clean_polygon_coordinates(ring_coords)
            pts = utils.no_fit_polygon(cleaned_ring, cleaned_moving)
            all_nfp_pts.extend(pts)

        # 3) Compute bounding‐box of all placed pieces
        flattened = [
            (x + p.translation[0], y + p.translation[1])
            for p in self.placed
            for x, y in p.get_outer_path()
        ]
        xs_placed = [v[0] for v in flattened]
        ys_placed = [v[1] for v in flattened]
        min_px, max_px = min(xs_placed), max(xs_placed)
        min_py, max_py = min(ys_placed), max(ys_placed)

        # Initialize best_score based on placement mode
        if self.placement_mode == PlacementMode.MAX_OVERLAP:
            best_score = -1.0  # maximize overlap
        elif self.placement_mode == PlacementMode.BOTTOM_LEFT:
            best_score = (float('inf'), float('inf'))  # minimize (y, x)
        elif self.placement_mode in [PlacementMode.MIN_BBOX_LENGTH, PlacementMode.MIN_BBOX_AREA]:
            best_score = float('inf')  # minimize bbox length or area

        best_x = best_y = None

        # 4) Evaluate each candidate
        for dx_rel, dy_rel in all_nfp_pts:
            x_cand = dx_rel
            y_cand = dy_rel

            candidate_vertices = [(x + x_cand, y + y_cand) for x, y in moving_path]
            xs_cand = [v[0] for v in candidate_vertices]
            ys_cand = [v[1] for v in candidate_vertices]
            if (
                min(xs_cand) < 0
                or max(xs_cand) > self.container.width
                or min(ys_cand) < 0
                or max(ys_cand) > self.container.height
            ):
                continue

            if not self._fits(piece, x_cand, y_cand):
                continue

            cand_min_x = min(xs_cand)
            cand_max_x = max(xs_cand)
            cand_min_y = min(ys_cand)
            cand_max_y = max(ys_cand)

            # Calculate score based on placement mode
            if self.placement_mode == PlacementMode.MAX_OVERLAP:
                # Original max overlap logic
                overlap_w = min(max_px, cand_max_x) - max(min_px, cand_min_x)
                overlap_h = min(max_py, cand_max_y) - max(min_py, cand_min_y)
                if overlap_w <= 0 or overlap_h <= 0:
                    score = 0.0
                else:
                    score = overlap_w * overlap_h

                if score > best_score:
                    best_score = score
                    best_x = x_cand
                    best_y = y_cand

            elif self.placement_mode == PlacementMode.BOTTOM_LEFT:
                # Choose bottommost, then leftmost position
                score = (y_cand, x_cand)  # tuple comparison: y first (higher is worse), then x (higher is worse)

                if score < best_score:
                    best_score = score
                    best_x = x_cand
                    best_y = y_cand

            elif self.placement_mode == PlacementMode.MIN_BBOX_LENGTH:
                # Calculate what the bounding box length would be with this placement
                all_vertices = []
                for p in self.placed:
                    for x, y in p.get_outer_path():
                        all_vertices.append((x + p.translation[0], y + p.translation[1]))
                for x, y in piece.get_outer_path():
                    all_vertices.append((x + x_cand, y + y_cand))

                xs = [v[0] for v in all_vertices]
                bbox_length = max(xs) - min(xs)

                if bbox_length < best_score:
                    best_score = bbox_length
                    best_x = x_cand
                    best_y = y_cand

            elif self.placement_mode == PlacementMode.MIN_BBOX_AREA:
                # Calculate what the bounding box area would be with this placement
                all_vertices = []
                for p in self.placed:
                    for x, y in p.get_outer_path():
                        all_vertices.append((x + p.translation[0], y + p.translation[1]))
                for x, y in piece.get_outer_path():
                    all_vertices.append((x + x_cand, y + y_cand))

                xs = [v[0] for v in all_vertices]
                ys = [v[1] for v in all_vertices]
                bbox_area = (max(xs) - min(xs)) * (max(ys) - min(ys))

                if bbox_area < best_score:
                    best_score = bbox_area
                    best_x = x_cand
                    best_y = y_cand

        if best_x is None or best_y is None:
            ax, ay = self.anchor(piece)
            best_x, best_y = self.gravitate(piece, ax, ay)
        elif gravitate_on:
            best_x, best_y = self.gravitate(piece, best_x, best_y)

        return best_x, best_y

    def _nfp(self, stationary: Piece, moving: Piece) -> List[Tuple[float, float]]:
        """
        Return (cached) no‐fit polygon between `stationary` and `moving`.  
        Only used if you need pairwise NFP, but in the code above we compute
        NFP against the entire contour of placed pieces instead.
        """
        key = (stationary.id, moving.id)
        if key not in self._nfp_cache:
            # Clean both polygons before calculating NFP
            cleaned_stationary = utils.clean_polygon_coordinates(stationary.get_outer_path())
            cleaned_moving = utils.clean_polygon_coordinates(moving.get_outer_path())
            
            self._nfp_cache[key] = utils.no_fit_polygon(
                cleaned_stationary,
                cleaned_moving
            )
        return self._nfp_cache[key]
    

@register_decoder("TOPOS")
class TOPOSDecoder(PlacementEngine):
    """
    Based on the TOPOS paper
    Parameters
    ----------
    control : {"local_search","initial_sorting"}
        Search control (Sect. 2.5): evaluate all remaining piece/orientation options
        each iteration ("local_search"), or sort pieces first and at each iteration
        evaluate only orientations of the next piece ("initial_sorting").
    nesting_strategy : {"min_area","min_length","max_overlap"}
        Placement-point selection rule (Sect. 2.3): choose candidate (dx,dy) that
        minimizes area or length of the rectangular enclosure with the partial
        solution, or maximizes the overlap between the two rectangular enclosures.
    eval_terms : tuple subset of {"waste","overlap","distance"}
        Evaluation criteria for comparing alternative partial solutions (Sect. 2.4).
        Combined as: score = WASTE_rel - OVERLAP_abs + DISTANCE_rel over the
        selected terms (defaults to all three with unit weights).
    rotations_on : bool
        If True, test all orientations in config.ALLOWED_ROTATIONS (e.g., {0,180}).
    sorting_key : {"length","area","concavity","rectangularity","total_area", None}
        Initial sorting criterion for "initial_sorting" (Fig. 6). Ignored for
        "local_search".
    edge_samples : int
        # samples per NFP edge for "extended search" along edges (Sect. 2.3).
        Vertices are always included; edges add (edge_samples-1) interior samples.
    """

    def __init__(
        self,
        layout: Layout,
        container: Container,
        *,
        control: str = "local_search",
        nesting_strategy: Literal["min_area","min_length","max_overlap"] = "min_area",
        eval_terms: Tuple[str, ...] = ("waste","overlap","distance"),
        rotations_on: bool = True,
        sorting_key: Optional[Literal["length","area","concavity","rectangularity","total_area"]] = None,
        edge_samples: int = 16,
        **kwargs
    ):
        super().__init__(layout, container, **kwargs)
        self.control = control
        self.nesting_strategy = nesting_strategy
        self.eval_terms = tuple(eval_terms)
        self.rotations_on = rotations_on
        self.sorting_key = sorting_key
        self.edge_samples = max(2, int(edge_samples))

    # ---------- Public API ----------

    def decode(self) -> list[Tuple[str, float, float, float]]:
        if self.control not in ("local_search","initial_sorting"):
            raise ValueError(f"TOPOSDecoder: unsupported control '{self.control}'")

        # Optionally reorder upfront (initial sorting variant, Sect. 2.5)
        if self.control == "initial_sorting" and self.sorting_key:
            self._apply_initial_sorting(self.sorting_key)

        remaining: OrderedDict[str, Piece] = OrderedDict(self.layout.order)

        # Iteratively build the partial solution (Fig. 7)
        while remaining:
            if self.control == "local_search":
                # Evaluate *all* piece/orientation options from S (Sect. 2.5)
                best = None
                for pid, piece in list(remaining.items()):
                    for angle in self._iter_orientations():
                        cand = self._place_and_score(piece, angle)
                        if cand is None:
                            continue
                        score, dx, dy, chosen_angle = cand
                        if (best is None) or (score < best[0]):  # minimize
                            best = (score, pid, piece, dx, dy, chosen_angle)
                # if best is None:
                #     # Fallback: nothing feasible — try to BL-place next arbitrary piece
                #     pid, piece = next(iter(remaining.items()))
                #     dx, dy = self.anchor(piece)
                #     dx, dy = self.gravitate(piece, dx, dy)
                #     piece.translation = (dx, dy)
                #     self.placed.append(piece)
                #     self._update_exterior_contour(piece)
                #     del remaining[pid]
                #     continue

                _, pid, piece, dx, dy, angle = best
                self._commit(piece, pid, dx, dy, angle)
                del remaining[pid]

            else:  # initial_sorting
                pid, piece = next(iter(remaining.items()))
                best = None
                for angle in self._iter_orientations():
                    cand = self._place_and_score(piece, angle)
                    if cand is None:
                        continue
                    score, dx, dy, chosen_angle = cand
                    if (best is None) or (score < best[0]):  # minimize
                        best = (score, dx, dy, chosen_angle)
                if best is None:
                    # Fallback: BL-place this piece
                    dx, dy = self.anchor(piece)
                    dx, dy = self.gravitate(piece, dx, dy)
                    piece.translation = (dx, dy)
                    self.placed.append(piece)
                    self._update_exterior_contour(piece)
                    del remaining[pid]
                    continue

                score, dx, dy, angle = best
                self._commit(piece, pid, dx, dy, angle)
                del remaining[pid]

        return [(p.id, p.translation[0], p.translation[1], p.rotation) for p in self.placed]

    # ---------- Core helpers ----------

    def _commit(self, piece: "Piece", pid: str, dx: float, dy: float, angle: float) -> None:
        orig = piece.rotation
        try:
            if self.rotations_on:
                piece.rotate(angle)
            piece.translation = (dx, dy)
            self.placed.append(piece)
            self._update_exterior_contour(piece)
        finally:
            # Keep piece at chosen rotation; we won't reuse it after placement
            pass

    def _iter_orientations(self):
        if not self.rotations_on:
            yield 0.0
            return
        allowed = getattr(config, "ALLOWED_ROTATIONS", [0.0, 180.0])
        for a in allowed:
            yield float(a)

    def _place_and_score(self, piece: "Piece", angle: float):
        """Rotate piece (temporarily), choose best placement point by the nesting strategy,
        then evaluate the resulting partial solution by TOPOS criteria."""
        orig_rot = piece.rotation
        try:
            if self.rotations_on:
                piece.rotate(angle)
            dx, dy = self._select_placement_point(piece)
            if dx is None:
                return None
            score = self._evaluate_partial_solution(piece, dx, dy)
            return (score, dx, dy, angle)
        finally:
            # restore orientation
            if self.rotations_on:
                piece.rotate(orig_rot)

    # ---------- Nesting strategy (Sect. 2.3) ----------

    def _select_placement_point(self, piece: "Piece") -> Tuple[Optional[float], Optional[float]]:
        # First piece: put it bottom-left (to match our y-down convention)
        if not self.placed:
            return 0.0, self.container.height - piece.height

        # Gather NFP candidate points (vertices + extended-search samples on edges)
        candidates: list[Tuple[float, float]] = []
        moving_path = utils.clean_polygon_coordinates(piece.get_outer_path())

        contour = self._exterior_contour
        rings: list[list[Tuple[float, float]]] = []
        if isinstance(contour, Polygon):
            rings.append(list(contour.exterior.coords))
        elif contour is not None:
            for geom in getattr(contour, "geoms", []):
                if isinstance(geom, Polygon):
                    rings.append(list(geom.exterior.coords))

        # Build candidates from each exterior ring
        for ring in rings:
            ring = utils.clean_polygon_coordinates(ring)
            for nfp_path in self._nfp_polygons(ring, moving_path):
                if len(nfp_path) < 2:
                    continue
                # vertices
                candidates.extend(nfp_path[:-1])  # exclude closing duplicate
                # extended-search samples (uniform along edges)
                for (x0, y0), (x1, y1) in zip(nfp_path, nfp_path[1:]):
                    for t in range(1, self.edge_samples):
                        tt = t / self.edge_samples
                        candidates.append((x0 + tt * (x1 - x0), y0 + tt * (y1 - y0)))

        if not candidates:
            # Fallback to BL gravity relative to anchor
            ax, ay = self.anchor(piece)
            return self.gravitate(piece, ax, ay)

        # Evaluate objective for each feasible candidate
        best_val = float("inf")
        best_xy = (None, None)

        ps_rect = self._partial_solution_rect()
        for (cx, cy) in candidates:
            if not self._fits(piece, cx, cy):
                continue
            val = self._location_objective(piece, cx, cy, ps_rect)
            if val < best_val:
                best_val = val
                best_xy = (cx, cy)

        if best_xy[0] is None:
            # Fallback to gravity
            ax, ay = self.anchor(piece)
            return self.gravitate(piece, ax, ay)

        # Optional: light gravity polish (keeps “tightness”)
        # if getattr(config, "NFP_GRAVITATE_ON", True):
        #     gx, gy = self.gravitate(piece, best_xy[0], best_xy[1])
        #     if self._fits(piece, gx, gy):
        #         return gx, gy
        return best_xy

    def _location_objective(self, piece: "Piece", dx: float, dy: float, ps_rect: Tuple[float,float,float,float]) -> float:
        """Min-area / min-length / max-overlap between rectangular enclosures."""
        p_rect = self._piece_rect_at(piece, dx, dy)
        if self.nesting_strategy == "min_area":
            rect = self._rect_union(ps_rect, p_rect)
            return self._rect_area(rect)
        elif self.nesting_strategy == "min_length":
            rect = self._rect_union(ps_rect, p_rect)
            return rect[2] - rect[0]  # width (x-extent)
        elif self.nesting_strategy == "max_overlap":
            # Maximize overlap ⇒ minimize negative overlap
            return -self._rect_overlap_area(ps_rect, p_rect)
        else:
            raise ValueError(f"Unknown nesting_strategy: {self.nesting_strategy}")

    # ---------- Evaluation criteria (Sect. 2.4) ----------

    def _evaluate_partial_solution(self, piece: "Piece", dx: float, dy: float) -> float:
        """Return WASTE_rel - OVERLAP_abs + DISTANCE_rel across selected terms."""
        # Rectangles
        ps_rect_before = self._partial_solution_rect()
        p_rect = self._piece_rect_at(piece, dx, dy)
        union_rect = self._rect_union(ps_rect_before, p_rect)

        # Areas
        union_area = self._rect_area(union_rect)
        total_piece_area = sum(utils.polygon_area(p.get_outer_path()) for p in self.placed) + \
                           utils.polygon_area(piece.get_outer_path())

        # Terms
        waste_rel = 0.0
        if "waste" in self.eval_terms:
            # Waste = rectangular enclosure area minus sum of polygon areas, normalized by piece area
            piece_area = max(utils.polygon_area(piece.get_outer_path()), 1e-12)
            waste = union_area - total_piece_area
            waste_rel = waste / piece_area

        overlap_abs = 0.0
        if "overlap" in self.eval_terms:
            # Sum overlap between p_rect and each placed piece's rectangle
            for q in self.placed:
                q_rect = self._piece_rect_at(q, *q.translation)
                overlap_abs += self._rect_overlap_area(p_rect, q_rect)

        distance_rel = 0.0
        if "distance" in self.eval_terms:
            # Distance between centers of p_rect and current partial solution rect (before placement),
            # normalized by (piece_rect_width + piece_rect_height)
            pcx, pcy = self._rect_center(p_rect)
            scx, scy = self._rect_center(ps_rect_before)
            dist = math.hypot(pcx - scx, pcy - scy)
            pw = (p_rect[2] - p_rect[0])
            ph = (p_rect[3] - p_rect[1])
            denom = max(pw + ph, 1e-9)
            distance_rel = dist / denom

        # Combine (unit weights)
        score = 0.0
        if "waste" in self.eval_terms:
            score += waste_rel
        if "overlap" in self.eval_terms:
            score -= overlap_abs
        if "distance" in self.eval_terms:
            score += distance_rel
        return score

    # ---------- Initial sorting (Sect. 2.5, Fig. 6) ----------

    def _apply_initial_sorting(self, key: str):
        metrics = {}
        for pid, p in self.layout.order.items():
            poly = Polygon(utils.clean_polygon_coordinates(p.get_outer_path()))
            minx, miny, maxx, maxy = poly.bounds
            width = maxx - minx
            height = maxy - miny
            area = max(poly.area, 0.0)
            hull_area = max(poly.convex_hull.area, 1e-12)
            rect_area = max(width * height, 1e-12)
            rectangularity = area / rect_area
            concavity = (hull_area - area) / hull_area  # ∈[0,1]

            if key == "length":
                metrics[pid] = width             # decreasing
            elif key == "area":
                metrics[pid] = area              # decreasing
            elif key == "concavity":
                metrics[pid] = concavity         # decreasing
            elif key == "rectangularity":
                metrics[pid] = rectangularity    # increasing
            elif key == "total_area":
                # If quantities are already expanded in Layout, this equals 'area'
                metrics[pid] = area              # decreasing
            else:
                raise ValueError(f"Unknown sorting_key: {key}")

        if key == "rectangularity":
            sorted_ids = sorted(metrics.keys(), key=lambda pid: metrics[pid])  # increasing
        else:
            sorted_ids = sorted(metrics.keys(), key=lambda pid: metrics[pid], reverse=True)

        self.layout = Layout(OrderedDict((pid, self.layout.order[pid]) for pid in sorted_ids))

    # ---------- Geometry utilities ----------

    def _partial_solution_rect(self) -> Tuple[float,float,float,float]:
        if not self.placed:
            return (0.0, 0.0, 0.0, 0.0)
        xs, ys = [], []
        for p in self.placed:
            for (x, y) in utils._translate_polygon(p.get_outer_path(), *p.translation):
                xs.append(x)
                ys.append(y)
        return (min(xs), min(ys), max(xs), max(ys))

    def _piece_rect_at(self, piece: "Piece", dx: float, dy: float) -> Tuple[float,float,float,float]:
        xs, ys = [], []
        for (x, y) in utils._translate_polygon(piece.get_outer_path(), dx, dy):
            xs.append(x); ys.append(y)
        return (min(xs), min(ys), max(xs), max(ys))

    @staticmethod
    def _rect_union(a, b):
        if a == (0.0,0.0,0.0,0.0):
            return b
        return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))

    @staticmethod
    def _rect_area(r):
        return max(0.0, (r[2] - r[0])) * max(0.0, (r[3] - r[1]))

    @staticmethod
    def _rect_overlap_area(a, b):
        dx = min(a[2], b[2]) - max(a[0], b[0])
        dy = min(a[3], b[3]) - max(a[1], b[1])
        return max(0.0, dx) * max(0.0, dy)

    @staticmethod
    def _rect_center(r):
        return ((r[0] + r[2]) * 0.5, (r[1] + r[3]) * 0.5)

    # ---------- NFP with full polygon paths (to enable edge "extended search") ----------

    def _nfp_polygons(self, stationary: list[tuple[float,float]], moving: list[tuple[float,float]]) -> list[list[tuple[float,float]]]:
        """
        Return list of NFP polygon paths (floats) for moving about stationary.
        Matches utils.no_fit_polygon but **preserves paths** for edge sampling.
        """
        import pyclipper
        # Reference at the top-leftmost vertex of moving (y grows downward)
        idx = utils.find_topleft_vertex(moving)
        rx, ry = moving[idx]
        moving_centered = [(x - rx, y - ry) for (x, y) in moving]

        A = utils.to_clipper(stationary)
        B = [(-x, -y) for (x, y) in moving_centered]
        B = utils.to_clipper(B)

        raw = pyclipper.MinkowskiSum(B, A, True)
        return [utils.from_clipper(p) for p in raw]
