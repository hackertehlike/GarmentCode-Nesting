from collections import OrderedDict
from typing import Literal, Optional, List, Tuple, Union

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

    def usage_BB(self) -> float:
        flattened = self._flatten_piece_list()
        if not flattened:
            return 0.0
        xs = [v[0] for v in flattened]
        ys = [v[1] for v in flattened]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        bbox_area = (max_x - min_x) * (max_y - min_y)
        total_area = sum(utils.polygon_area(p.get_outer_path()) for p in self.placed)
        ratio = total_area / bbox_area if bbox_area > 0 else 0.0
        return ratio if self.layout_is_valid() else 0.0

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

    # ── UNIFIED α-SHAPE (CONCAVE HULL) HELPER ────────────────────────────────────

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
            poly = Polygon([(x + piece.translation[0], y + piece.translation[1])
                            for x, y in raw])
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

            hull = Polygon(snapped)

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
        piece_poly = Polygon(shifted)

        if self._exterior_contour is None:
            self._exterior_contour = piece_poly
            return

        merged = self._exterior_contour.union(piece_poly)
        self._exterior_contour = merged
        return

    def concave_hull_utilization(self) -> float:
        if not self.placed:
            return 0.0

        pts = self._flatten_piece_list()
        hull = self.alpha_shape(pts,
                                trim_ratio=config.HULL_TRIM_RATIO,
                                interior_spacing=config.INTERIOR_SAMPLE_SPACING,
                                boundary_spacing=config.BOUNDARY_SAMPLE_SPACING)
        hull_area = hull.area if not hull.is_empty else 0.0
        total_area = sum(utils.polygon_area(p.get_outer_path()) for p in self.placed)
        return (total_area / hull_area) if (hull_area > 0 and self.layout_is_valid()) else 0.0

@register_decoder("BL")
class BottomLeftDecoder(PlacementEngine):
    def __init__(self, layout: Layout, container: Container, *, step=None, **kwargs):
        super().__init__(layout, container)
        self.step = step

    
    def decode(self) -> list[Tuple[str, float, float, float]]:
        """
        For each piece, compute (dx,dy) by either:
          - Traditional BL (fully left, then fully down), if self.traditional=True
          - Iterative gravitate otherwise.
        Place them in order and record their translations + rotation.
        """
        for piece_id, piece in self.layout.order.items():
            # 1) Anchor against top‐right
            x0, y0 = self.anchor(piece)

            # 2) Gravitate according to selected mode
            if self.gravitate_once:
                dx, dy = self._gravitate_once(piece, x0, y0, step=self.step)
            else:
                dx, dy = self.gravitate(piece, x0, y0, step=self.step)

            # 3) Place and record
            piece.translation = (dx, dy)
            self.placed.append(piece)

        # Return a list of (id, x, y, rotation) for all placed pieces
        return [(p.id, p.translation[0], p.translation[1], p.rotation) for p in self.placed]


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
    NFPDecoder inherits concave‐hull utilities from PlacementEngine but
    provides its own decode() and _find_best_position() using NFP logic.
    """

    def __init__(self, layout: Layout, container: Container, *, step=None, **kwargs):
        super().__init__(layout, container)
        self._nfp_cache = {}

    def decode(self):
        for piece in self.layout.order.values():
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
        all_nfp_pts: List[Tuple[float, float]] = []
        for ring_coords in rings:
            pts = utils.no_fit_polygon(ring_coords, moving_path)
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

        best_score = -1.0
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
            self._nfp_cache[key] = utils.no_fit_polygon(
                stationary.get_outer_path(),
                moving.get_outer_path()
            )
        return self._nfp_cache[key]