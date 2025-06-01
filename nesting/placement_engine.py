from collections import OrderedDict
from typing import Optional
from .layout import Layout, Container, Piece
from abc import ABC, abstractmethod
import numpy as np, scipy.spatial as sps
import random
from nesting import utils
import nesting.config as config
# from shapely.geometry import MultiPoint
# import alphashape
from shapely.geometry import Point, Polygon, MultiPoint, MultiLineString
from shapely.ops import unary_union, polygonize
from scipy.spatial import Delaunay
import numpy as np


DECODER_REGISTRY: dict[str, type] = {}

def register_decoder(name: str):
    def deco(cls):
        DECODER_REGISTRY[name] = cls
        return cls
    return deco


class PlacementEngine():
    """
    Base class for layout placement strategies.
    """

    def __init__(self,
                 layout: Layout,
                 container: Container,
                 **kwargs,                      # ← absorb any extra arguments
    ):        
        self.layout = layout
        self.container = container
        self.placed: list[Piece] = []
        self._last_hull: Optional[Polygon] = None  # for concave hull utilization

    def decode(self):
        pass

    @property
    def concave_hull_polygon(self) -> Optional[Polygon]:
        """Return the last computed concave hull, if any."""
        return self._last_hull

    def _fits(self, piece: Piece, dx: float, dy: float) -> bool:
        """Return True iff piece can be put at (dx, dy) [cm]"""
        poly = utils._translate_polygon(piece.get_outer_path(), dx, dy)

        # container boundaries (all cm)
        xs, ys = zip(*poly)
        if (min(xs) < 0 or max(xs) > self.container.width
                or min(ys) < 0 or max(ys) > self.container.height):
            return False

        # intersections with already‑placed parts
        for other in self.placed:
            ox, oy = other.translation
            other_poly = utils._translate_polygon(
                other.get_outer_path(), ox, oy)
            if utils.polygons_overlap(poly, other_poly):
                return False

        return True    

    def anchor(self, piece):
        """Default: push piece against the container’s top-right corner."""
        # get the vertices of the piece

        # print (f"Piece {piece.id} anchoring")
        vertices = piece.get_outer_path()
        # get the x and y values of the vertices
        xs = [v[0] for v in vertices]
        # ys = [v[1] for v in vertices]
        # get the min and max x and y values   
        min_x = min(xs)
        max_x = max(xs)
        # min_y = min(ys)
        # max_y = max(ys)
        start_x = self.container.width  - (max_x - min_x)
        start_y = 0.0
        return start_x, start_y
    
    def usage_BB(self):
        """
        Returns the ratio of the used area of the bounding box of placed pieces
        to the area of that bounding box.

        Call AFTER decode() to get the correct values.
        """
        flattened_vertices = [(x + piece.translation[0],
                               y + piece.translation[1])
                              for piece in self.placed
                              for x, y in piece.get_outer_path()]
        # print (f"flattened vertices: {flattened_vertices}")
        x_vals = [v[0] for v in flattened_vertices]
        y_vals = [v[1] for v in flattened_vertices]
        min_x = min(x_vals)
        max_x = max(x_vals)
        min_y = min(y_vals)
        max_y = max(y_vals)

        # print(f"Bounding box: ({min_x}, {min_y}), ({max_x}, {max_y})")
        
        # calculate the area total of placed pieces
        total_area = sum([utils.polygon_area(p.get_outer_path()) for p in self.placed])
        # calculate the area of the bounding box
        bounding_box_area = (max_x - min_x) * (max_y - min_y)

        # calculate the ratio of the used area of the bounding box to the area of the bounding box
        ratio = total_area / bounding_box_area

        if self.layout_is_valid():
            return ratio
        else:
            return 0.0  # if the layout is not valid, return 0.0


    def rest_length(self):
        """
        Returns the length of the rest of the container that is not used by the pieces.
        Call AFTER decode() to get the correct values.
        """
        # the rightmost x coordinate of the bounding box
        flattened_vertices = self._flatten_piece_list()
        x_vals = [v[0] for v in flattened_vertices]
        # y_vals = [v[1] for v in flattened_vertices]
        # min_x = min(x_vals)
        max_x = max(x_vals)

        return self.container.width - max_x
    
    
    def alpha_shape(self,
                points: list[tuple[float, float]],
                *,
                trim_ratio: float = 7.0,
                interior_spacing: float = 5.0,
                boundary_spacing: float = 1.0
               ) -> Polygon:
        """
        Compute the concave hull of a set of points using the alpha shape algorithm.
        Args:
            points (list[tuple[float, float]]): List of points to form the concave hull.
            trim_ratio (float): Ratio to trim the edges of the Delaunay triangulation.
            interior_spacing (float): Spacing for sampling points in the interior of the polygon.
            boundary_spacing (float): Spacing for sampling points along the boundary of the polygon.
        Returns:
            Polygon: The concave hull of the points.
        Raises:
            ValueError: If the points list is empty or contains fewer than 4 points.
        """

        # Ensure spacings are floats
        interior_spacing = float(interior_spacing)
        boundary_spacing = float(boundary_spacing)

        # --- helper: sample a polygon’s interior ---
        def sample_interior(poly: Polygon) -> list[tuple[float,float]]:
            minx, miny, maxx, maxy = poly.bounds
            # center the grid so we don't start exactly on the boundary
            xs = np.arange(minx + interior_spacing/2.0, maxx, interior_spacing)
            ys = np.arange(miny + interior_spacing/2.0, maxy, interior_spacing)
            pts = []
            for x in xs:
                for y in ys:
                    if poly.contains(Point(x, y)):
                        pts.append((float(x), float(y)))
            return pts

        # --- helper: sample a polygon’s exterior ---
        def sample_boundary(poly: Polygon) -> list[tuple[float,float]]:
            coords = list(poly.exterior.coords)
            pts = []
            for (x0, y0), (x1, y1) in zip(coords, coords[1:]):
                edge_len = float(np.hypot(x1 - x0, y1 - y0))
                # compute number of sample points as an integer
                n_samples = max(int(edge_len / boundary_spacing), 1)
                for t in np.linspace(0.0, 1.0, n_samples, endpoint=False):
                    pts.append((float(x0 + t * (x1 - x0)),
                                float(y0 + t * (y1 - y0))))
            return pts

        # --- build sampled point cloud over all placed pieces ---
        all_pts: list[tuple[float, float]] = []
        centroids = []
        for piece in self.placed:
            raw = piece.get_outer_path()
            poly = Polygon([(x + piece.translation[0], y + piece.translation[1])
                            for x, y in raw])
            # boundary first (dense)
            all_pts.extend(sample_boundary(poly))
            # then a few interior points
            all_pts.extend(sample_interior(poly))
            centroids.append(poly.centroid.coords[0])

        pts = np.array(all_pts, dtype=float)
        if pts.shape[0] < 4:
            hull = MultiPoint(pts).convex_hull
            self._last_hull = hull
            return hull

        # --- Delaunay + trim by edge-length ---
        tri = Delaunay(pts)
        lengths = []
        for simplex in tri.simplices:
            i, j, k = simplex
            lengths.extend([
                np.linalg.norm(pts[i] - pts[j]),
                np.linalg.norm(pts[j] - pts[k]),
                np.linalg.norm(pts[k] - pts[i]),
            ])
        median_len = float(np.median(lengths))
        max_edge   = median_len * trim_ratio

        kept = set()
        for simplex in tri.simplices:
            i, j, k = simplex
            lij = np.linalg.norm(pts[i] - pts[j])
            ljk = np.linalg.norm(pts[j] - pts[k])
            lki = np.linalg.norm(pts[k] - pts[i])
            if lij <= max_edge and ljk <= max_edge and lki <= max_edge:
                kept.update({
                    tuple(sorted((i, j))),
                    tuple(sorted((j, k))),
                    tuple(sorted((k, i))),
                })
        if not kept:
            hull = MultiPoint(pts).convex_hull
            self._last_hull = hull
            return hull

        # --- polygonize the kept edges ---
        mls = MultiLineString([ (tuple(pts[i]), tuple(pts[j])) for i,j in kept ])
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
            # at least one centroid got cut off ⇒ merge with global convex hull
            full_hull = MultiPoint(pts).convex_hull
            hull = unary_union([hull, full_hull])

        if config.SNAP:
            # Snap vertices close to container boundaries and insert corner points
            tol = config.SNAP_TOLERANCE
            w, h = self.container.width, self.container.height
            coords = list(hull.exterior.coords)
            snapped: list[tuple[float, float]] = []
            n = len(coords)

            def is_on_horiz(pt: tuple[float, float]) -> bool:
                return abs(pt[1]) < tol or abs(pt[1] - h) < tol

            def is_on_vert(pt: tuple[float, float]) -> bool:
                return abs(pt[0]) < tol or abs(pt[0] - w) < tol

            for i, (x, y) in enumerate(coords):
                prev = coords[i - 1]
                nxt  = coords[(i + 1) % n]

                # snap X coordinate
                if abs(x) < tol or (abs(prev[0]) < tol and abs(nxt[0]) < tol):
                    x = 0.0
                elif abs(x - w) < tol or (abs(prev[0] - w) < tol and abs(nxt[0] - w) < tol):
                    x = w

                # snap Y coordinate
                if abs(y) < tol or (abs(prev[1]) < tol and abs(nxt[1]) < tol):
                    y = 0.0
                elif abs(y - h) < tol or (abs(prev[1] - h) < tol and abs(nxt[1] - h) < tol):
                    y = h

                curr = (x, y)
                snapped.append(curr)

                # Prepare snapped version of next for corner decision:
                x2, y2 = coords[(i + 1) % n]
                if abs(x2) < tol:
                    x2 = 0.0
                elif abs(x2 - w) < tol:
                    x2 = w
                if abs(y2) < tol:
                    y2 = 0.0
                elif abs(y2 - h) < tol:
                    y2 = h
                nxt_snapped = (x2, y2)

                # If one is on horizontal boundary and the other on vertical, insert corner
                if (is_on_horiz(curr) and is_on_vert(nxt_snapped)) or \
                (is_on_vert(curr)  and is_on_horiz(nxt_snapped)):
                    # x from the vertical‐on point, y fbrom the horizontal‐on point
                    corner = (
                        curr[0] if is_on_vert(curr) else nxt_snapped[0],
                        curr[1] if is_on_horiz(curr) else nxt_snapped[1]
                    )
                    # Only insert if it’s not already identical to last snapped
                    if corner not in snapped:
                        snapped.append(corner)

            # Rebuild hull from the augmented vertex list
            hull = Polygon(snapped)
        
        self._last_hull = hull
        return hull


    
    def concave_hull_utilization(self) -> float:
        """
        Returns the ratio of the used area of the concave hull of placed pieces
        to the area of that hull. Call AFTER decode().
        """
        if not self.placed:
            return 0.0
        # Flatten all vertices into global coordinates
        flattened = [(x + piece.translation[0], y + piece.translation[1])
                     for piece in self.placed
                     for x, y in piece.get_outer_path()]
        # Build concave hull
        hull = self.alpha_shape(flattened, trim_ratio=config.HULL_TRIM_RATIO, 
                                interior_spacing=config.INTERIOR_SAMPLE_SPACING,
                                boundary_spacing=config.BOUNDARY_SAMPLE_SPACING)
        self._last_hull = hull
        #print(f"Concave hull: {hull}")
        if hull.is_empty:
            raise ValueError("Concave hull is empty, cannot compute utilization.")
        hull_area = hull.area
        #print(f"Hull area: {hull_area}")
        total = sum(utils.polygon_area(p.get_outer_path()) for p in self.placed)
        #print(f"Total area of placed pieces: {total}")

        if self.layout_is_valid():
            return total / hull_area if hull_area > 0 else 0.0
        else:
            return 0.0


    def gravitate(self, piece, x, y, step=config.GRAVITATE_STEP):
        """Slide left as far as possible, then down; repeat until jammed."""
        # print (f"Piece {piece.id} gravitating")
        #print piece type
        # print (f"Piece {piece.id} type: {type(piece)}")
        moved = True
        while moved:
            moved = False
            while x - step >= 0 and self._fits(piece, x - step, y):
                x -= step
                moved = True
            while y + step <= self.container.height and self._fits(piece, x, y + step):
                y += step
                moved = True
        # print (f"Piece {piece.id} gravitated to ({x}, {y})")
        return x, y
    
    def _flatten_piece_list(self):
        return [(x + piece.translation[0], y + piece.translation[1]) for piece in self.placed
                for x, y in piece.get_outer_path()]
    
    def layout_is_valid(self) -> bool:
        n = len(self.placed)
        for i in range(n):
            pi  = self.placed[i]
            poly_i = utils._translate_polygon(pi.get_outer_path(), *pi.translation)

            # Container bounds
            xs_i, ys_i = zip(*poly_i)
            if (min(xs_i) < 0 or
                max(xs_i) > self.container.width or
                min(ys_i) < 0 or
                max(ys_i) > self.container.height):
                return False

            # Pairwise intersections (j > i prevents double checks)
            for j in range(i + 1, n):
                pj = self.placed[j]
                poly_j = utils._translate_polygon(pj.get_outer_path(), *pj.translation)
                if utils.polygons_overlap(poly_i, poly_j):
                    return False
        return True

@register_decoder("BL")
class BottomLeftDecoder(PlacementEngine):

    def __init__(self, layout, container, *, step=None, **kwargs):
        super().__init__(layout, container)
        self.step = step

    def decode(self):
        for piece_id, piece in self.layout.order.items():
            x0, y0 = self.anchor(piece)
            dx, dy = self.gravitate(piece, x0, y0)
            # print (f"Piece {piece.id} placed at ({dx}, {dy})")
            self.placed.append((piece))
            piece.translation = (dx, dy)
            # piece.rotate(piece.rotation)

        return [(p.id, *p.translation, p.rotation) for p in self.placed]

@register_decoder("Greedy")

class GreedyBLDecoder(BottomLeftDecoder):
    def __init__(self, layout, container, *, step=None, **kwargs):
        super().__init__(layout, container, step=config.GRAVITATE_STEP, **kwargs)

        # sort the pieces by area
        # Create a list of (piece_id, piece) tuples sorted by area
        sorted_pieces = sorted(self.layout.order.items(), key=lambda item: item[1].bbox_area, reverse=True)

        # Update layout.order with the sorted list
        self.layout.order = dict(sorted_pieces)

@register_decoder("Random")
class RandomDecoder(PlacementEngine):
    def __init__(self,
                 layout: Layout,
                 container: Container,
                 *,
                 rotations_on: bool = config.ENABLE_ROTATIONS,
                 **kwargs):
        # shuffle the insertion order
        ids = list(layout.order)
        random.shuffle(ids)
        shuffled = OrderedDict((i, layout.order[i]) for i in ids)
        super().__init__(Layout(shuffled), container, **kwargs)
        self.rotations_on = rotations_on

    def decode(self):
        # optional random rotations
        if self.rotations_on:
            for p in self.layout.order.values():
                p.rotate(random.choice(config.ALLOWED_ROTATIONS))

        # bottom-left place on *this* layout
        bl = BottomLeftDecoder(self.layout, self.container, step=config.GRAVITATE_STEP)
        placements = bl.decode()

        # mirror the placed list so that usage_BB() sees it
        self.placed = bl.placed

        return placements


@register_decoder("NFP")
class NFPDecoder(PlacementEngine):

    def __init__(self, layout, container, *, step=None, **kwargs):
        super().__init__(layout, container)
        # self.placed = []  # (piece, x, y)
        self.container = container
        self._nfp_cache = {}
        

    def decode(self):
       # go in layout order
        for piece in self.layout.order.values():
            # find the best position for the piece
            # print(f"Placing piece {piece.id}...")
            best_x, best_y = self._find_best_position(piece)
            # print(f"Placing piece {piece.id} at ({best_x}, {best_y})")
            self.placed.append((piece))
            piece.translation = (best_x, best_y)
        
        # print pieces
        # print([piece.translation for piece in self.layout.order.values()])

        return [(p.id, *p.translation, p.rotation) for p in self.placed]


    def _find_best_position(self, piece: Piece, gravitate_on = False):
        """
        Finds the best position to place a given piece within the container.
        This method determines the optimal position for a piece by evaluating 
        potential placements based on the no-fit polygon (NFP) and the inner 
        fit rectangle of the container. It ensures that the piece does not 
        overlap with already placed pieces and adheres to the container's 
        boundaries.
        If no valid position is found with the NFP, it defaults to gravitating
        the piece to the bottom-left corner of the container.
        Args:
            piece (Piece): The piece to be placed in the container.
            gravitate_on (bool, optional): If True, the piece will be 
                gravitated towards the bottom-left corner after determining 
                the best position. Defaults to False.
        Returns:
            tuple: A tuple (best_x, best_y) representing the coordinates of 
            the best position for the piece. If no valid position is found, 
            returns (None, None).
        """


        best_x, best_y = None, None

        # get inner fit rectangle
        # for the piece in the container
        
        inner_fit = self.container.inner_fit_rectangle(piece)

        if not inner_fit:
            return None, None

        # iterate over nfps all previously placed pieces
        # choose the bottom leftest point out of all vertices that
        # are inside the inner fit rectangle and on one of the nfp edges
        # note: y grows down, x grows right

        # if first piece, place it at the bottom left corner
        if not self.placed:
            best_x = 0
            best_y = self.container.height - piece.height
            return best_x, best_y

        for other in self.placed:
            ox, oy = other.translation
            # print(f"Checking piece {piece.id} with {other.id}")
            nfp = self._nfp(other, piece)
            # print(f"Piece {piece.id} nfp with {other.id}: {nfp}")
            for x, y in nfp:
                # translate the nfp wrt the stationary piece
                x_translated = x + ox
                y_translated = y + oy
                # check if the translated nfp is inside the inner fit rectangle
                
                
                x0 = min(p[0] for p in inner_fit)
                x1 = max(p[0] for p in inner_fit)
                y0 = min(p[1] for p in inner_fit)
                y1 = max(p[1] for p in inner_fit)

                if x0 <= x_translated <= x1 and y0 <= y_translated <= y1:
                    # check if the translated nfp is inside the container
                    if best_x is None or (x_translated < best_x or
                                          (x_translated == best_x and y_translated < best_y)):
                    
                        # check if we are intersecting with other placed pieces
                        #self.num_comparisons += 1
                        if self._fits(piece, x_translated, y_translated):
                            # update the best position
                            best_x = x_translated
                            best_y = y_translated
                            # print(f"Piece {piece.id} best position: ({best_x}, {best_y})")

        # if no position was found
        if best_x is None or best_y is None:
            # Could not place via NFP → behave like Bottom‑Left
            ax, ay = self.anchor(piece)
            best_x, best_y = (self.gravitate(piece, ax, ay))
        
        # gravitate the piece to the bottom left corner
        if gravitate_on:
            best_x, best_y = self.gravitate(piece, best_x, best_y)
        
        # print(f"Piece {piece.id} placed at ({best_x}, {best_y})")
        return best_x, best_y


    def _nfp(self, stationary, moving):
        """
        Returns the no-fit polygon (NFP) between two pieces.
        """
        key = (stationary.id, moving.id)
        if key not in self._nfp_cache:
            self._nfp_cache[key] = utils.no_fit_polygon(stationary.get_outer_path(),
                                                        moving.get_outer_path())
                                                        
        return self._nfp_cache[key]