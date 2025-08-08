import math
from types import MappingProxyType
from typing import Dict, Tuple

from shapely import MultiPolygon, unary_union
from nesting import utils
from shapely.geometry import Polygon, LineString
from shapely.ops import split as shapely_split
from shapely.errors import GEOSException as TopologyException

from collections import OrderedDict
import copy
class Piece:
    """
    A class representing a piece in the layout.
    Each piece has an ID, a list of outer_path, a rotation angle, and a locked state.
    """

    def __init__(self, original_path, id=None):
        self.id = id
        self.root_id     = id

        self.inner_path = original_path     # original inner geometry (in cm)
        self.outer_path = original_path.copy()     # original outer geometry (in cm)
        
        #self.add_seam_allowance()  # original outer geometry (in cm)
        # self.translation = (0.0, 0.0)
        self.rotation = 0 # wrt to the original piece
        self._translation = (0, 0)
        self.locked = False
        self.parent_id = None  # ID of the parent piece, if this is a child piece
        #self.scale = 1.0


        #bounding box
        self.update_bbox()

    def update_bbox(self) -> None:
        #print("Updating bounding box for piece ", self.id)

        # get the min and max x and y coordinates
        xs = [pt[0] for pt in self.outer_path]
        ys = [pt[1] for pt in self.outer_path]
        # print(f"Piece {self.id} outer_path: {xs}, {ys}")
        self.min_x = min(xs)
        self.max_x = max(xs)
        self.min_y = min(ys)
        self.max_y = max(ys)
        # print(f"Types: {type(self.min_x)}, {type(self.max_x)}, {type(self.min_y)}, {type(self.max_y)}")
        self.bbox_width = self.max_x - self.min_x
        self.bbox_height = self.max_y - self.min_y
        self.bbox_area = self.bbox_width * self.bbox_height
        # print(f"Piece {self.id} bounding box updated: ({self.min_x}, {self.min_y}), ({self.max_x}, {self.max_y})")

    
    def get_inner_path(self) -> list[list[float]]:
        """Returns the path of the piece as a list of [x, y] outer_path."""
        return self.inner_path
    
    def get_outer_path(self) -> list[list[float]]:
        """Returns the path of the piece as a list of [x, y] outer_path."""
        return self.outer_path

    @property
    def height(self) -> float:
        """Axis-aligned bounding-box height."""
        return self.max_y - self.min_y

    @property
    def width(self) -> float:
        """Axis-aligned bounding-box width."""
        return self.max_x - self.min_x

    def rotate(self, angle: float):
        """
        Rotate the piece *in place* by angle (in degrees).
        Modifies the piece's outer_path and inner_path.
        The rotation is done around the origin (0, 0) of the piece (bounding box).
        """

        #print (f"Rotating piece {self.id} by {angle} degrees")

        if (angle == 0):
            return
        
        angle_rad = math.radians(angle)
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)

        for i, (x, y) in enumerate(self.inner_path):
            # Apply rotation matrix
            new_x = x * cos_angle - y * sin_angle
            new_y = x * sin_angle + y * cos_angle
            self.inner_path[i] = (new_x, new_y)

        for i, (x, y) in enumerate(self.outer_path):
            # Apply rotation matrix
            new_x = x * cos_angle - y * sin_angle
            new_y = x * sin_angle + y * cos_angle
            self.outer_path[i] = (new_x, new_y)
        # bookkeeping
        self.rotation += angle
        self.rotation = self.rotation % 360
        # update the bounding box
        # shift
        # utils.shift_coordinates(self.outer_path)
        self.outer_path = utils.shift_coordinates(self.outer_path)
        self.inner_path = utils.shift_coordinates(self.inner_path)

        self.update_bbox()

    def translate(self, dx: float, dy: float):
        """Translate the piece *in place* by (dx, dy)."""
        for i, (x, y) in enumerate(self.outer_path):
            self.outer_path[i][0] = x + dx
            self.outer_path[i][1] = y + dy

        # bookkeeping
        self._translation = (self._translation[0] + dx, self._translation[1] + dy)

    @property
    def translation(self) -> Tuple[float, float]:
        """Returns the translation of the piece."""
        #return (self._translation[0] * self.scale, self._translation[1] * self.scale)
        return (self._translation[0], self._translation[1])

    @translation.setter
    def translation(self, value: Tuple[float, float]):
        """Sets the translation of the piece."""
        # self._translation = (value[0] / self.scale, value[1] / self.scale)
        self._translation = (value[0], value[1])

        # self.translate(value[0], value[1])

    def add_seam_allowance(self, allowance: float = 1.0) -> None:
        """
        Update this piece’s outer path with a seam allowance.
        The inner path is unchanged.
        This method calls utils.compute_offset_path, which handles all Pyclipper details.
        """
        contour = self.get_inner_path()
        if not contour or len(contour) < 3:
            raise ValueError("Piece has no inner path to offset.")

        #print(f"Adding seam allowance of {allowance} to piece {self.id}")
        new_outer = utils.compute_offset_path(contour, allowance)
        self.outer_path = new_outer
        self.update_bbox()
        #print(f"Piece {self.id} outer path updated with seam allowance")

    def reset_rotation(self) -> None:
        """
        Reset the rotation of the piece to 0 degrees.
        """
        needed_rotation = (360 - self.rotation) % 360
        if needed_rotation != 0:
            self.rotate(needed_rotation)
            #self.rotation = 0
            print(f"Piece {self.id} rotation reset to 0 degrees")
        else:
            print(f"Piece {self.id} is already at 0 degrees")

    # def split(self) -> Tuple["Piece", "Piece"]:
            
    #         #TODO: FIX
    #         """
    #         Split this piece into two new Piece objects by a vertical line through the
    #         middle of its bounding box. This version handles cases where the inner
    #         polygon may break into more than two fragments. We group those fragments
    #         into “left” vs. “right” halves by centroid-x, then union them back.
    #         """
    #         original_rotation = self.rotation
    #         # Build Shapely geometry for inner polygon only
    #         poly_inner = Polygon(self.inner_path)

    #          # Compute vertical split-line at mid-x of inner bounding box
    #         minx, miny, maxx, maxy = poly_inner.bounds
    #         midx = (minx + maxx) / 2.0
    #         split_line = LineString([(midx, miny - 1.0), (midx, maxy + 1.0)])

    #         # Split only the inner polygon
    #         result_inner = shapely_split(poly_inner, split_line)
            
    #         # Get all polygon parts from the split
    #         parts_inner: list[Polygon] = [g for g in result_inner.geoms if isinstance(g, Polygon)]
            
    #         # For now, just take the first two parts (comment out fragment grouping)
    #         if len(parts_inner) < 2:
    #             raise ValueError(f"Piece {self.id} split resulted in less than 2 parts: {len(parts_inner)} parts found.")
    #         elif len(parts_inner) > 2:
    #             # More than two parts, raise an error
    #             raise ValueError(f"Piece {self.id} split resulted in more than 2 parts: {len(parts_inner)} parts found.")

    #         inner_left = parts_inner[0]
    #         inner_right = parts_inner[1]

    #         # Fragment grouping logic (commented out for now)
    #         # Group fragments by centroid-x into left and right
    #         left_list = [g for g in parts_inner if g.centroid.x <= midx]
    #         right_list = [g for g in parts_inner if g.centroid.x > midx]
    #         if not left_list:
    #             left_list = parts_inner[:1]
    #         if not right_list:
    #             right_list = parts_inner[-1:]

    #         # Union fragments into single polygon per side
    #         def union_to_single_polygon(fragments: list[Polygon]) -> Polygon:
    #             combined = unary_union(fragments)
    #             if isinstance(combined, Polygon):
    #                 return combined
    #             if isinstance(combined, MultiPolygon):
    #                 return max(combined.geoms, key=lambda p: p.area)
    #             raise ValueError(f"Unexpected geometry type after union: {type(combined)}")

    #         inner_left = union_to_single_polygon(left_list)
    #         inner_right = union_to_single_polygon(right_list)

    #         # Create new Piece instances from inner paths
    #         new_pieces: list[Piece] = []
    #         for idx, inner_geom in enumerate((inner_left, inner_right), start=1):
    #             # Determine local origin from inner geometry bounding box
    #             ox_min, oy_min, _, _ = inner_geom.bounds
    #             # Convert inner_geom exterior to local coordinates
    #             i_coords_local = [(x - ox_min, y - oy_min) for x, y in inner_geom.exterior.coords]
    #             new_id = f"{self.id}_split_{'left' if idx == 0 else 'right'}"
    #             new_piece = Piece(i_coords_local, id=new_id)
    #             new_piece.parent_id = self.id
    #             new_piece.root_id = getattr(self, "root_id", self.id)
                
    #             # Set translation (accounting for local origin offset)
    #             tx, ty = self.translation
    #             new_piece.translation = (tx + ox_min, ty + oy_min)
                
    #             # Apply seam allowance to generate outer_path and update bbox
    #             new_piece.add_seam_allowance()
    #             new_piece.update_bbox()
                
    #             # Now rotate the new piece to match the original rotation
    #             if original_rotation != 0:
    #                 new_piece.rotate(original_rotation)
                
    #             new_pieces.append(new_piece)

    #         # Restore original rotation to this piece (if we modified it)
    #         if original_rotation != 0:
    #             self.rotate(original_rotation)

    #         return new_pieces[0], new_pieces[1]

    def split(
        self,
        use_centroid: bool = False,
        epsilon: float = 1e-7,          # geometric tolerance
    ) -> Tuple["Piece", "Piece"]:
        """
        Split this piece into two new Piece objects by a vertical line
        through either the x-centroid (`use_centroid=True`) or the
        midpoint of its bounding box (`use_centroid=False`, default).

        The algorithm:
        1.  Intersect the polygon with the vertical line to obtain an
            even list of points  p0 … p(2k-1).
        2.  Every consecutive pair (p0,p1), (p2,p3)… is an *inside*
            segment.  For each such segment:
               • split the polygon along that segment,
               • measure the area difference between the two parts.
        3.  Keep the pair with the **smallest** area difference.
        4.  Wrap the two resulting polygons into new `Piece` objects
            that inherit
               • `parent_id`, `root_id`
               • translation, rotation, seam allowance, etc.
        """

        from shapely.geometry import Polygon, LineString, MultiPoint
        from shapely.ops import split as shapely_split, unary_union
        from shapely.errors import GEOSException

        # --- 0.  build Shapely polygon from the *inner* path ----------
        # Clean the path first by removing duplicate consecutive points
        cleaned_path = []
        for i, coord in enumerate(self.inner_path):
            if i == 0 or coord != self.inner_path[i-1]:
                cleaned_path.append(coord)
        
        # Ensure polygon is closed (first == last)
        if len(cleaned_path) > 0 and cleaned_path[0] != cleaned_path[-1]:
            cleaned_path.append(cleaned_path[0])
            
        poly_inner = Polygon(cleaned_path)
        
        # If still invalid, try to fix with buffer operation
        if not poly_inner.is_valid:
            print(f"[DEBUG] Attempting to fix invalid polygon for piece {self.id}")
            try:
                # Buffer by a tiny amount to fix self-intersections
                poly_inner = poly_inner.buffer(0)
                if poly_inner.is_valid:
                    print(f"[DEBUG] Successfully fixed polygon using buffer operation")
                else:
                    print(f"[DEBUG] Buffer operation failed to fix polygon")
            except Exception as e:
                print(f"[DEBUG] Buffer operation failed: {e}")
        
        if poly_inner.is_empty or not poly_inner.is_valid:
            print(f"[DEBUG] Piece {self.id}: Polygon validation failed")
            print(f"[DEBUG] Inner path length: {len(self.inner_path)}")
            print(f"[DEBUG] Inner path: {self.inner_path}")
            print(f"[DEBUG] Polygon is_empty: {poly_inner.is_empty}")
            print(f"[DEBUG] Polygon is_valid: {poly_inner.is_valid}")
            if not poly_inner.is_valid:
                print(f"[DEBUG] Polygon validation error: {poly_inner}")
                # Get specific validation error reason using shapely function
                try:
                    from shapely.predicates import is_valid_reason
                    reason = is_valid_reason(poly_inner)
                    print(f"[DEBUG] Validation reason: {reason}")
                except ImportError:
                    print("[DEBUG] shapely.predicates.is_valid_reason not available")
                except Exception as e:
                    print(f"[DEBUG] Error with is_valid_reason: {e}")
                
                # Additional debugging - check for common issues
                print(f"[DEBUG] Number of coordinates: {len(self.inner_path)}")
                if len(self.inner_path) > 0:
                    print(f"[DEBUG] First coordinate: {self.inner_path[0]}")
                    print(f"[DEBUG] Last coordinate: {self.inner_path[-1]}")
                    # Check if polygon is closed
                    if self.inner_path[0] != self.inner_path[-1]:
                        print("[DEBUG] Polygon may not be closed (first != last coordinate)")
                
                # Check for duplicate consecutive points
                if len(self.inner_path) > 1:
                    duplicates = [i for i in range(1, len(self.inner_path)) 
                                if self.inner_path[i] == self.inner_path[i-1]]
                    if duplicates:
                        print(f"[DEBUG] Duplicate consecutive points at indices: {duplicates}")
                        
            raise ValueError(f"Piece {self.id}: invalid polygon given.")

        minx, miny, maxx, maxy = poly_inner.bounds
        x0 = poly_inner.centroid.x if use_centroid else 0.5 * (minx + maxx)

        # a tall vertical line that surely crosses the piece
        vertical = LineString([(x0, miny - 1.0), (x0, maxy + 1.0)])

        # --- 1.  boundary ∩ line  -> even number of points ------------
        # try:
        hits = poly_inner.boundary.intersection(vertical)
        # except GEOSException as exc:
        #     raise ValueError(f"Piece {self.id}: intersection failed.") from exc

        if isinstance(hits, MultiPoint):
            pts = sorted(hits.geoms, key=lambda p: p.y)  # Extract individual points with .geoms
        else:                                 # includes single Point
            pts = [hits] if hits else []
        if len(pts) < 2 or len(pts) % 2:
            raise ValueError(
                f"Piece {self.id}: expected even # of intersections, got {len(pts)}."
            )

        # --- 2-3.  test every inside segment --------------------------
        best_pair   = None          # (p_even, p_odd)
        best_split  = None          # (polyL, polyR)
        best_delta  = float("inf")  # area difference

        for i in range(0, len(pts), 2):
            p, q = pts[i], pts[i + 1]          # inside segment endpoints

            # shave a *very* thin buffer so Shapely treats the segment as “knife”
                        
            dy  = q.y - p.y
            p0  = (p.x, p.y - epsilon * max(1.0, abs(dy)))
            q0  = (q.x, q.y + epsilon * max(1.0, abs(dy)))
            knife = LineString([p0, q0])        # **still a LineString** here

            parts_gc = shapely_split(poly_inner, knife)   # → GeometryCollection
            parts = [g for g in parts_gc.geoms if g.geom_type == "Polygon"]

            try:
                parts_gc = shapely_split(poly_inner, knife)
                parts = [g for g in parts_gc.geoms if g.geom_type == "Polygon"]
            except GEOSException:
                continue                       # numerical failure → skip

            # if len(parts) != 2:                # should have exactly 2 polygons
            #     continue

            areaL, areaR = parts[0].area, parts[1].area
            delta = abs(areaL - areaR)
            if delta < best_delta:
                best_delta = delta
                best_pair  = (p, q)
                best_split = (parts[0], parts[1])

        if best_split is None:
            raise RuntimeError(f"Piece {self.id}: no valid split produced.")

        # ------------------------------------------------------------------
        # 4.  Wrap the two polygons as new Piece objects
        # ------------------------------------------------------------------
        new_pieces: list["Piece"] = []
        for idx, geom in enumerate(best_split, start=1):

            # local coordinates (0,0) at polygon’s min-x/min-y
            gx_min, gy_min, _, _ = geom.bounds
            local_coords = [(x - gx_min, y - gy_min)
                            for x, y in list(geom.exterior.coords)[:-1]]

            side = "left" if idx == 1 else "right"
            new_id = f"{self.id}_{side}"

            child = Piece(local_coords, id=new_id)
            child.parent_id = self.id
            child.root_id   = getattr(self, "root_id", self.id)

            # keep the absolute placement inherited from this piece
            tx, ty            = self.translation
            child.translation = (tx + gx_min, ty + gy_min)

            # propagate rotation/seam-allowance bookkeeping
            if self.rotation:
                child.rotate(self.rotation)

            # *outer* path = inner + seam allowance (same allowance you used)
            try:
                child.add_seam_allowance()     # falls back to default 1.0 cm
            except Exception:
                # if you sometimes call add_seam_allowance() later, skip here
                pass

            child.update_bbox()
            new_pieces.append(child)

        return tuple(new_pieces)  # (left_piece, right_piece)



class Layout:

    """
    A class representing a layout.
    Each layout has a list of pieces and defines an insertion order.
    """

    def __init__(self, polygon_paths: dict[str, Piece], translations: Dict[str, Tuple[float, float]] = None):
        self.order : OrderedDict[str, Piece] = polygon_paths

        # the tallest piece in the layout
        #self.max_height = max(piece.height for piece in polygon_paths.values())

        if translations:
            for piece_id, translation in translations.items():
                if piece_id in self.order:
                    self.order[piece_id].translation = translation
                else:
                    raise ValueError(f"Translation for piece {piece_id} not found in layout.")
        

class LayoutView:
    """Expose .order for decoders without copying geometry or mutating state."""
    def __init__(self, pieces: list[Piece]):
        tmp = OrderedDict((p.id, p) for p in pieces)
        self.order = MappingProxyType(tmp)
class Container:
    """
    A class representing a rectangular container.
    """

    def __init__ (self, width, height):
        self.width = width
        self.height = height

    
    def update(self, width: float, height: float):
        self.width = width
        self.height = height


    def inner_fit_rectangle(self,
                            piece:      "Piece"
    ) -> list[tuple[float, float]]:
        """
        Return the clockwise Inner‑Fit Rectangle (IFR) for *piece* with
        its anchor at the piece’s top‑left corner.

        Coordinates are given in container space (positive y downward).
        Empty list ⇒ piece is larger than the container.
        """
        Wc, Hc = self.width, self.height
        Wp = max(x for x, _ in piece.get_outer_path())
        Hp = max(y for _, y in piece.get_outer_path())

        if Wp > Wc or Hp > Hc:           # piece does not fit
            return []

        # TL -> TR -> BR -> BL  (CW with y‑down)
        return [
            (0.0,        0.0),           # top‑left
            (Wc - Wp,    0.0),           # top‑right
            (Wc - Wp, Hc - Hp),          # bottom‑right
            (0.0,     Hc - Hp),          # bottom‑left
        ]

