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

    def split(self) -> Tuple["Piece", "Piece"]:
            """
            Split this piece into two new Piece objects by a vertical line through the
            middle of its bounding box. This version handles cases where the inner
            polygon may break into more than two fragments. We group those fragments
            into “left” vs. “right” halves by centroid-x, then union them back.
            """
            # Build Shapely geometry for inner polygon only
            poly_inner = Polygon(self.inner_path)

            # Compute vertical split-line at mid-x of inner bounding box
            minx, miny, maxx, maxy = poly_inner.bounds
            midx = (minx + maxx) / 2.0
            split_line = LineString([(midx, miny - 1.0), (midx, maxy + 1.0)])

            # Split only the inner polygon
            result_inner = shapely_split(poly_inner, split_line)
            

            parts_inner: list[Polygon] = [g for g in result_inner.geoms if isinstance(g, Polygon)]

            # Group fragments by centroid-x into left and right
            left_list = [g for g in parts_inner if g.centroid.x <= midx]
            right_list = [g for g in parts_inner if g.centroid.x > midx]
            if not left_list:
                left_list = parts_inner[:1]
            if not right_list:
                right_list = parts_inner[-1:]

            # Union fragments into single polygon per side
            def union_to_single_polygon(fragments: list[Polygon]) -> Polygon:
                combined = unary_union(fragments)
                if isinstance(combined, Polygon):
                    return combined
                if isinstance(combined, MultiPolygon):
                    return max(combined.geoms, key=lambda p: p.area)
                raise ValueError(f"Unexpected geometry type after union: {type(combined)}")

            inner_left = union_to_single_polygon(left_list)
            inner_right = union_to_single_polygon(right_list)

            # Create new Piece instances from inner paths and apply seam allowances
            new_pieces: list[Piece] = []
            for idx, inner_geom in enumerate((inner_left, inner_right), start=1):
                # Determine local origin from inner geometry bounding box
                ox_min, oy_min, _, _ = inner_geom.bounds
                # Convert inner_geom exterior to local coordinates
                i_coords_local = [(x - ox_min, y - oy_min) for x, y in inner_geom.exterior.coords]
                new_id = f"{self.id}_s{idx}"
                new_piece = Piece(i_coords_local, id=new_id)
                new_piece.parent_id = self.id
                new_piece.root_id   = getattr(self, "root_id", self.id)
                # Set translation and rotation
                tx, ty = self.translation
                new_piece.translation = (tx + ox_min, ty + oy_min)
                new_piece.rotation    = self.rotation
                # Apply seam allowance to generate outer_path and update bbox
                new_piece.add_seam_allowance()
                new_piece.update_bbox()
                new_pieces.append(new_piece)

            return new_pieces[0], new_pieces[1]


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

