import math
from types import MappingProxyType
from typing import Dict, Tuple
from nesting import utils

from collections import OrderedDict
class Piece:
    """
    A class representing a piece in the layout.
    Each piece has an ID, a list of outer_path, a rotation angle, and a locked state.
    """

    def __init__(self, original_path, id=None):
        self.id = id

        self.inner_path = original_path     # original inner geometry (in cm)
        self.outer_path = original_path.copy()     # original outer geometry (in cm)
        
        #self.add_seam_allowance()  # original outer geometry (in cm)
        # self.translation = (0.0, 0.0)
        self.rotation = 0 # wrt to the original piece
        self._translation = (0, 0)
        self.locked = False
        #self.scale = 1.0

        #bounding box
        #self.update_bbox()

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
        self.update_bbox()
        # shift
        # utils.shift_coordinates(self.outer_path)
        self.outer_path = utils.shift_coordinates(self.outer_path)
        self.inner_path = utils.shift_coordinates(self.inner_path)

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

        print(f"Adding seam allowance of {allowance} to piece {self.id}")
        new_outer = utils.compute_offset_path(contour, allowance)
        self.outer_path = new_outer
        self.update_bbox()
        print(f"Piece {self.id} outer path updated with seam allowance")

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

