
import math
from typing import Dict, List, Tuple


class Piece:
    """
    A class representing a piece in the layout.
    Each piece has an ID, a list of vertices, a rotation angle, and a locked state.
    """

    def __init__ (self, vertices, id = None):
        self.id = id
        # self.vertices = vertices
        self.rotation = 0 # wrt to the original piece
        self._translation = (0, 0)

        self.locked = False

        self.inner_path : List[Tuple[float, float]] = vertices
        self.outer_path : List[Tuple[float, float]] = vertices # by default, the path has NO seam allowance
        self.scale = 1.0
        #bounding box
        self.update_bbox()

    def update_bbox(self):
        print("Updating bounding box for piece ", self.id)

        # get the min and max x and y coordinates
        xs = [pt[0] for pt in self.outer_path]
        ys = [pt[1] for pt in self.outer_path]
        # print(f"Piece {self.id} vertices: {xs}, {ys}")
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
        """Returns the path of the piece as a list of [x, y] vertices."""
        return self.inner_path
    
    def get_outer_path(self) -> list[list[float]]:
        """Returns the path of the piece as a list of [x, y] vertices."""
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
        """Rotate the piece *in place* by *angle* degrees."""
        rad = math.radians(angle)
        cos_theta, sin_theta = math.cos(rad), math.sin(rad)

        for i, (x, y) in enumerate(self.vertices):
            self.vertices[i][0] = x * cos_theta - y * sin_theta
            self.vertices[i][1] = x * sin_theta + y * cos_theta

        # bookkeeping
        self.rotation += angle
        self.rotation %= 360

        self.update_bbox()

    def translate(self, dx: float, dy: float):
        """Translate the piece *in place* by (dx, dy)."""
        for i, (x, y) in enumerate(self.vertices):
            self.vertices[i][0] = x + dx
            self.vertices[i][1] = y + dy

        # bookkeeping
        self._translation = (self._translation[0] + dx, self._translation[1] + dy)

    @property
    def translation(self) -> Tuple[float, float]:
        """Returns the translation of the piece."""
        return (self._translation[0] * self.scale, self._translation[1] * self.scale)
    
    @translation.setter
    def translation(self, value: Tuple[float, float]):
        """Sets the translation of the piece."""
        self._translation = (value[0] / self.scale, value[1] / self.scale)
        # self.translate(value[0], value[1])



class Layout:

    """
    A class representing a layout.
    Each layout has a list of pieces and defines an insertion order.
    """

    def __init__(self, polygon_paths: dict[str, Piece]):
        self.order = polygon_paths

        # the tallest piece in the layout
        self.max_height = max(piece.height for piece in polygon_paths.values())
        

class Container:
    """
    A class representing a rectangular container.
    """

    def __init__ (self, width, height):
        self.width = width
        self.height = height