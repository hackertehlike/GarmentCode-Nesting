

import math

class Piece:
    """
    A class representing a piece in the layout.
    Each piece has an ID, a list of vertices, a rotation angle, and a locked state.
    """

    def __init__ (self, vertices):
        self.id = -1
        self.vertices = vertices
        self.rotation = 0
        self.locked = False

    def rotate (self, angle):
        rad = angle * math.pi / 180
        cos_theta = math.cos(rad)
        sin_theta = math.sin(rad)

        for i in range(len(self.vertices)):
            x = self.vertices[i][0]
            y = self.vertices[i][1]
            self.vertices[i][0] = x * cos_theta - y * sin_theta
            self.vertices[i][1] = x * sin_theta + y * cos_theta
            self.rotation += angle
            self.rotation %= 360

class Layout:

    """
    A class representing a layout.
    Each layout has a list of pieces and defines an insertion order.
    """

    def __init__(self, polygon_paths: dict[str, list[list[float]]]):
        self.order = []
        for name, path in polygon_paths.items():     # preserve current order
            p = Piece(path)
            p.id   = name                            
            self.order.append(p)

        

class Container:
    """
    A class representing a rectangular container.
    """

    def __init__ (self, width, height):
        self.width = width
        self.height = height