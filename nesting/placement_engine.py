from nesting.layout import Layout, Container, Piece
from abc import ABC, abstractmethod

from nesting import utils

class PlacementEngine(ABC):
    """
    Base class for layout placement strategies.
    """

    def __init__(self, layout: Layout, container: Container):
        self.layout = layout
        self.container = container

    def decode(self):
        raise NotImplementedError("Subclasses should implement the decode() method.")


class BottomLeftDecoder(PlacementEngine):
    """
    Places the pieces strictly in the given order with a bottom‑left strategy.
    """

    def __init__(self, layout, container,
                 step = 1.0):
        super().__init__(layout, container)
        self.step = step                              # scan / slide increment
        self.placed: list[tuple[Piece, float, float]] = []  # (piece, x, y)


    def decode(self):
        for piece in self.layout.order:
            dx, dy = self._place_single_piece(piece)
            self.placed.append((piece, dx, dy))

        # return summary
        return [(p.id, dx, dy) for p, dx, dy in self.placed]

    
    def _fits(self, poly, dx, dy):
        """
        Check if the polygon *poly* fits in the container at (dx, dy).
        The polygon is translated by (dx, dy) and checked against the
        container boundaries and all previously placed pieces.
        """
        xs, ys = zip(*utils._translate_polygon(poly, dx, dy))
        if (min(xs) < 0 or max(xs) > self.container.width or
            min(ys) < 0 or max(ys) > self.container.height):
            return False

        for other, ox, oy in self.placed:
            if utils.polygons_overlap(
                    utils._translate_polygon(poly, dx, dy),
                    utils._translate_polygon(other.vertices, ox, oy)):
                return False
        return True

    def _place_single_piece(self, piece):
        """
        Start searching at the **top‑right** corner, scan leftwards and
        downwards until a free cell is found, then slide left and down.
        """
        poly  = piece.vertices
        step  = self.step
        found = False

        xs, ys = zip(*poly)
        poly_w = max(xs) - min(xs)
        poly_h = max(ys) - min(ys)

        # place the piece so its *furthest* x,y lie exactly on the container edge:
        x = self.container.width - poly_w
        y = 0
        # # if not self._fits(poly, x, y):
        # #     raise ValueError(f'Piece {piece.id} cannot be placed at {x},{y}')
        
        # slide left/down from a guaranteed-valid corner
        moved = True
        while moved:
            moved = False
            print(f"Starting position: x={x}, y={y}")
            while x - step >= 0 and self._fits(poly, x - step, y):
                x -= step
                moved = True
                # print(f"Moved left to: x={x}, y={y}")
            while y + step <= self.container.height and self._fits(poly, x, y + step):
                y += step
                moved = True
                # print(f"Moved down to: x={x}, y={y}")

        print(f"Final position for piece {piece.id}: x={x}, y={y}")
        return x, y

class GreedyBLDecoder(BottomLeftDecoder):
    """
    Greedily places pieces in the container sorted by order of area.
    The pieces are placed using BL placement strategy. 
    """

    def __init__(self, layout, container):
        super().__init__(layout, container)

        # sort the pieces by area
        self.layout.order.sort(key=lambda p: utils.polygon_area(p.vertices), reverse=True)
        print (f"Sorted pieces by area: {[p.id for p in self.layout.order]}")
        self.container = container
        self.step = 1.0
        self.placed = []

    def decode(self):
        return super().decode()