from nesting.layout import Layout, Container, Piece
from abc import ABC, abstractmethod

from nesting import utils

class PlacementEngine():
    """
    Base class for layout placement strategies.
    """

    def __init__(self, layout: Layout, container: Container):
        self.layout = layout
        self.container = container

    def decode(self):
        pass

    def _fits(self, polygon, dx: float, dy: float) -> bool:
        """
        Check if the polygon fits in the container at (dx, dy).
        The polygon is translated by (dx, dy) and checked against the
        container boundaries and all previously placed pieces.
        """

        translated = utils._translate_polygon(polygon, dx, dy)

        # inside container?
        xs, ys = zip(*translated)
        if (min(xs) < 0 or max(xs) > self.container.width or
            min(ys) < 0 or max(ys) > self.container.height):
            return False

        # no overlap with already fixed parts?
        for other, ox, oy in self.placed:
            if utils.polygons_overlap(
                    translated,
                    utils._translate_polygon(other.vertices, ox, oy)):
                return False
        return True
    

    def anchor(self, piece):
        """Default: push piece against the container’s top-right corner."""
        xs, ys  = zip(*piece.vertices)
        start_x = self.container.width  - (max(xs) - min(xs))
        start_y = 0.0
        return start_x, start_y
    
    def usage_BB(self):
        """ Returns the ratio of the used area of the bounding box of the placed pieces
        to the area of the bounding box.

        Call AFTER decode() to get the correct values.
        """
        flattened_vertices = [(v[0]+ p[1], v[1]+ p[2]) for p in self.placed for v in p[0].vertices]
        print (f"flattened vertices: {flattened_vertices}")
        x_vals = [v[0] for v in flattened_vertices]
        y_vals = [v[1] for v in flattened_vertices]
        min_x = min(x_vals)
        max_x = max(x_vals)
        min_y = min(y_vals)
        max_y = max(y_vals)
        # get the bounding box of the placed pieces
        # min_x = min([v[0] for v in flattened_vertices])#min([p[1] for p in self.placed])
        # max_x = max([p[1] + p[0].max_x for p in self.placed])
        # min_y = min([p[2] for p in self.placed])
        # max_y = max([p[2] + p[0].max_y for p in self.placed])

        # calculate the area total of placed pieces
        total_area = sum([utils.polygon_area(p[0].vertices) for p in self.placed])
        # calculate the area of the bounding box
        bounding_box_area = (max_x - min_x) * (max_y - min_y)

        # calculate the ratio of the used area of the bounding box to the area of the bounding box
        ratio = total_area / bounding_box_area
        return ratio



class BottomLeftDecoder(PlacementEngine):
    """
    Places the pieces strictly in the given order with a bottom‑left strategy.
    """

    def __init__(self, layout, container,
                 step = 1.0):
        super().__init__(layout, container)
        self.step = step                              # scan / slide increment
        self.placed: list[tuple[Piece, float, float]] = []  # (piece, x, y)


    def gravitate(self, piece, x, y):
        """Slide left as far as possible, then down; repeat until jammed."""
        moved = True
        while moved:
            moved = False
            while x - self.step >= 0 and self._fits(piece.vertices, x - self.step, y):
                x -= self.step
                moved = True
            while y + self.step <= self.container.height and self._fits(piece.vertices, x, y + self.step):
                y += self.step
                moved = True
        return x, y

    def decode(self):
        for piece in self.layout.order:
            x0, y0 = self.anchor(piece)
            dx, dy = self.gravitate(piece, x0, y0)
            self.placed.append((piece, dx, dy))
        return [(p.id, dx, dy) for p, dx, dy in self.placed]
    

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
    
class NFPDecoder(PlacementEngine):

    def __init__(self, layout, container, wall_step=1.0):
        super().__init__(layout, container)
        self.placed = []  # (piece, x, y)
        self.container = container
        self.wall_step = wall_step
        self._nfp_cache = {}

    def decode(self):
       # go in layout order
        for piece in self.layout.order:
            # find the best position for the piece
            print(f"Placing piece {piece.id}...")
            best_x, best_y = self._find_best_position(piece)
            print(f"Placing piece {piece.id} at ({best_x}, {best_y})")
            self.placed.append((piece, best_x, best_y))
        return [(p.id, dx, dy) for p, dx, dy in self.placed]
    
    def _find_best_position(self, piece):
        """
        Find the best position for the piece in the container.
        The piece is placed using the BLF strategy.
        """
        best_x, best_y = None, None

        # get inner fit rectangle
        # for the piece in the container
        inner_fit = utils.inner_fit_rectangle(self.container, piece)
        if not inner_fit:
            print(f"Piece {piece.id} has no ifr")
            return None, None

        # iterate over nfps all previously placed pieces
        # choose the bottom leftest point out of all vertices that
        # are inside the inner fit rectangle and on one of the nfp edges
        # note: y grows down, x grows right

        # if first piece, place it at the bottom left corner
        if not self.placed:
            best_x = 0
            best_y = self.container.height - piece.max_y
            return best_x, best_y

        for other, ox, oy in self.placed:
            print(f"Checking piece {piece.id} with {other.id}")
            nfp = self._nfp(other, piece)
            # print(f"Piece {piece.id} nfp with {other.id}: {nfp}")
            for x, y in nfp:
                # translate the nfp wrt the stationary piece
                x_translated = x + ox
                y_translated = y + oy
                # check if the translated nfp is inside the inner fit rectangle
                if (inner_fit[0][0] <= x_translated <= inner_fit[1][0] and
                    inner_fit[0][1] <= y_translated <= inner_fit[3][1]):
                        # check if the translated nfp is inside the container
                        if best_x is None or (x_translated < best_x or
                                              (x_translated == best_x and y_translated < best_y)):
                        
                            # check if we are intersecting with other placed pieces
                            if self._fits(piece.vertices, x_translated, y_translated):
                                # update the best position
                                best_x = x_translated
                                best_y = y_translated

        # if no position was found, return None
        if best_x is None or best_y is None:
            print(f"Piece {piece.id} does not fit in the container")
            return None, None
        
        return best_x, best_y


    def _nfp(self, stationary, moving):
        key = (stationary.id, moving.id)
        if key not in self._nfp_cache:
            self._nfp_cache[key] = utils.no_fit_polygon(stationary.vertices,
                                                        moving.vertices)
                                                        
        return self._nfp_cache[key]