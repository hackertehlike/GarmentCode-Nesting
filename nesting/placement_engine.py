from .layout import Layout, Container, Piece
from abc import ABC, abstractmethod
import numpy as np, scipy.spatial as sps

from nesting import utils

class PlacementEngine():
    """
    Base class for layout placement strategies.
    """

    def __init__(self, layout: Layout, container: Container):
        self.layout = layout
        self.container = container
        self.placed: list[Piece] = [] 

    def decode(self):
        pass

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
        start_x = self.container.width  - (max(xs) - min(xs))
        start_y = 0.0
        return start_x, start_y
    
    def usage_BB(self):
        """ Returns the ratio of the used area of the bounding box of the placed pieces
        to the area of the bounding box.

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
        
        # calculate the area total of placed pieces
        total_area = sum([utils.polygon_area(p.get_outer_path()) for p in self.placed])
        # calculate the area of the bounding box
        bounding_box_area = (max_x - min_x) * (max_y - min_y)

        # calculate the ratio of the used area of the bounding box to the area of the bounding box
        ratio = total_area / bounding_box_area
        return ratio


    def rest_length(self):
        """ Returns the length of the rest of the container that is not used by the pieces.
        Call AFTER decode() to get the correct values.
        """
        # the rightmost x coordinate of the bounding box
        flattened_vertices = self._flatten_piece_list()
        x_vals = [v[0] for v in flattened_vertices]
        # y_vals = [v[1] for v in flattened_vertices]
        # min_x = min(x_vals)
        max_x = max(x_vals)

        return self.container.width - max_x
    
    
    def concave_hull_utilization(self):
        
        # get the concave hull of the placed pieces

        # get the vertices of the pieces
        flattened_vertices = self._flatten_piece_list()
        # get the concave hull of the vertices
        dists, _ = sps.cKDTree(flattened_vertices).query(flattened_vertices, k=2)
        d = np.median(dists[:,1])          # median first neighbour
        alpha = (1.5 * d)**2
        # alpha = 100
        concave_hull = utils.concave_hull(flattened_vertices, alpha)
        # get the area of the concave hull
        print("concave hull computed")
        area = utils.polygon_area(concave_hull)
        total_area = sum([utils.polygon_area(p[0].get_outer_path()) for p in self.placed])
       
        return area / total_area



    def gravitate(self, piece, x, y, step=1.0):
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
    
class BottomLeftDecoder(PlacementEngine):
    """
    Places the pieces strictly in the given order with a bottom‑left strategy.
    """

    def __init__(self, layout, container,
                 step = 1.0):
        super().__init__(layout, container)
        # self.placed: list[tuple[Piece, float, float]] = []  # (piece, x, y)

    def decode(self):
        for piece_id, piece in self.layout.order.items():
            x0, y0 = self.anchor(piece)
            dx, dy = self.gravitate(piece, x0, y0)
            # print (f"Piece {piece.id} placed at ({dx}, {dy})")
            self.placed.append((piece))
            piece.translation = (dx, dy)
        return [(p.id, *p.translation) for p in self.placed]
    

class GreedyBLDecoder(BottomLeftDecoder):
    """
    Greedily places pieces in the container sorted by order of area.
    The pieces are placed using BL placement strategy. 
    """

    def __init__(self, layout, container):
        super().__init__(layout, container)

        # sort the pieces by area
        # Create a list of (piece_id, piece) tuples sorted by area
        sorted_pieces = sorted(self.layout.order.items(), key=lambda item: item[1].bbox_area, reverse=True)

        # Update layout.order with the sorted list
        self.layout.order = dict(sorted_pieces)
        # print (f"Sorted pieces by area: {[p.id for p in pieces]}")
        self.container = container
        # self.placed = []

    def decode(self):
        return super().decode()
    
class NFPDecoder(PlacementEngine):

    def __init__(self, layout, container, wall_step=1.0):
        super().__init__(layout, container)
        # self.placed = []  # (piece, x, y)
        self.container = container
        self.wall_step = wall_step
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

        # print (f"Placed pieces: {[p.id for p, _, _ in self.placed]}")
        return [(p.id, *piece.translation) for p in self.placed]
    
    def _find_best_position(self, piece: Piece):

        """
        Find the best position for the piece in the container.
        The piece is placed using the BLF strategy.
        """
        best_x, best_y = None, None

        # get inner fit rectangle
        # for the piece in the container
        # print (f"Piece {piece.id} finding best position")
        # inner_fit = utils.inner_fit_rectangle(self.container, piece)
        inner_fit = Container.inner_fit_rectangle(self.container, piece)
        # print("inner fit rectangle: ", inner_fit)
        if not inner_fit:
            # print(f"Piece {piece.id} has no ifr")
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
                if (inner_fit[0][0] <= x_translated <= inner_fit[1][0] and
                    inner_fit[0][1] <= y_translated <= inner_fit[3][1]):
                    # check if the translated nfp is inside the container
                    if best_x is None or (x_translated < best_x or
                                          (x_translated == best_x and y_translated < best_y)):
                    
                        # check if we are intersecting with other placed pieces
                        if self._fits(piece, x_translated, y_translated):
                            # update the best position
                            best_x = x_translated
                            best_y = y_translated
                            print(f"Piece {piece.id} best position: ({best_x}, {best_y})")

        # if no position was found, return None
        if best_x is None or best_y is None:
            print(f"Piece {piece.id} does not fit in the container")
            return None, None
        
        # gravitate the piece to the bottom left corner
        best_x, best_y = self.gravitate(piece, best_x, best_y)
        
        return best_x, best_y


    def _nfp(self, stationary, moving):
        key = (stationary.id, moving.id)
        if key not in self._nfp_cache:
            self._nfp_cache[key] = utils.no_fit_polygon(stationary.get_outer_path(),
                                                        moving.get_outer_path())
                                                        
        return self._nfp_cache[key]
    

