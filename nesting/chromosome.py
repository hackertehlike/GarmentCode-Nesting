from __future__ import annotations
from collections import OrderedDict
# from functools import cached_property
# from typing import Generic, List, Sequence, TypeVar
from typing import List, Mapping
import random

from .layout import Layout, Piece, Container
from .placement_engine import *
import random
import numpy as np
# from nesting.placement_engine import BottomLeftDecoder

# class Gene:
#     """
#     A class representing a gene
#     """

#     def __init__(self, piece):

#         self.piece = None
#         self.rotation = 0

# T = TypeVar("T", bound=object)          # any hashable, comparable type

# fitness functions
def fitness_usage_bb(chromosome: Chromosome) -> float:
    decoder = BottomLeftDecoder(chromosome, chromosome.container)
    decoder.decode()
    return decoder.usage_BB()

def fitness_concave_hull(chromosome: Chromosome) -> float:
    decoder = BottomLeftDecoder(chromosome, chromosome.container)
    decoder.decode()
    return decoder.concave_hull_utilization()

def fitness_rest_length(chromosome: Chromosome) -> float:
    decoder = BottomLeftDecoder(chromosome, chromosome.container)
    decoder.decode()
    return decoder.rest_length()

# META VARIABLES

FITNESS_METRICS = {
    "usage_bb": fitness_usage_bb,
    "concave_hull": fitness_concave_hull,
    "rest_length": fitness_rest_length,
}

SELECTED_FITNESS_METRIC = "usage_bb"

class Chromosome(Layout):
    """
    A class representing a chromosome
    """

    # def __init__(self, layout: Layout | list[Piece], container: Container):
    #     # Accept either a Layout (use its `.order`) or a plain list of Piece
    #     genes = layout.order if isinstance(layout, Layout) else layout
    #     super().__init__(genes)
    #     self.container = container
    #     self.fitness: float = 0.0


    # def __init__(self, pieces: Layout | list[Piece], container: Container):

    #     # if isinstance(pieces, Layout):
    #     #     super().__init__(pieces.order.values())
    #     # else:
    #     #     super().__init__(pieces)

    #     self.genes: List[Piece] = []
    #     # get the genes from the layout
    #     if isinstance(pieces, Layout):
    #         # if pieces is a Layout, get the order
    #         genes = pieces.order.values()
    #     else:
    #         # if pieces is a list of Piece, use it directly
    #         genes = pieces
    #     # store the genes

    #     self.container = container
    #     self.fitness: float = 0.0

    def __init__(self,
                 pieces: Layout | Mapping[str, Piece] | list[Piece],
                 container: Container) -> None:

        # ── build the mapping that Layout expects ──────────────────
        if isinstance(pieces, Layout):
            mapping = pieces.order                           # already OrderedDict
        elif isinstance(pieces, Mapping):                    # dict / OrderedDict
            mapping = OrderedDict(pieces)
        else:                                                # assume a list
            mapping = OrderedDict((p.id, p) for p in pieces)

        # hand the mapping to Layout
        super().__init__(mapping)

        # ── GA‑specific state ──────────────────────────────────────
        self.genes: list[Piece] = list(self.order.values())  # flat list
        self.container = container
        self.fitness = 0.0
        self.calculate_fitness()                     # initial fitness

    # @cached_property
    # def order(self) -> OrderedDict[str, Piece]:
    #     """OrderedDict keyed by piece.id (built only once, then cached)."""
    #     return OrderedDict((p.id, p) for p in self.genes)        


    def calculate_fitness(self):
        # Get the fitness function from the mapping using the meta variable
        fitness_func = FITNESS_METRICS[SELECTED_FITNESS_METRIC]
        self.fitness = fitness_func(self)

    def mutate(self):
        """
        Perform mutation on the chromosome
        """
        print("!" * 50)
        print("MUTATION OCCURRED")
        print("!" * 50)

        # randomly select a mutation type: either split, rotate, or swap (with different probabilities)
        mutation_type = random.choices(
            ["split", "rotate", "swap"],
            # weights=[0.1, 0.3, 0.6],
            weights = [0, 0.5, 0.5], # for now, only rotate and swap
            k=1
        )[0]
        print(f"Selected mutation type: {mutation_type}")

        if mutation_type == "split":
            # TODO
            # for now, dont do anything
            # later i'll implement a split mutation
            # print("Split mutation selected, but not implemented yet.")
            return
        elif mutation_type == "rotate":
            # Select a random piece and rotate it by 90, 180, or 270 degrees
            piece_index = random.randint(0, len(self.genes) - 1)
            angle = random.choice([90, 180, 270])
            print(f"Rotating piece at index {piece_index} by {angle} degrees.")
            self.genes[piece_index].rotate(angle)
        elif mutation_type == "swap":
            # Swap two random pieces
            index1, index2 = random.sample(range(len(self.genes)), 2)
            print(f"Swapping pieces at indices {index1} and {index2}.")
            print(f"Before swap: {[piece.id for piece in self.genes]}")
            self.genes[index1], self.genes[index2] = self.genes[index2], self.genes[index1]
            print(f"After swap: {[piece.id for piece in self.genes]}")
        
        # if 'order' in self.__dict__:
        #     del self.__dict__['order']

    def crossover_pmx(self, other: "Chromosome") -> "Chromosome":
        """
        Partially‑Mapped Crossover (PMX)
        """

        # ── sanity checks ────────────────────────────────────────────────
        assert len(self.genes) == len(other.genes), "Parents must be equal length"
        size = len(self.genes)

        # ── 1. choose the crossover range ───────────────────────────────
        c1, c2 = sorted(random.sample(range(size), 2))          # inclusive
        while c1 == c2 or (c1 == 0 and c2 == size-1):           # avoid full copy
            c1, c2 = sorted(random.sample(range(size), 2))
        # c1 … c2   will be copied verbatim from parent 1

        # ── 2. start the child with the slice from parent 1 ─────────────
        child: list[Piece | None] = [None] * size
        child[c1 : c2 + 1] = self.genes[c1 : c2 + 1]

        # ── 3. fill the remaining positions from parent 2 ───────────────
        # iterate over indices *outside* the copied slice
        for i in list(range(0, c1)) + list(range(c2 + 1, size)):
            candidate = other.genes[i]

            # if the candidate is already in the copied slice,
            # follow the mapping chain until we find an unused gene
            while candidate in self.genes[c1 : c2 + 1]:
                idx_in_parent1 = self.genes.index(candidate)    # where it came from
                candidate = other.genes[idx_in_parent1]         # mapped partner

            child[i] = candidate

        # ── 4. build and return the child chromosome ────────────────────
        return Chromosome(child, self.container)

    
        # # ── 3. mapping: only   other‑gene → self‑gene   (one direction) ─
        # mapping = {other.genes[i]: self.genes[i] for i in range(c1, c2 + 1)}
        # # Print mapping with piece ids instead of objects
        # mapping_str = {piece.id: mapping[piece].id for piece in mapping}
        # print(f"Mapping created from crossover slice: {mapping_str}")

        # # ── 4. fill the remaining positions from the second parent ────
        # for i in range(size):
        #     if c1 <= i <= c2:
        #         continue

        #     gene = other.genes[i]
        #     visited = set()  # guards against 2‑gene cycles
        #     print(f"Processing gene at index {i}: {gene.id}")

        #     # follow the mapping chain until we hit a gene that is
        #     # *not* yet in the child or the chain runs out
        #     while gene in child and gene in mapping:
        #         if gene in visited:  # 2‑gene or longer cycle
        #             print(f"Cycle detected for gene {gene.id}, finding next unused gene.")
        #             # pick the next unused gene from the first parent
        #             gene = next(g for g in self.genes if g not in child)
        #             break
        #         visited.add(gene)
        #         print(f"Following mapping for gene {gene.id}.")
        #         gene = mapping[gene]

        #     child[i] = gene
        #     print(f"Child after placing gene at index {i}: {[p.id if p is not None else None for p in child]}")

        # print(f"Final child genes: {[p.id for p in child if p is not None]}")
        return Chromosome(child, self.container)


# if __name__ == "__main__":

    # # small test
    # p0 = GenericChromosome([1, 2, 3, 4, 5, 6, 7, 8])
    # p1 = GenericChromosome([3, 7, 5, 1, 6, 8, 2, 4])

    # child = p0.crossover_pmx(p1)
    # print("Parent 0:", p0.genes)
    # print("Parent 1:", p1.genes)
    # print("Child   :", child.genes)

# class GenericChromosome(Generic[T]):
#     """
#     A chromosome that stores an ordered list of genes of *any* type.
#     Sub‑classes inherit the PMX operator and may add fitness logic, etc.
#     """

#     def __init__(self, genes: Sequence[T]):
#         # store a *copy* so we never mutate the original list from outside
#         self.genes: List[T] = list(genes)

#     # ————————————————————————————— PMX crossover ————————————————————————————— #

#     def crossover_pmx(self, other: "GenericChromosome[T]") -> "GenericChromosome[T]":
#         """
#         Partially‑Mapped Crossover (PMX) — independent of the gene type.
#         Produces a child of the *same* concrete class as the parent.
#         """
#         assert len(self.genes) == len(other.genes), "Parents must be equal length"
#         size = len(self.genes)

#         # choose the crossover range
#         c1, c2 = sorted(random.sample(range(size), 2))
#         print(f"c1: {c1}, c2: {c2}")

#         # initialize the child with None
#         child: List[T | None] = [None] * size
#         child[c1 : c2 + 1] = self.genes[c1 : c2 + 1]            # copy slice from self

#         # build the mapping induced by the selected slice
#         mapping: dict[T, T] = {}
#         for i in range(c1, c2 + 1):
#             a, b = other.genes[i], self.genes[i]
#             mapping[a] = b   # other → self
#             mapping[b] = a   # self  → other

#         # fill the remaining positions
#         for i in range(size):
#             if c1 <= i <= c2:
#                 continue

#             gene = other.genes[i]
#             # if it's already in the child, follow the mapping until we find a gene that isn't
#             while gene in child:
#                 gene = mapping[gene]
#             child[i] = gene


#         # return a new instance of *this* class,
#         # so sub‑classes keep their specialised behaviour
#         return self.__class__(child)

#     # def __repr__(self) -> str:
#     #     return f"{self.__class__.__name__}({self.genes!r})"
#
