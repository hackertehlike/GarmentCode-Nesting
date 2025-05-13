from __future__ import annotations
from collections import OrderedDict
# from functools import cached_property
# from typing import Generic, List, Sequence, TypeVar
from itertools import chain
from typing import List, Mapping
import random
from itertools import chain
import copy


from .layout import Layout, Piece, Container
from .placement_engine import *
import random
import numpy as np
import copy
# from nesting.placement_engine import BottomLeftDecoder

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

    def __init__(self,
                 pieces: Layout | Mapping[str, Piece] | list[Piece],
                 container: Container) -> None:

        # ── build the mapping that Layout expects ──────────────────
        if isinstance(pieces, Layout):
            mapping = OrderedDict((pid, copy.deepcopy(p)) for pid, p in pieces.order.items())
        elif isinstance(pieces, Mapping):
            mapping = OrderedDict((pid, copy.deepcopy(p)) for pid, p in pieces.items())
        else:
            mapping = OrderedDict((p.id, copy.deepcopy(p)) for p in pieces)

        # hand the mapping to Layout
        super().__init__(mapping)

        # ── GA‑specific state ──────────────────────────────────────
        self.genes: list[Piece] = list(self.order.values())  # flat list
        self.container = container
        self.fitness = 0.0
        self.calculate_fitness()                     # initial fitness  


    def calculate_fitness(self):
        # Get the fitness function from the mapping using the meta variable
        fitness_func = FITNESS_METRICS[SELECTED_FITNESS_METRIC]
        self.fitness = fitness_func(self)

    def mutate(self):
        # TODO: add more mutation types
        """
        Perform mutation on the chromosome
        """
        print()
        print("!" * 50)
        print("MUTATION OCCURRED")
        print("!" * 50)
        print()
        
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
            self.sync_order()  # sync the order
        elif mutation_type == "swap":
            # Swap two random pieces
            index1, index2 = random.sample(range(len(self.genes)), 2)
            print(f"Swapping pieces at indices {index1} and {index2}.")
            print(f"Before swap: {[piece.id for piece in self.genes]}")
            self.genes[index1], self.genes[index2] = self.genes[index2], self.genes[index1]
            self.sync_order()  # sync the order
            print(f"After swap: {[piece.id for piece in self.genes]}")

    def crossover_ox1(self, other: "Chromosome") -> "Chromosome":
        """Order Crossover (OX1) with *k* randomly chosen segments.

        This generalises the classical 2‑point OX1 by allowing an arbitrary
        (random) number of *non‑overlapping* segments to be copied from the
        first parent (``self``) to the child.  The missing genes are then
        inserted in their relative order taken from the second parent
        (``other``), exactly as illustrated in the reference screenshot.
        """
        assert len(self.genes) == len(other.genes), "Parents must be equal length"
        size = len(self.genes)

        # 1) Decide how many segments to copy (at least 1, but never all genes)
        max_segments_reasonable = max(1, min(3, size // 3))  # conservative upper bound
        n_segments = random.randint(1, max_segments_reasonable)

        # 2) Pick 2·n unique cut points and turn them into ordered pairs (segments)
        cut_points = sorted(random.sample(range(size), 2 * n_segments))
        segments: list[tuple[int, int]] = []
        for a, b in zip(cut_points[::2], cut_points[1::2]):
            start, end = (a, b) if a <= b else (b, a)
            segments.append((start, end))

        # 3) Build the child and copy slices from *self*
        child: List[Piece | None] = [None] * size
        placed_ids: set[str] = set()
        for start, end in segments:
            for idx in range(start, end + 1):
                gene = copy.deepcopy(self.genes[idx])
                child[idx] = gene
                placed_ids.add(gene.id)

        # 4) Fill the remaining slots with genes from *other* in order
        other_iter = (g for g in other.genes if g.id not in placed_ids)
        for idx in range(size):
            if child[idx] is None:
                child[idx] = copy.deepcopy(next(other_iter))

        return Chromosome(child, self.container)
        
    def crossover_ox1_k(self, other: "Chromosome") -> "Chromosome":
        """Order Crossover (OX1) with *k* randomly chosen segments.

        Allows an arbitrary (random) number of non‑overlapping segments to be copied from the
        first parent (self) to the child.  The missing genes are then
        inserted in their relative order taken from the other parent
        """
        assert len(self.genes) == len(other.genes), "Parents must be equal length"
        size = len(self.genes)

        # 1) Decide how many segments to copy (at least 1, but never all genes)
        max_segments_reasonable = max(1, min(3, size // 3))  # conservative upper bound
        n_segments = random.randint(1, max_segments_reasonable)

        # 2) Pick 2·n unique cut points and turn them into ordered pairs (segments)
        cut_points = sorted(random.sample(range(size), 2 * n_segments))
        segments: list[tuple[int, int]] = []
        for a, b in zip(cut_points[::2], cut_points[1::2]):
            start, end = (a, b) if a <= b else (b, a)
            segments.append((start, end))

        # 3) Build the child and copy slices from *self*
        child: List[Piece | None] = [None] * size
        placed_ids: set[str] = set()
        for start, end in segments:
            for idx in range(start, end + 1):
                gene = copy.deepcopy(self.genes[idx])
                child[idx] = gene
                placed_ids.add(gene.id)

        # 4) Fill the remaining slots with genes from *other* in order
        other_iter = (g for g in other.genes if g.id not in placed_ids)
        for idx in range(size):
            if child[idx] is None:
                child[idx] = copy.deepcopy(next(other_iter))

        return Chromosome(child, self.container)
    
    def crossover_ox1(self, other: "Chromosome") -> "Chromosome":
        """Order Crossover (OX1).

        Algorithm steps:
        1. Choose two random crossover points (c1, c2).
        2. Copy the slice between c1 and c2 from *this* parent into the child.
        3. Starting from the position after c2 in *other* parent, copy genes in order
           that are not yet present in the child, wrapping around until the child is full.
        """
        assert len(self.genes) == len(other.genes), "Parents must be equal length"
        size = len(self.genes)

        # 1. choose crossover points ensuring we do not copy the whole chromosome
        c1, c2 = sorted(random.sample(range(size), 2))
        while c1 == 0 and c2 == size - 1:  # avoid trivial copy
            c1, c2 = sorted(random.sample(range(size), 2))

        # 2. create child and insert slice from parent 1 (self)
        child: list[Piece | None] = [None] * size
        child[c1 : c2 + 1] = [copy.deepcopy(p) for p in self.genes[c1 : c2 + 1]]

        # helper: set of ids already placed in child
        placed_ids = {p.id for p in child[c1 : c2 + 1] if p is not None}

        # 3. fill remaining positions with genes from parent 2 (other)
        current_idx = (c2 + 1) % size  # first position to fill
        for gene in chain(other.genes[c2 + 1 :], other.genes[: c2 + 1]):
            if gene.id in placed_ids:
                continue  # skip duplicates
            child[current_idx] = copy.deepcopy(gene)
            placed_ids.add(gene.id)
            current_idx = (current_idx + 1) % size

        # 4. return new chromosome instance
        return Chromosome(child, self.container)

    def crossover_pmx(self, other: "Chromosome") -> "Chromosome":
        """Partially‑Mapped Crossover (PMX) that matches pieces by *id*."""
        assert len(self.genes) == len(other.genes), "Parents must be equal length"
        size = len(self.genes)

        # 1. crossover window (avoid copying the whole chromosome)
        c1, c2 = sorted(random.sample(range(size), 2))
        while c1 == c2 or (c1 == 0 and c2 == size - 1):
            c1, c2 = sorted(random.sample(range(size), 2))

        # 2. start the child with the slice from parent 1
        child: list[Piece | None] = [None] * size
        child[c1 : c2 + 1] = self.genes[c1 : c2 + 1]

        # helper structures for ID‑based look‑ups
        slice_ids        = {p.id for p in child[c1 : c2 + 1]}
        id_to_index_self = {p.id: idx for idx, p in enumerate(self.genes)}

        # 3. fill the remaining positions from parent 2
        for i in chain(range(0, c1), range(c2 + 1, size)):
            candidate = other.genes[i]

            # follow the mapping chain until we find a piece *not* in the slice
            while candidate.id in slice_ids:
                idx_in_parent1 = id_to_index_self[candidate.id]
                candidate      = other.genes[idx_in_parent1]

            child[i] = copy.deepcopy(candidate)      # independent gene

        # 4. build and return the child chromosome
        return Chromosome(child, self.container)
    
    def sync_order(self) -> None:
        """Sync the order of the genes with the order dict."""
        self.order = OrderedDict((p.id, p) for p in self.genes)
    
    def _signature(self) -> tuple:
        """Immutable fingerprint: ((id, rotation), …) in gene order."""
        return tuple((p.id, p.rotation) for p in self.genes)

    # equality by value
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Chromosome):
            return NotImplemented
        return self._signature() == other._signature()

    # hash to allow membership tests in sets / dict keys
    def __hash__(self) -> int:
        return hash(self._signature())
    
    def __repr__(self) -> str:
        return f"Chromosome({self._signature()})"