from __future__ import annotations
from collections import OrderedDict
# from functools import cached_property
# from typing import Generic, List, Sequence, TypeVar
from itertools import chain
import time
from typing import Callable, List
import random
from itertools import chain
import copy


from .layout import *
from .placement_engine import *
import nesting.config as config
# from .config import SELECTED_FITNESS_METRIC  # use selected metric from central config
import random
import copy

from .placement_engine import DECODER_REGISTRY

METRIC_REGISTRY: dict[str, Callable] = {}

def register_metric(name: str):
    def deco(fn):
        METRIC_REGISTRY[name] = fn
        return fn
    return deco

def _run_decoder(chromosome, decoder_name: str):
    view = LayoutView(chromosome.genes)
    Decoder = DECODER_REGISTRY[decoder_name]
    decoder = Decoder(view, chromosome.container)
    decoder.decode()
    return decoder

@register_metric("usage_bb")
def fitness_usage_bb(chromosome, decoder: str):
    dec = _run_decoder(chromosome, decoder)
    return dec.usage_BB()

@register_metric("concave_hull")
def fitness_concave_hull(chromosome, decoder: str):
    dec = _run_decoder(chromosome, decoder)
    return dec.concave_hull_utilization()

@register_metric("rest_length")
def fitness_rest_length(chromosome, decoder: str):
    dec = _run_decoder(chromosome, decoder)
    return dec.rest_length()
class Chromosome(Layout):
    
    def __init__(self, pieces: list[Piece], container: Container):
        # deep-copy once, store as list
        self._genes = [copy.deepcopy(p) for p in pieces]
        self.container = container
        self.fitness = None
        #self.calculate_fitness()

    @property
    def genes(self) -> list[Piece]:
        return self._genes


    def calculate_fitness(self):
        metric_fn = METRIC_REGISTRY[config.SELECTED_FITNESS_METRIC]
        # pass in the selected decoder name, too
        self.fitness = metric_fn(self, config.SELECTED_DECODER)

    def mutate(self):
        # TODO: add more mutation types
        """
        Perform mutation on the chromosome
        """
        start = time.time()
        print()
        print("!" * 50)
        print("MUTATION OCCURRED")
        print("!" * 50)
        print()
        
        # randomly select a mutation type: either split, rotate, or swap (with different probabilities)
        mutation_types, weights = zip(*config.MUTATION_WEIGHTS.items())

        mutation = random.choices(mutation_types, weights=weights, k=1)[0]
        print(f"Selected mutation type: {mutation}")

        if mutation == "split":
            # TODO
            # for now, dont do anything
            # later i'll implement a split mutation
            # print("Split mutation selected, but not implemented yet.")
            return
        elif mutation == "rotate":
            # Select a random piece and rotate it by 90, 180, or 270 degrees
            piece_index = random.randint(0, len(self.genes) - 1)
            angle = random.choice([90, 180, 270])
            print(f"Rotating piece at index {piece_index} by {angle} degrees.")
            self.genes[piece_index].rotate(angle)
            #self.sync_order()  # sync the order
        elif mutation == "swap":
            # Swap two random pieces
            index1, index2 = random.sample(range(len(self.genes)), 2)
            print(f"Swapping pieces at indices {index1} and {index2}.")
            print(f"Before swap: {[piece.id for piece in self.genes]}")
            #self.genes[index1], self.genes[index2] = self.genes[index2], self.genes[index1]
            # self.sync_order()  # sync the order
            genes = self.genes
            genes[index1], genes[index2] = genes[index2], genes[index1]
            return Chromosome(genes, self.container)

            print(f"After swap: {[piece.id for piece in self.genes]}")
        #self.sync_order()  # ensure the order is updated after mutation
        end = time.time()
        if config.LOG_TIME:
            print(f"Mutation took {end - start:.4f} seconds")

        
    def crossover_ox1_k(self, other: "Chromosome") -> "Chromosome":
        """Order Crossover (OX1) with *k* randomly chosen segments.

        Allows an arbitrary (random) number of non‑overlapping segments to be copied from the
        first parent (self) to the child.  The missing genes are then
        inserted in their relative order taken from the other parent
        """
        start = time.time()
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

        end = time.time()
        
        if config.LOG_TIME:
            print(f"OX1-k crossover took {end - start:.4f} seconds with {n_segments} segments")

        return Chromosome(child, self.container)
    
    def crossover_ox1(self, other: "Chromosome") -> "Chromosome":
        """Order Crossover (OX1).

        Algorithm steps:
        1. Choose two random crossover points (c1, c2).
        2. Copy the slice between c1 and c2 from *this* parent into the child.
        3. Starting from the position after c2 in *other* parent, copy genes in order
           that are not yet present in the child, wrapping around until the child is full.
        """

        start = time.time()

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

        end = time.time()
        
        if config.LOG_TIME:
            print(f"OX1 crossover took {end - start:.4f} seconds")

        # 4. return new chromosome instance
        return Chromosome(child, self.container)

    def crossover_pmx(self, other: "Chromosome") -> "Chromosome":
        """Partially‑Mapped Crossover (PMX) that matches pieces by *id*."""
        start = time.time()
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

        end = time.time()
        
        if config.LOG_TIME:
            print(f"PMX crossover took {end - start:.4f} seconds")

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