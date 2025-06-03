# chromosome.py

from __future__ import annotations
from collections import OrderedDict
from itertools import chain
import time
import random
import copy
from typing import Callable

from .layout import Piece, Container, Layout, LayoutView
from .placement_engine import DECODER_REGISTRY
import nesting.config as config

# ── Metric Registration ─────────────────────────────────────────────────────────

METRIC_REGISTRY: dict[str, Callable] = {}

def register_metric(name: str):
    def deco(fn):
        METRIC_REGISTRY[name] = fn
        return fn
    return deco

def _run_decoder(chromosome: Chromosome, decoder_name: str):
    view = LayoutView(chromosome.genes)
    Decoder = DECODER_REGISTRY[decoder_name]
    decoder = Decoder(view, chromosome.container)
    decoder.decode()
    return decoder

@register_metric("usage_bb")
def fitness_usage_bb(chromosome: Chromosome, decoder: str):
    dec = _run_decoder(chromosome, decoder)
    return dec.usage_BB()

@register_metric("concave_hull")
def fitness_concave_hull(chromosome: Chromosome, decoder: str):
    dec = _run_decoder(chromosome, decoder)
    return dec.concave_hull_utilization()

@register_metric("rest_length")
def fitness_rest_length(chromosome: Chromosome, decoder: str):
    dec = _run_decoder(chromosome, decoder)
    return dec.rest_length()


# ── Chromosome Definition ───────────────────────────────────────────────────────

class Chromosome(Layout):

    def __init__(
        self,
        pieces: list[Piece],
        container: Container,
        origin: str = "random"
    ):
        # Store a deep copy of each piece
        self._genes = [copy.deepcopy(p) for p in pieces]
        self.container = container
        self.fitness: float | None = None

        # Track origin and last mutation type
        self.origin: str | None = origin
        self.last_mutation: str | None = None

    @property
    def genes(self) -> list[Piece]:
        return self._genes

    def calculate_fitness(self) -> None:
        """Compute fitness via the registered metric and decoder."""
        metric_fn = METRIC_REGISTRY[config.SELECTED_FITNESS_METRIC]
        self.fitness = metric_fn(self, config.SELECTED_DECODER)

    def mutate(self) -> Chromosome:
        """
        Perform a single mutation on the chromosome. Mutation types and
        their probabilities are read from config.MUTATION_WEIGHTS.
        """
        start = time.time()

        # Choose mutation type according to weights
        mutation_types, weights = zip(*config.MUTATION_WEIGHTS.items())
        mutation = random.choices(mutation_types, weights=weights, k=1)[0]
        self.last_mutation = mutation

        n = len(self.genes)

        if mutation == "split":
            # Split a random gene into two new genes
            idx = random.randrange(n)
            piece = self.genes[idx]
            new_genes = piece.split()
            self.genes.pop(idx)
            self.genes[idx:idx] = new_genes

        elif mutation == "rotate":
            # Pick how many genes to rotate (1..n), then apply in batch
            num_genes = random.randint(1, n)
            # Pre‐generate (idx, angle) pairs via comprehension
            indices_and_angles = [
                (random.randrange(n), random.choice(config.ALLOWED_ROTATIONS))
                for _ in range(num_genes)
            ]
            for idx, angle in indices_and_angles:
                self.genes[idx].rotate(angle)

        elif mutation == "swap":
            i, j = random.sample(range(n), 2)
            self.genes[i], self.genes[j] = self.genes[j], self.genes[i]

        elif mutation == "inversion":
            i, j = sorted(random.sample(range(n), 2))
            # Reverse the slice [i:j+1] in one step
            self.genes[i : j + 1] = self.genes[i : j + 1][::-1]

        elif mutation == "insertion":
            i = random.randrange(n)
            gene = self.genes.pop(i)
            j = random.randrange(n)
            self.genes.insert(j, gene)

        elif mutation == "scramble":
            i, j = sorted(random.sample(range(n), 2))
            sub = self.genes[i : j + 1]
            random.shuffle(sub)
            self.genes[i : j + 1] = sub

        else:
            raise ValueError(f"Unknown mutation type: {mutation}")

        end = time.time()
        if config.LOG_TIME:
            print(f"[Chromosome.mutate] '{mutation}' took {end - start:.4f} s")
        return self

    def crossover_ox1(self, other: Chromosome, k: int = 1) -> Chromosome:
        """
        Generalized OX1 crossover with k segments.
        1. Pick 2*k cut points.
        2. Copy each of the k segments from self into child.
        3. Fill remaining slots from other in order.
        """
        start = time.time()
        assert len(self.genes) == len(other.genes), "Parents must be equal length"
        size = len(self.genes)

        # Choose 2*k cut points and build segments
        cut_points = sorted(random.sample(range(size), 2 * k))
        segments = [(a, b) if a <= b else (b, a) for a, b in zip(cut_points[::2], cut_points[1::2])]

        # Prepare child skeleton and track placed IDs
        child: list[Piece | None] = [None] * size
        placed_ids: set[str] = set()
        # Copy segments from parent1
        for seg_start, seg_end in segments:
            chunk = [copy.deepcopy(self.genes[i]) for i in range(seg_start, seg_end + 1)]
            child[seg_start : seg_end + 1] = chunk
            placed_ids.update(p.id for p in chunk)

        # Fill in remaining positions from parent2
        for idx in range(size):
            if child[idx] is None:
                for g in other.genes:
                    if g.id not in placed_ids:
                        child[idx] = copy.deepcopy(g)
                        placed_ids.add(g.id)
                        break

        end = time.time()
        if config.LOG_TIME:
            print(f"[Chromosome.crossover_ox1] k={k} took {end - start:.4f} s, segments={segments}")
        return Chromosome(child, self.container)

    def crossover_pmx(self, other: Chromosome) -> Chromosome:
        """
        Partially Mapped Crossover (PMX). For each position outside the chosen
        window [c1:c2], map according to the slice from self, resolving conflicts.
        """
        start = time.time()
        assert len(self.genes) == len(other.genes), "Parents must be equal length"
        size = len(self.genes)

        # 1. Choose c1, c2 (avoid trivial full‐copy)
        c1, c2 = sorted(random.sample(range(size), 2))
        while c1 == c2 or (c1 == 0 and c2 == size - 1):
            c1, c2 = sorted(random.sample(range(size), 2))

        # 2. Initialize child array and copy the middle slice from parent1
        child: list[Piece | None] = [None] * size
        child[c1 : c2 + 1] = self.genes[c1 : c2 + 1]

        # Prepare lookups for piece IDs
        slice_ids = {p.id for p in child[c1 : c2 + 1]}
        id_to_index_self = {p.id: idx for idx, p in enumerate(self.genes)}

        # 3. Fill remaining slots from parent2, resolving conflicts
        for i in chain(range(0, c1), range(c2 + 1, size)):
            candidate = other.genes[i]
            # Follow mapping until we find an ID not in slice_ids
            while candidate.id in slice_ids:
                idx_in_parent1 = id_to_index_self[candidate.id]
                candidate = other.genes[idx_in_parent1]
            child[i] = copy.deepcopy(candidate)

        end = time.time()
        if config.LOG_TIME:
            print(f"[Chromosome.crossover_pmx] took {end - start:.4f} s")
        return Chromosome(child, self.container)

    # def sync_order(self) -> None:
    #     """Sync the order dict to reflect the current gene sequence."""
    #     self.order = OrderedDict((p.id, p) for p in self.genes)

    def _signature(self) -> tuple[tuple[str, int], ...]:
        """An immutable fingerprint: ((id, rotation), …) in gene order."""
        return tuple((p.id, p.rotation) for p in self.genes)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Chromosome):
            return NotImplemented
        return self._signature() == other._signature()

    def __hash__(self) -> int:
        return hash(self._signature())

    def __repr__(self) -> str:
        return f"Chromosome({self._signature()})"
