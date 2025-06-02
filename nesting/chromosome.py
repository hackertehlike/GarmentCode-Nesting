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
            # Placeholder for future split‐type mutation
            pass

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

    def crossover_ox1_k(self, other: Chromosome) -> Chromosome:
        """
        Order Crossover (OX1) with k randomly chosen segments.
        Copies k disjoint segments from self into child, then fills
        remaining slots in order from other.
        """
        start = time.time()
        assert len(self.genes) == len(other.genes), "Parents must be equal length"
        size = len(self.genes)

        # Determine number of segments (1 .. min(3, size//3))
        max_segments = max(1, min(3, size // 3))
        n_segments = random.randint(1, max_segments)

        # Choose 2*n cut points and group into segments
        cut_points = sorted(random.sample(range(size), 2 * n_segments))
        segments = [
            (a, b) if a <= b else (b, a)
            for a, b in zip(cut_points[::2], cut_points[1::2])
        ]

        # Prepare child skeleton and a set of placed IDs
        child: list[Piece | None] = [None] * size

        # Copy each segment from self into child using a comprehension
        placed_ids = set()
        for seg_start, seg_end in segments:
            chunk = [copy.deepcopy(self.genes[i]) for i in range(seg_start, seg_end + 1)]
            child[seg_start : seg_end + 1] = chunk
            placed_ids.update(p.id for p in chunk)

        # Fill remaining slots from other.genes in order
        other_iter = (g for g in other.genes if g.id not in placed_ids)
        for idx in range(size):
            if child[idx] is None:
                child[idx] = copy.deepcopy(next(other_iter))

        end = time.time()
        if config.LOG_TIME:
            print(f"[Chromosome.crossover_ox1_k] took {end - start:.4f} s, segments={n_segments}")
        return Chromosome(child, self.container)

    def crossover_ox1(self, other: Chromosome) -> Chromosome:
        """
        Standard OX1 crossover:
          1. Pick two cut points c1 < c2 (avoid copying entire sequence).
          2. Copy slice [c1:c2+1] from self into child.
          3. Fill in remaining slots from other (wrapping around) in order.
        """
        start = time.time()
        assert len(self.genes) == len(other.genes), "Parents must be equal length"
        size = len(self.genes)

        # 1. Choose c1, c2 such that we don’t copy the entire chromosome
        c1, c2 = sorted(random.sample(range(size), 2))
        while c1 == 0 and c2 == size - 1:
            c1, c2 = sorted(random.sample(range(size), 2))

        # 2. Build child and copy slice from parent1
        child: list[Piece | None] = [None] * size
        child_slice = [copy.deepcopy(p) for p in self.genes[c1 : c2 + 1]]
        child[c1 : c2 + 1] = child_slice

        # Track which IDs are already placed
        placed_ids = {p.id for p in child_slice}

        # 3. Fill in remaining positions from parent2 (wrapping)
        current_idx = (c2 + 1) % size
        for gene in chain(other.genes[c2 + 1 :], other.genes[: c2 + 1]):
            if gene.id not in placed_ids:
                child[current_idx] = copy.deepcopy(gene)
                placed_ids.add(gene.id)
                current_idx = (current_idx + 1) % size
                if current_idx == c1:
                    # Once we return to index c1, the child is full
                    break

        end = time.time()
        if config.LOG_TIME:
            print(f"[Chromosome.crossover_ox1] took {end - start:.4f} s")
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
