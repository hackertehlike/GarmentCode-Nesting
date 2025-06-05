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
            # Split up to config.NUM_SPLITS random unsplit genes
            max_splits = min(config.NUM_SPLITS, len(self.genes))
            num_splits = random.randint(5, max_splits)
            for _ in range(num_splits):
                # pick an unsplit gene (parent_id is None)
                unsplit_idxs = [i for i, g in enumerate(self.genes) if g.parent_id is None]
                if not unsplit_idxs:
                    break
                # Favor larger pieces: weight by bounding-box area
                areas = []
                for i in unsplit_idxs:
                    # ensure bounding box is up to date
                    self.genes[i].update_bbox()
                    areas.append(self.genes[i].bbox_area)
                idx = random.choices(unsplit_idxs, weights=areas, k=1)[0]
                piece = self.genes[idx]
                left, right = piece.split()
                # replace original with left and right
                self.genes.pop(idx)
                self.genes.insert(idx, left)
                self.genes.insert(idx + 1, right)
                # relocate one of the split halves to a random position
                child_to_move = random.choice([left, right])
                # remove the chosen child from its current position
                cur_pos = self.genes.index(child_to_move)
                self.genes.pop(cur_pos)
                # choose a new insert index across the genes list
                new_pos = random.randrange(len(self.genes) + 1)
                self.genes.insert(new_pos, child_to_move)
                if config.VERBOSE:
                    print(f"[Chromosome.mutate] relocated {child_to_move.id} to position {new_pos}")
                if config.VERBOSE:
                    print(f"[Chromosome.mutate] split {piece.id} into {left.id} and {right.id}")
            # print resulting gene sequence
            if config.VERBOSE:
                print(f"[Chromosome.mutate] new genes: {[p.id for p in self.genes]}")

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

    
    def crossover_ox1(self, other: "Chromosome", k: int = 1) -> "Chromosome":
        """
        Generalised OX1 crossover that respects component split trees.

        Rules
        -----
        • Parents may contain different numbers of leaves (genes).
        • The child inherits complete split-trees: every original component
        (root_id) is taken entirely from *one* parent, never mixed.
        • Step 1 is still “OX1 style”: pick 2·k cut points on *self* and copy
        those segments (plus any additional leaves that belong to the same
        components) into the child, preserving order.

        Parameters
        ----------
        other : Chromosome
            The second parent.
        k : int
            Number of OX1 segments (default 1).

        Returns
        -------
        Chromosome
            The offspring chromosome.
        Component-completion OX1 crossover (CC-OX1).

        Parents may differ in length.
        A split-tree (root_id) is inherited wholesale from exactly one parent.
        """

        t0 = time.time()

        # ------------------------------------------------------------------
        # helpers -----------------------------------------------------------
        # ------------------------------------------------------------------
        def get_root_id(piece):
            """Return the identifier of the original (pre-split) component."""
            return getattr(piece, "root_id", None) or piece.id.split("_")[0]

        def copy_all_leaves(src_genes, root_id, placed):
            """
            Copy *all* leaves of `root_id` from `src_genes` into child_genes,
            preserving their order in `src_genes`.  Uses deep-copies and
            updates `placed`.
            """
            copied = []
            for g in src_genes:
                if get_root_id(g) == root_id and g.id not in placed:
                    child_genes.append(copy.deepcopy(g))
                    placed.add(g.id)
                    copied.append(g.id)
            return copied

        # ------------------------------------------------------------------
        # 1) choose 2·k cut points on *self* and create first draft segment --
        # ------------------------------------------------------------------
        size_self = len(self.genes)
        if size_self == 0:
            raise ValueError("Parent 1 has no genes")

        if 2 * k > size_self:
            raise ValueError("2·k cut points exceed chromosome length")

        cut_points = sorted(random.sample(range(size_self), 2 * k))
        segments = [(a, b) if a <= b else (b, a)
                    for a, b in zip(cut_points[::2], cut_points[1::2])]

        child_genes: list["Piece"] = []
        placed_ids: set[str] = set()
        chosen_source: dict[str, str] = {}     # root_id → "self" | "other"

        # Copy the chosen segments from self (plus all leaves of their components)
        for seg_start, seg_end in segments:
            for idx in range(seg_start, seg_end + 1):
                g = self.genes[idx]
                rid = get_root_id(g)
                if rid in chosen_source:
                    # This component was already taken earlier in this loop
                    continue
                chosen_source[rid] = "self"
                copy_all_leaves(self.genes, rid, placed_ids)

        # ------------------------------------------------------------------
        # 2) walk through parent 2 and take components not chosen yet --------
        # ------------------------------------------------------------------
        for g in other.genes:
            rid = get_root_id(g)
            if rid in chosen_source:                # competing tree → skip
                continue
            chosen_source[rid] = "other"
            copy_all_leaves(other.genes, rid, placed_ids)

        # ------------------------------------------------------------------
        # 3) finally, copy any still-missing components from parent 1 --------
        # ------------------------------------------------------------------
        for g in self.genes:
            rid = get_root_id(g)
            if rid in chosen_source:                # already taken
                continue
            chosen_source[rid] = "self"
            copy_all_leaves(self.genes, rid, placed_ids)

        # ------------------------------------------------------------------
        # 4) finished -------------------------------------------------------
        # ------------------------------------------------------------------
        if config.LOG_TIME:
            dt = time.time() - t0
            print(f"[Chromosome.crossover_ox1] k={k} produced "
                f"{len(child_genes)} genes in {dt:.4f} s, segments={segments}")

        return Chromosome(child_genes, self.container)


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
