# chromosome.py

from __future__ import annotations
import time
import random
import copy
import json
import csv
from fnmatch import fnmatch
from pathlib import Path
from typing import Callable, Iterable, Sequence, Any
from functools import wraps

from collections import deque
import copy
from typing import List, Set

from .layout import Piece, Container, Layout, LayoutView
from .placement_engine import DECODER_REGISTRY
#from pygarment.garmentcode.params import DesignSampler
from pygarment.garmentcode.utils import nested_get, nested_set, nested_del
import nesting.config as config
from nesting.panel_mapping import affected_panels


# ── Metric Registration ─────────────────────────────────────────────────────────

METRIC_REGISTRY: dict[str, Callable] = {}

def register_metric(name: str):
    def deco(fn):
        METRIC_REGISTRY[name] = fn
        return fn
    return deco

# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

JsonDict = dict[str, Any]


def track_roots(fn: Callable):
    """Decorator to track root IDs before and after a mutation."""
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        before = {g.root_id for g in self.genes}
        result = fn(self, *args, **kwargs)
        mutation = fn.__name__.replace("_mutate_", "")
        self._warn_if_panel_lost(before, mutation)
        return result
    return wrapper


def _run_decoder(chrom: "Chromosome", decoder_name: str):
    """Helper to instantiate & run the placement decoder."""
    view = LayoutView(chrom.genes)
    Decoder = DECODER_REGISTRY[decoder_name]
    dec = Decoder(view, chrom.container)
    dec.decode()
    return dec


def _weighted_choice(options: dict[str, float]) -> str:
    """Return a key from *options* using their values as weights."""
    choices, weights = zip(*options.items())
    return random.choices(choices, weights)[0]


def _next_owned_run(
    seq: list["Piece"], owner: dict[str, str], take_owner: str, start: int
) -> tuple[int | None, int | None]:
    """Return the start and end of the next run owned by ``take_owner``.

    The returned tuple is ``(s, e)`` where ``seq[s:e]`` is the maximal
    contiguous slice of ``seq`` owned by ``take_owner`` beginning at or after
    ``start``. If no such run exists, ``(None, None)`` is returned.
    """
    n = len(seq)
    s = start
    while s < n and owner.get(seq[s].root_id) != take_owner:
        s += 1
    if s >= n:
        return None, None
    e = s
    while e < n and owner.get(seq[e].root_id) == take_owner:
        e += 1
    return s, e  # [s, e)


# ──────────────────────────────────────────────────────────────────────────────
# Design‑parameter helpers
# ──────────────────────────────────────────────────────────────────────────────

def _flatten_param_paths(node: JsonDict, prefix: list[str] | None = None) -> list[str]:
    """Return all *non‑meta* leaf‑paths that contain a ``v`` key."""
    prefix = prefix or []
    paths: list[str] = []
    for k, v in node.items():
        # if k == "meta":
        #     continue  # never touch meta‑data
        if isinstance(v, dict):
            if "v" in v:
                paths.append(".".join(prefix + [k]))
            else:
                paths += _flatten_param_paths(v, prefix + [k])
    return paths


# # TODO: FIXME: this does not work
# def _filter_excluded(paths: Sequence[str]) -> list[str]:
#     patterns = config.EXCLUDED_PARAM_PATHS or []
#     if not patterns:
#         return list(paths)
#     return [p for p in paths if not any(fnmatch(p, pat) for pat in patterns)]


def _numeric_range_ok(node: JsonDict) -> bool:
    rng = node.get("range", [])
    return len(rng) >= 2 and all(isinstance(x, (int, float)) for x in rng[:2])


# def _choose_param(params: JsonDict, paths: Sequence[str]) -> str | None:
#     """Return a random parameter. """
#     numeric = [p for p in paths if _numeric_range_ok(nested_get(params, p.split(".")))]
#     return random.choice(numeric)


def _random_value(old: Any, base: Any, p_type: str, rng: Sequence[float | int]):
    """Return a *new* value that differs from *base* by ≤ margin param % of *range* width
    while still being different from *old*.

    The *base* value represents the original design parameter value before any
    mutations, which allows us to limit cumulative drift across generations.
    """
    
    # # Handle boolean type explicitly
    # if p_type == "bool" or isinstance(old, bool):
    #     return old  # Don't randomize boolean values
        
    # # Handle None value
    # if old is None:
    #     return None
        
    lower, upper = rng[0], rng[1]
    span = upper - lower
    max_delta = config.PARAM_CHANGE_MARGIN * span

    # if span <= 0:
    #     return old  # degenerate range – nothing we can do

    # integer parameters --------------------------------------------------
    if p_type == "int":
        vals = list(range(int(lower), int(upper) + 1))
        candidates = [v for v in vals if abs(v - base) <= max_delta and v != old]
        if not candidates:
            return old  # no valid candidates, return old value
        return random.choice(candidates)

    # float parameters ----------------------------------------------------
    attempts = 0
    while attempts < 10:
        cand = random.uniform(lower, upper)
        if abs(cand - base) <= max_delta and cand != old:
            return cand
        attempts += 1
    # fallback – choose a value slightly different from base but within max_delta
    cand = base + max_delta / 2
    if cand <= upper and cand != old:
        return cand
    cand = base - max_delta / 2
    if cand >= lower and cand != old:
        return cand
    # should not happen in theory
    return old
    

def _collect_mutatable_params(
    design_params: JsonDict,
    genes: list[Piece]
) -> list[str]:
    """Return all numeric param paths that (1) are not excluded and
       (2) affect at least one of the current chromosome's panels."""
    if design_params is None:
        return []

    # flatten design param paths
    all_paths = _flatten_param_paths(design_params)

    # filter out non-numeric & excluded
    patterns = config.EXCLUDED_PARAM_PATHS or []
    numeric_ok = [
        p for p in all_paths
        if _numeric_range_ok(nested_get(design_params, p.split(".")))
        and not any(fnmatch(p, pat) for pat in patterns)
    ]
    #print(f"[DEBUG] After numeric & exclusion filtering: {len(numeric_ok)} paths")
    
    # filter out the ones that don't affect any panel in this chromosome
    panel_ids = {g.id for g in genes}
    # print(f"[DEBUG] Panel IDs in chromosome: {panel_ids}")
    mutatable = [p for p in numeric_ok if (panel_patterns := affected_panels([p], design_params)) 
                 and any(fnmatch(pid, pat) for pat in panel_patterns for pid in panel_ids)]
            #print(f"[DEBUG] Parameter {p} affects panels: {matching_panels}")
    
    #print(f"[DEBUG] Final mutatable parameters: {len(mutatable)}")
    # if config.VERBOSE or True:  # Always print for debugging
    #     print(f"[Chromosome] Mutatable design params: {mutatable}")

    return mutatable

@register_metric("usage_bb")
def fitness_usage_bb(chromosome: Chromosome, decoder: str):
    dec = _run_decoder(chromosome, decoder)
    return dec.usage_BB()

@register_metric("concave_hull")
def fitness_concave_hull(chromosome: Chromosome, decoder: str):
    dec = _run_decoder(chromosome, decoder)
    return dec.concave_hull_utilization()

@register_metric("concave_hull_area")
def fitness_concave_hull_area(chromosome: Chromosome, decoder: str):
    dec = _run_decoder(chromosome, decoder)
    area = dec.concave_hull_area()

    if area < 1e-6:
        return 0
    
    return 10000/dec.concave_hull_area()

@register_metric("rest_length")
def fitness_rest_length(chromosome: Chromosome, decoder: str):
    dec = _run_decoder(chromosome, decoder)
    return dec.rest_length()

@register_metric("rest_height")
def fitness_rest_height(chromosome: Chromosome, decoder: str):
    dec = _run_decoder(chromosome, decoder)
    return dec.rest_height()

@register_metric("cc_with_rest_height")
def fitness_cc_height_combined(chromosome: Chromosome, decoder: str):
    """
    Combined fitness metric for concave hull height and rest height.
    
    This metric returns the sum of the concave hull height and the rest height.
    It is useful for evaluating the overall vertical space utilization of the layout.
    """
    dec = _run_decoder(chromosome, decoder)
    cc = dec.concave_hull_utilization()
    rest_height = dec.rest_height()
    if cc == 0:
        return 0
    return cc + config.REST_PENALTY * rest_height

@register_metric("bb_with_rest_length")
def fitness_bb_length_combined(chromosome: Chromosome, decoder: str):
    """
    Combined fitness metric for bounding box length and rest length.
    
    This metric returns the sum of the bounding box length and the rest length.
    It is useful for evaluating the overall vertical space utilization of the layout.
    """
    dec = _run_decoder(chromosome, decoder)
    bb = dec.usage_BB()
    rest_length = dec.rest_length()
    if bb == 0:
        return 0
    return bb + config.REST_PENALTY * rest_length

@register_metric("cc_with_rest_length")
def fitness_cc_length_combined(chromosome: Chromosome, decoder: str):
    """
    Combined fitness metric for concave hull length and rest length.
    
    This metric returns the sum of the concave hull length and the rest length.
    It is useful for evaluating the overall horizontal space utilization of the layout.
    """
    dec = _run_decoder(chromosome, decoder)
    cc = dec.concave_hull_utilization()
    rest_length = dec.rest_length()
    if cc == 0:
        return 0
    return cc + config.REST_PENALTY * rest_length

@register_metric("bb_cc")
def fitness_bb_cc(chromosome: Chromosome, decoder: str):
    """
    Combined fitness metric for bounding box area and concave hull area.
    
    This metric returns the sum of the bounding box area and the concave hull area.
    It is useful for evaluating the overall space utilization of the layout.
    """
    dec = _run_decoder(chromosome, decoder)
    bb_area = dec.usage_BB()
    cc_area = dec.concave_hull_utilization()

    if bb_area == 0 or cc_area == 0:
        return 0
    return config.BB_WEIGHT * bb_area + config.CC_WEIGHT * cc_area

@register_metric("bb_area")
def fitness_bb_area(chromosome: Chromosome, decoder: str):
    dec = _run_decoder(chromosome, decoder)
    bb_area = dec.bbox_area()
    # cc_area = dec.concave_hull_area()

    if bb_area == 0:
        return 0
    return 10000 / bb_area


@register_metric("bb_cc_area")
def fitness_bb_cc_area(chromosome: Chromosome, decoder: str):
    dec = _run_decoder(chromosome, decoder)
    bb_area = dec.bbox_area()
    cc_area = dec.concave_hull_area()

    if bb_area == 0 or cc_area == 0:
        return 0
    return 10000 / (config.BB_WEIGHT * bb_area + config.CC_WEIGHT * cc_area)

# ── Chromosome Definition ───────────────────────────────────────────────────────

class Chromosome(Layout):

    def __init__(
        self,
        pieces: list[Piece],
        container: Container,
        origin: str = "random",
        *,
        design_params: dict | None = None,
        body_params: object | None = None,
        initial_design_params: dict | None = None,
        #design_sampler: "DesignSampler" | None = None,
    ):
        # Store a deep copy of each piece
        self._genes = [copy.deepcopy(p) for p in pieces]
        self.container = container
        self.fitness: float | None = None

        self.design_params = copy.deepcopy(design_params) if design_params else None
        self.body_params = body_params
        if initial_design_params is not None:
            self.initial_design_params = copy.deepcopy(initial_design_params)
        else:
            self.initial_design_params = copy.deepcopy(design_params) if design_params else None
        self._mutatable_params = _collect_mutatable_params(
            self.design_params, self._genes
        )

        # Track origin and last mutation type
        self.origin: str | None = origin
        self.last_mutation: str | None = None
        
        # Initialize empty list to track parameter changes in the current generation
        self.param_changes_this_gen = []
        
        # Add tracking for mutation statistics
        self.old_fitness: float | None = None
        self.new_fitness: float | None = None
        self.mutation_improvement: float | None = None
        self.split_history: list[tuple[str, float]] = []

        # Keep a MetaGarment instance
        self.meta_garment = None
        if self.design_params and self.body_params:
            try:
                from assets.garment_programs.meta_garment import MetaGarment
                self.meta_garment = MetaGarment("chromosome_base", self.body_params, self.design_params)
            except Exception as exc:
                print(f"[Chromosome] Failed to create MetaGarment: {exc}")
        

    @property
    def genes(self) -> list[Piece]:
        return self._genes

    @genes.setter
    def genes(self, value: list[Piece]):
        """Allow reassignment of the gene list (wraps _genes)."""
        self._genes = value

    def calculate_fitness(self) -> None:
        """Compute fitness via the registered metric and decoder."""
        metric_fn = METRIC_REGISTRY[config.SELECTED_FITNESS_METRIC]
        self.fitness = metric_fn(self, config.SELECTED_DECODER)

    def mutate(self) -> "Chromosome":
        """Pick a mutation based on config.MUTATION_WEIGHTS and apply it."""
        start = time.time()
        self.last_mutation = mutation = _weighted_choice(config.MUTATION_WEIGHTS)

        # Clear the parameter changes tracking for this generation
        self.param_changes_this_gen = []
        
        if config.VERBOSE:
            print(f"[Chromosome.mutate] : {mutation}")

        handler = {
            "split": self._mutate_split,
            "rotate": self._mutate_rotate,
            "swap": self._mutate_swap,
            "inversion": self._mutate_inversion,
            "insertion": self._mutate_insertion,
            "scramble": self._mutate_scramble,
            "design_params": self._mutate_design_params,
        }.get(mutation)

        if handler is None:
            raise ValueError(f"Unknown mutation type: {mutation}")

        handler()  # apply the mutation

        if config.LOG_TIME:
            print(f"[Chromosome.mutate] {mutation} took {time.time() - start:.3f}s")
        
        # Debug: Check for fitness change
        if self.old_fitness is not None and self.new_fitness is not None and self.old_fitness == self.new_fitness:
            print(f"[DEBUG] Mutation '{self.last_mutation}' resulted in no fitness change.")

        return self

    def _warn_if_panel_lost(self, before_roots: set[str], mutation: str) -> None:
        """Emit a warning if any panels disappeared after a mutation."""
        after_roots = {g.root_id for g in self.genes}
        missing = before_roots - after_roots
        if missing:
            # print(f"[Chromosome] WARNING: mutation '{mutation}' lost panels: {sorted(missing)}")
            # throw an error
            raise AssertionError(f"[Chromosome] ERROR: mutation '{mutation}' lost panels: {sorted(missing)}. This should not happen!")

        return

    # ── simple mutations ───────────────────────────────────────────────
    def meta_split(self, piece: Piece, idx: int, piece_mirror: Piece | None = None) -> bool:
        from nesting.path_extractor import PatternPathExtractor
        from assets.garment_programs.meta_garment import MetaGarment
        from pathlib import Path
        import tempfile

        mg = self.meta_garment

        # choose split proportion and perform
        proportion = random.uniform(config.SPLIT_LOWER_BOUND, config.SPLIT_UPPER_BOUND)
        new_panel_names = mg.split_panel(piece.id, proportion=proportion)
        if not new_panel_names:
            print(f"[Chromosome] MetaGarment split failed for {piece.id}: no new panels returned")
            return False

        # record only successful split, store ROOT id
        self.split_history.append((piece.root_id, proportion))
        print(f"[Chromosome] Split {piece.id} into {new_panel_names} with proportion {proportion}")

        new_panel_names_mirror = None
        if piece_mirror and config.SYMMETRIC_SPLITS:
            # If there's a mirror piece, also split it with complementary proportion
            mirror_proportion = 1 - proportion
            new_panel_names_mirror = mg.split_panel(piece_mirror.id, proportion=mirror_proportion)
            self.split_set.add(piece_mirror)  # track mirror piece as split
            if not new_panel_names_mirror:
                print(f"[Chromosome] MetaGarment split failed for mirror {piece_mirror.id}")
            else:
                # record mirror split using ROOT id
                self.split_history.append((piece_mirror.root_id, mirror_proportion))
                print(f"[Chromosome] Split {piece_mirror.id} into {new_panel_names_mirror} with proportion {mirror_proportion}")

        # Rebuild pattern and extract pieces
        pattern = mg.assembly()
        with tempfile.TemporaryDirectory() as td:
            spec_file = Path(td) / f"{pattern.name}_specification.json"
            pattern.serialize(Path(td), to_subfolder=False, with_3d=False,
                            with_text=False, view_ids=False)
            extractor = PatternPathExtractor(spec_file)
            all_pieces = extractor.get_all_panel_pieces(
                samples_per_edge=config.SAMPLES_PER_EDGE
            )

        # Build new Piece objects for the split panels
        left_piece = all_pieces.get(new_panel_names[0]) if new_panel_names else None
        right_piece = all_pieces.get(new_panel_names[1]) if new_panel_names else None

        if left_piece is None or right_piece is None:
            print(f"[Chromosome] Split pieces not found for {piece.id} in regenerated pattern")
            return False

        for p_new in (left_piece, right_piece):
            p_new.add_seam_allowance(config.SEAM_ALLOWANCE_CM)
            p_new.parent_id = piece.id
            p_new.root_id = getattr(piece, "root_id", piece.id)
            p_new.rotation = piece.rotation
            p_new.translation = piece.translation
            p_new.update_bbox()

        mirror_left, mirror_right = None, None
        if piece_mirror and config.SYMMETRIC_SPLITS and new_panel_names_mirror:
            mirror_left = all_pieces.get(new_panel_names_mirror[0])
            mirror_right = all_pieces.get(new_panel_names_mirror[1])
            if mirror_left is None or mirror_right is None:
                print(f"[Chromosome] Mirror split pieces not found for {piece_mirror.id} in regenerated pattern")
            else:
                for p_new in (mirror_left, mirror_right):
                    p_new.add_seam_allowance(config.SEAM_ALLOWANCE_CM)
                    p_new.parent_id = piece_mirror.id
                    p_new.root_id = getattr(piece_mirror, "root_id", piece_mirror.id)
                    p_new.rotation = piece_mirror.rotation
                    p_new.translation = piece_mirror.translation
                    p_new.update_bbox()

        # Replace the original piece with the two split pieces
        self.genes = self._replace(piece, left_piece, right_piece, self.genes)

        if mirror_left is not None and mirror_right is not None:
            try:
                self.genes = self._replace(piece_mirror, mirror_left, mirror_right, self.genes)
            except ValueError:
                print(f"[Chromosome] WARNING: piece_mirror {piece_mirror.id} not in genes, but mirror_left and mirror_right were created.")

        return True  # Indicate successful mutation


    # Recompute mutatable params since gene ids changed
    # self._mutatable_params = _collect_mutatable_params(self.design_params, self._genes)

    

    def _replace(self, piece: Piece, new_piece_left: Piece, new_piece_right: Piece, genes:list) -> None:
        """Replace a piece at index *idx* with *new_piece_left* and insert the second
        piece at a random position."""

        idx = self.genes.index(piece) # get the index of the piece to replace
        genes.remove(piece) # remove it
        genes.insert(idx, new_piece_left) # insert the first piece at the original index
        genes.insert(random.randrange(len(genes)), new_piece_right) # insert the second piece at a random position

        return genes

    @track_roots
    def _mutate_split(self):
        #from nesting.panel_mapping import dispatch_split

        # Track already-split root pieces across calls
        self.split_set = getattr(self, 'split_set', set())

        def _unsplit_roots():
            return [g for g in self.genes if g.parent_id is None and g not in self.split_set]

        candidates = _unsplit_roots()

        # TODO: allow recursive splits
        # DO NOT TURN THIS ON, 
        # IT DOES NOT CURRENTLY WORK
        if not candidates and not config.ALLOW_RECURSIVE_SPLITS:
            return False


        # randomly pick a piece from the candidates
        if config.WEIGHT_BY_BBOX:
            weights = [c.bbox_area for c in candidates]
            piece = random.choices(candidates, weights=weights, k=1)[0]
        else:
            piece = random.choice(candidates)

        self.split_set.add(piece)

        # Mirror lookup (basic left/right root swap)
        # TODO: make this less sucky
        piece_mirror = next((p for p in self.genes
                              if p.parent_id is None and p is not piece and (
                                  p.id == piece.id.replace("left", "right") or
                                  p.id == piece.id.replace("right", "left"))), None)

        if not self.meta_garment: # no parameter pipeline
            left, right = piece.split()
            self._replace(piece, left, right, self.genes)
            # self.split_set.add(left); self.split_set.add(right) # split set is for roots so this should not be here
            if piece_mirror and config.SYMMETRIC_SPLITS:  # if we have a mirror piece
                self.split_set.add(piece_mirror)
                left_mirror, right_mirror = piece_mirror.split()
                self.genes = self._replace(piece_mirror, left_mirror, right_mirror, self.genes)
                # self.split_set.add(left_mirror); self.split_set.add(right_mirror) # see above
        else: # pipeline with design and body parameters
            success = self.meta_split(piece, self.genes.index(piece), piece_mirror if config.SYMMETRIC_SPLITS else None)
            if not success:
                raise ValueError(f"[Chromosome] MetaGarment split failed for {piece.id}")

        return True

    @track_roots
    def _mutate_rotate(self):
        for _ in range(random.randint(1, len(self.genes))):
            idx = random.randrange(len(self.genes))
            self.genes[idx].rotate(random.choice(config.ALLOWED_ROTATIONS))

    @track_roots
    def _mutate_swap(self, k=config.SWAP_MUTATION_K):
        """
        Swap two non-overlapping contiguous blocks of length k.
        If k is None, choose a random valid k in [1, floor(n/2)].
        """

        n = len(self.genes)
        if n < 2:
            return

        # Determine a suitable k
        max_k = n // 2  # must allow two non-overlapping blocks
        if max_k < 1:
            return
        if k is None:
            k = random.randint(1, max_k)

        print(f"[DEBUG] k: {k}, max_k: {max_k}")

    # roots tracked by decorator

        # Build all valid non-overlapping (i, j) with i < j and j >= i + k
        pairs: list[tuple[int, int]] = []
        last_start = n - k
        for i in range(0, n - 2 * k + 1):
            for j in range(i + k, last_start + 1):
                pairs.append((i, j))

        if not pairs:
            # No valid non-overlapping blocks (shouldn't happen with k <= n//2)
            return

        i, j = random.choice(pairs)

        # Splice: genes = pre + B + mid + A + post
        pre = self.genes[:i]
        A = self.genes[i:i + k]
        mid = self.genes[i + k:j]
        B = self.genes[j:j + k]
        post = self.genes[j + k:]
        self.genes = pre + B + mid + A + post

    # post-check is handled by decorator

    @track_roots
    def _mutate_inversion(self):
        i, j = sorted(random.sample(range(len(self.genes)), 2))
        self.genes[i:j + 1] = reversed(self.genes[i:j + 1])

    @track_roots
    def _mutate_insertion(self, k = 1):
        if len(self.genes) < 2:
            return
        
        n = len(self.genes)

        if k > n:
            raise ValueError("Invalid insertion operation: k too big for n")
        
    # roots tracked by decorator
        
        i = random.randrange(len(self.genes))
        insert_at = random.randrange(len(self.genes) + 1)
        gene = self.genes.pop(i)
        self.genes.insert(insert_at, gene)
    # post-check is handled by decorator

    @track_roots
    def _mutate_scramble(self):
        i, j = sorted(random.sample(range(len(self.genes)), 2))
        subset = self.genes[i:j + 1]
        random.shuffle(subset)
        self.genes[i:j + 1] = subset
        # post-check is handled by decorator

    # ── design‑parameter mutation ──────────────────────────

    def _select_candidate_params(self) -> list[str]:
        """Return a shuffled list of parameter paths to try."""
        params = list(self._mutatable_params)
        random.shuffle(params)
        return params

    def _apply_single_param_change(
        self,
        param: str,
        original_design_params: JsonDict,
        original_genes: list[Piece],
        original_fitness: float,
    ) -> bool:
        """Attempt a single parameter modification and evaluate fitness."""
        self.design_params = copy.deepcopy(original_design_params)
        self.genes = copy.deepcopy(original_genes)

        node = nested_get(self.design_params, param.split("."))
        p_type = node.get("type", "float")
        old_val = node["v"]
        base_val = nested_get(self.initial_design_params, param.split("."))["v"]
        new_val = _random_value(old_val, base_val, p_type, node["range"])

        if old_val == new_val:
            return False

        nested_set(self.design_params, param.split(".") + ["v"], new_val)

        if self._apply_design_param_change(param, old_val, new_val):
            self.calculate_fitness()
            if self.fitness != original_fitness:
                self._record_mutation_stats(param, old_val, new_val, original_fitness)
                return True
        return False

    def _record_mutation_stats(
        self, param: str, old_val: Any, new_val: Any, old_fitness: float
    ) -> None:
        """Store parameter change and mutation improvement statistics."""
        print(f"[Chromosome] Successful mutation on {param}: {old_val} -> {new_val}")
        self.param_changes_this_gen.append(
            {"param_path": param, "old_value": old_val, "new_value": new_val}
        )
        self.old_fitness = old_fitness
        self.new_fitness = self.fitness
        if self.new_fitness is not None and old_fitness is not None:
            self.mutation_improvement = self.new_fitness - old_fitness

    def _restore_state_after_failure(
        self,
        original_design_params: JsonDict,
        original_genes: list[Piece],
        original_fitness: float,
    ) -> None:
        """Restore chromosome state when no parameter change succeeds."""
        self.design_params = original_design_params
        self.genes = original_genes
        self.fitness = original_fitness
        if config.VERBOSE:
            print("[Chromosome] No design param mutation resulted in a fitness change.")

    @track_roots
    def _mutate_design_params(self):
        if not (self.design_params and self.body_params):
            if config.VERBOSE:
                print("[Chromosome] design‑param mutation skipped - missing design or body params")
            return

        # roots tracked by decorator

        if self.fitness is None:
            self.calculate_fitness()
        original_fitness = self.fitness

        original_design_params = copy.deepcopy(self.design_params)
        original_genes = copy.deepcopy(self.genes)

        for param in self._select_candidate_params():
            if self._apply_single_param_change(
                param, original_design_params, original_genes, original_fitness
            ):
                return True

        self._restore_state_after_failure(
            original_design_params, original_genes, original_fitness
        )
        return False

    def _apply_design_param_change(self, path: str, old_val: Any, new_val: Any) -> bool:
        """Helper to regenerate garment pieces after a design param change and update genes."""
        from assets.garment_programs.meta_garment import MetaGarment
        from nesting.panel_mapping import affected_panels, select_genes
        from nesting.path_extractor import PatternPathExtractor
        import tempfile

        panel_ids = {g.id for g in self.genes}
        affected = affected_panels([path], self.design_params)

        if not any(fnmatch(pid, pat) for pat in affected for pid in panel_ids):
            if config.VERBOSE:
                print(f"[Chromosome] No affected panels for {path} in this chromosome")
            return False

        mg = MetaGarment("design_mut", self.body_params, self.design_params)

        # Restore splits from history
        for panel_name, proportion in self.split_history:
            print(f"[Chromosome] Reapplying split for {panel_name} with proportion {proportion}")
            mg.split_panel(panel_name, proportion)

        pattern = mg.assembly()
        with tempfile.TemporaryDirectory() as td:
            spec_file = Path(td) / f"{pattern.name}_specification.json"
            pattern.serialize(Path(td), to_subfolder=False, with_3d=False,
                                with_text=False, view_ids=False)
            extractor = PatternPathExtractor(spec_file)
            new_pieces = extractor.get_all_panel_pieces(
                samples_per_edge=config.SAMPLES_PER_EDGE)
            for piece in new_pieces.values():
                piece.add_seam_allowance(config.SEAM_ALLOWANCE_CM)

        changed_ids = select_genes(new_pieces.keys(), affected)
        # Rebuild mapping for quicker lookups
        new_piece_ids = set(new_pieces.keys())

        # Extend changed ids: if a root id changed and we have split fragments for it in the current genes
        # include those fragment ids so we directly replace them from new_pieces instead of re-splitting.
        extended_changed_ids = set(changed_ids)
        root_ids_changed = {cid for cid in changed_ids if cid in new_piece_ids}
        for root_id in root_ids_changed:
            # any existing gene whose root_id == root_id should be refreshed if its concrete id exists in new_pieces
            for g in self.genes:
                if g.root_id == root_id and g.id in new_piece_ids:
                    extended_changed_ids.add(g.id)

        for i, g in enumerate(self.genes):
            if g.id in extended_changed_ids and g.id in new_pieces:
                # Direct replacement from regenerated pattern; preserve placement attributes
                replacement = copy.deepcopy(new_pieces[g.id])
                replacement.rotation = g.rotation
                replacement.parent_id = g.parent_id
                replacement.root_id = g.root_id
                #replacement.translation = g.translation
                self.genes[i] = replacement

        # Removed CSV logging of design parameter paths
        return True
    
       # -------------------- helpers for crossover --------------------

    def _flatten_param_paths(self, node: dict, prefix: list[str] | None = None) -> list[str]:
        prefix = prefix or []
        out = []
        for k, v in node.items():
            if isinstance(v, dict):
                if "v" in v:
                    out.append(".".join(prefix + [k]))
                else:
                    out += self._flatten_param_paths(v, prefix + [k])
        return out

    def _conflict_groups(self, dp1: dict | None, dp2: dict | None, all_root_ids: set[str]):
        """"
        Return [{"roots": set[str], "paths": set[str]}] for params whose *values differ* in dp1 vs dp2.
        Assumes both parents share the same design-parameter schema; only numeric 'v' values differ.
        Uses dp1 as the canonical schema for affected_panels().
        """
        if not (dp1 and dp2):
            return []

        from pygarment.garmentcode.utils import nested_get
        from nesting.panel_mapping import affected_panels, select_genes

        def _get_val(dp: dict, path: str):
            # fetches the current value of a design parameter at *path*.
            try:
                node = nested_get(dp, path.split("."))
            except Exception:
                return None
            return node.get("v") if isinstance(node, dict) and "v" in node else node

        # --- strict schema equivalence checks (fast fail if anything drifts) ---
        paths1 = set(self._flatten_param_paths(dp1))
        paths2 = set(self._flatten_param_paths(dp2))
        if paths1 != paths2:
            # If this ever trips, your "only numeric values differ" assumption has been violated.
            raise RuntimeError("Design-parameter schema mismatch between parents (path sets differ).")

        # same metadata checks per path
        for p in paths1:
            n1 = nested_get(dp1, p.split("."))
            n2 = nested_get(dp2, p.split("."))
            t1, t2 = (n1.get("type"), n2.get("type"))
            if t1 != t2:
                raise RuntimeError(f"Type mismatch at '{p}': {t1} vs {t2}")
            if "range" in n1 or "range" in n2:
                if n1.get("range") != n2.get("range"):
                    raise RuntimeError(f"Range mismatch at '{p}': {n1.get('range')} vs {n2.get('range')}")

        # --- collect differing-value paths ---
        diffs = [p for p in paths1 if _get_val(dp1, p) != _get_val(dp2, p)]
        if not diffs:
            return []

        # --- build groups: one group per differing param path (NO transitive merging) ---
        # Requirement: panels share a group ONLY if they are co-affected by the *same* differing parameter.
        # So if A,B affected by param p1 and B,C by param p2, we create two groups:
        #   G1: roots {A,B}, paths {p1}
        #   G2: roots {B,C}, paths {p2}
        # A and C never appear together just because of B.
        groups: list[dict] = []
        for param in diffs:
            pats = affected_panels([param], dp1) or []
            roots = set(select_genes(all_root_ids, pats) or [])
            if not roots:
                continue
            groups.append({"roots": roots, "paths": {param}})

        return groups

    @staticmethod
    def _history_by_root(history: list[tuple[str, float]] | None) -> dict[str, list[tuple[str, float]]]:
        """Convert a history list of (root_id, property_value) tuples into a dict keyed by root_id."""
        d: dict[str, list[tuple[str, float]]] = {}
        for rid, prop in (history or []):
            d.setdefault(rid, []).append((rid, prop))
        return d

    # -------- helpers for owner-stable weave (preserve non-contiguous patterns) --------

    @staticmethod
    def _root_order_index_map(order: list[str]) -> dict[str, int]:
        """root_id -> its index in the child OX root order (for deterministic weaving)."""
        return {r: i for i, r in enumerate(order)}
    
    def _root_order(parent: "Chromosome") -> list[str]:
        """Unique root_ids in the order they first appear."""
        seen, out = set(), []
        for g in parent.genes:
            if g.root_id not in seen:
                seen.add(g.root_id)
                out.append(g.root_id)
        return out

    
    @staticmethod
    def _leaf_ids(seq: list["Piece"]) -> dict[str, set[str]]:
        parent_ids = {p.parent_id for p in seq if p.parent_id}
        out: dict[str, set[str]] = {}
        for p in seq:
            if p.id not in parent_ids:
                out.setdefault(p.root_id, set()).add(p.id)
        return out



    @staticmethod
    def _filter_sequence_by_owner(seq: list["Piece"], owner: dict[str, str | None], take_owner: str) -> list["Piece"]:
        """Keep pieces whose root is owned by take_owner, preserving original order."""
        return [p for p in seq if owner.get(p.root_id) == take_owner]

    def _build_child_by_owner_weave(
        self,
        order_child_roots: list[str],
        owner: dict[str, str | None],
        p1_seq: list["Piece"],
        p2_seq: list["Piece"],
    ) -> list["Piece"]:
        """
        Stable-merge two parent sequences:
          - take only leaves from the owning parent for each root
          - preserve the relative order within each parent's kept subsequence
          - interleave deterministically using the child root order as a tie-breaker
        """
        p1_keep = self._filter_sequence_by_owner(p1_seq, owner, "P1")
        p2_keep = self._filter_sequence_by_owner(p2_seq, owner, "P2")

        i, j = 0, 0
        out: list["Piece"] = []
        root_rank = self._root_order_index_map(order_child_roots)

        # while we have remaining pieces in either parent
        while i < len(p1_keep) or j < len(p2_keep):
            if j >= len(p2_keep):  # all of p2_keep has been added
                out.append(copy.deepcopy(p1_keep[i])); i += 1; continue
            if i >= len(p1_keep):  # all of p1_keep has been added
                out.append(copy.deepcopy(p2_keep[j])); j += 1; continue

            r1 = p1_keep[i].root_id  # root_id of the next piece in p1
            r2 = p2_keep[j].root_id  # root_id of the next piece in p2

            assert r1 in root_rank and r2 in root_rank

            rank1 = root_rank[r1]
            rank2 = root_rank[r2]

            if rank1 < rank2:
                out.append(copy.deepcopy(p1_keep[i])); i += 1
            elif rank2 < rank1:
                out.append(copy.deepcopy(p2_keep[j])); j += 1
            else:
                # same rank (incl. same root): prefer the owning parent for that root
                if owner.get(r1) == "P1":
                    out.append(copy.deepcopy(p1_keep[i])); i += 1
                else:
                    out.append(copy.deepcopy(p2_keep[j])); j += 1

        return out

    def _select_oxk_root_segments(
        self,
        order_p1: list[str],
        order_p2: list[str],
        k: int,
    ) -> tuple[list[str], set[int]]:
        """Select root segments from *order_p1* using OX-k and return the
        resulting child root order along with the positions seeded from
        parent 1."""
        n = len(order_p1)
        if n >= 2:
            k = min(k, n // 2) if k > 0 else 1
            cuts = sorted(random.sample(range(n), 2 * k))
        else:
            k = 1
            cuts = [0, 0]

        child_roots: list[str | None] = [None] * n
        chosen: set[str] = set()
        segment_positions: set[int] = set()

        for i in range(0, len(cuts), 2):
            a, b = cuts[i], cuts[i + 1]
            for pos in range(a, b + 1):
                root = order_p1[pos]
                child_roots[pos] = root
                chosen.add(root)
                segment_positions.add(pos)

        it2 = (root for root in order_p2 if root not in chosen)
        for i in range(n):
            if child_roots[i] is None:
                try:
                    child_roots[i] = next(it2)
                except StopIteration:
                    remaining = [r for r in order_p2 if r not in set(child_roots)]
                    child_roots[i] = remaining[0] if remaining else order_p1[i]

        assert all(r is not None for r in child_roots), "OX-k produced empty slots."
        return [r for r in child_roots], segment_positions

    def _propagate_root_ownership(
        self,
        groups: list[dict],
        order_p1: list[str],
        segment_positions: set[int],
    ) -> dict[str, str | None]:
        """Propagate root ownership across differing-parameter groups."""
        owner: dict[str, str | None] = {r: None for r in order_p1}
        for pos in segment_positions:
            owner[order_p1[pos]] = "P1"

        seed_roots = {r for r, v in owner.items() if v == "P1"}
        if groups and seed_roots:
            adj: dict[str, set[str]] = {r: set() for r in order_p1}
            for g in groups:
                rs = list(g["roots"])
                if len(rs) > 1:
                    hub = rs[0]
                    for r in rs[1:]:
                        adj[hub].add(r)
                        adj[r].add(hub)
            q = deque(seed_roots)
            seen = set(seed_roots)
            while q:
                r = q.popleft()
                owner[r] = "P1"
                for nb in adj[r]:
                    if nb not in seen:
                        seen.add(nb)
                        q.append(nb)

        for g in groups:
            if all(owner[r] is None for r in g["roots"]):
                for r in g["roots"]:
                    owner[r] = "P2"

        roots_in_groups = set().union(*(g["roots"] for g in groups)) if groups else set()
        for pos, r in enumerate(order_p1):
            if owner[r] is None and r not in roots_in_groups:
                owner[r] = "P1" if pos in segment_positions else "P2"

        for g in groups:
            assert len({owner[r] for r in g["roots"]}) == 1, "Mixed ownership in a differing-parameter group."
        return owner

    def _merge_design_params(
        self,
        owner: dict[str, str | None],
        groups: list[dict],
        dp1: JsonDict | None,
        dp2: JsonDict | None,
    ) -> JsonDict | None:
        """Merge design parameters from parents based on root ownership."""
        child_dp = copy.deepcopy(dp1) if dp1 is not None else None
        if child_dp is not None and dp2 is not None and groups:
            for g in groups:
                g_owner = next(iter({owner[r] for r in g["roots"]}))
                if g_owner == "P2":
                    for path in g["paths"]:
                        try:
                            node = nested_get(dp2, path.split("."))
                        except Exception:
                            node = None
                        if node is None:
                            try:
                                nested_del(child_dp, path.split("."))
                            except Exception:
                                pass
                        else:
                            nested_set(child_dp, path.split("."), copy.deepcopy(node))
        return child_dp

    def _propagate_split_history(
        self,
        owner: dict[str, str | None],
        p1_hist: list,
        p2_hist: list,
        order: list[str],
    ) -> list:
        """Return split history entries from the owning parent for each root."""
        p1_hist_by_root = self._history_by_root(p1_hist)
        p2_hist_by_root = self._history_by_root(p2_hist)
        seen_entries: set[tuple[str, float]] = set()
        out: list = []
        for r in order:
            src = p1_hist_by_root if owner[r] == "P1" else p2_hist_by_root
            for entry in src.get(r, []):
                if entry not in seen_entries:
                    seen_entries.add(entry)
                    out.append(entry)
        return out

    # ------------------------- OX-k crossover (with weave) -------------------------

    def crossover_oxk(self, other: "Chromosome", k: int = 1) -> tuple["Chromosome", "Chromosome"]:
        """OX-k crossover over root order producing two complementary children.

        Child 1 uses the propagated ownership map; Child 2 uses the flipped map.
        Both children are built by the owner-stable weave, preserving non-contiguous
        families and taking leaves only from the owning parent for each root.
        """
        order_p1 = self._root_order()
        order_p2 = other._root_order()

        if not order_p1 and not order_p2:
            raise RuntimeError("Both parents have no roots (empty chromosomes).")

        if set(order_p1) != set(order_p2):
            raise RuntimeError("Parents must contain identical root sets (pre-split panels).")

        child_roots, segment_positions = self._select_oxk_root_segments(order_p1, order_p2, k)

        groups = self._conflict_groups(self.design_params, other.design_params, set(order_p1))
        owner = self._propagate_root_ownership(groups, order_p1, segment_positions)

        # Build first child using ownership map
        child1_genes: list[Piece] = self._build_child_by_owner_weave(
            order_child_roots=child_roots,
            owner=owner,
            p1_seq=self.genes,
            p2_seq=other.genes,
        )
        # Build second child using flipped ownership
        owner_flip = {r: ("P2" if ow == "P1" else "P1") for r, ow in owner.items()}
        child2_genes: list[Piece] = self._build_child_by_owner_weave(
            order_child_roots=child_roots,
            owner=owner_flip,
            p1_seq=self.genes,
            p2_seq=other.genes,
        )

        # Validate roots for both children
        expect_roots = set(order_p1)
        for tag, genes in (("child1", child1_genes), ("child2", child2_genes)):
            got = {g.root_id for g in genes}
            if got != expect_roots:
                raise RuntimeError(f"Root mismatch in {tag}: expected {sorted(expect_roots)}, got {sorted(got)}")

        p1_leaf_ids = self._leaf_ids(self.genes)
        p2_leaf_ids = self._leaf_ids(other.genes)
        for tag, genes in (("child1", child1_genes), ("child2", child2_genes)):
            child_leaf_ids = self._leaf_ids(genes)
            for r in order_p1:
                cl = child_leaf_ids.get(r, set())
                if cl != p1_leaf_ids.get(r, set()) and cl != p2_leaf_ids.get(r, set()):
                    raise RuntimeError(
                        f"Root {r} leaf mismatch in {tag}: child={sorted(cl)} "
                        f"not equal to P1={sorted(p1_leaf_ids.get(r, set()))} "
                        f"nor P2={sorted(p2_leaf_ids.get(r, set()))}"
                    )

        child1_dp = self._merge_design_params(owner, groups, self.design_params, other.design_params)
        child2_dp = self._merge_design_params(owner_flip, groups, self.design_params, other.design_params)

        child1 = Chromosome(
            pieces=child1_genes,
            container=self.container,
            origin="crossover",
            design_params=child1_dp,
            body_params=self.body_params,
            initial_design_params=self.initial_design_params,
        )
        child2 = Chromosome(
            pieces=child2_genes,
            container=self.container,
            origin="crossover",
            design_params=child2_dp,
            body_params=self.body_params,
            initial_design_params=self.initial_design_params,
        )

        child1.split_history = self._propagate_split_history(
            owner,
            getattr(self, "split_history", []),
            getattr(other, "split_history", []),
            order_p1,
        )
        child2.split_history = self._propagate_split_history(
            owner_flip,
            getattr(self, "split_history", []),
            getattr(other, "split_history", []),
            order_p1,
        )

        for ch in (child1, child2):
            try:
                ch.calculate_fitness()
            except Exception:
                pass

        import nesting.config as config
        for ch in (child1, child2):
            try:
                if config.FORCE_MUTATION_ON_CROSSOVER and (ch.fitness in (self.fitness, other.fitness)):
                    before = ch.fitness
                    ch.mutate()
                    ch.calculate_fitness()
                    if getattr(config, "VERBOSE", False):
                        print(f"[DEBUG] Forced mutation: {before} -> {ch.fitness}")
            except Exception:
                pass

        for ch in (child1, child2):
            ch._warn_if_panel_lost(set(order_p1), "crossover_oxk")

        return child1, child2


    # --------------------- Cross-Stitch OX ---------------------

    def _clash_sticky(
        self,
        out: list["Piece"],
        p1_seq: list["Piece"],
        p2_seq: list["Piece"],
        i: int,
        j: int,
        i1: int | None,
        j1: int | None,
    ) -> tuple[int, int]:
        assert i1 is not None
        while i < i1:
            out.append(copy.deepcopy(p1_seq[i]))
            i += 1
        return i, j

    def _clash_lexicographic(
        self,
        out: list["Piece"],
        p1_seq: list["Piece"],
        p2_seq: list["Piece"],
        i: int,
        j: int,
        i1: int | None,
        j1: int | None,
    ) -> tuple[int, int]:
        out.append(copy.deepcopy(p1_seq[i]))
        i += 1
        out.append(copy.deepcopy(p2_seq[j]))
        j += 1
        return i, j

    def _weave(
        self,
        owner: dict[str, str],
        p1_seq: list["Piece"],
        p2_seq: list["Piece"],
        clash_handler: Callable[..., tuple[int, int]],
    ) -> list["Piece"]:
        out: list["Piece"] = []
        i, j = 0, 0
        i0_init, _ = _next_owned_run(p1_seq, owner, "P1", 0)
        j0_init, _ = _next_owned_run(p2_seq, owner, "P2", 0)
        if i0_init is None and j0_init is None:
            return out
        if i0_init is None:
            side = "P2"; j = 0 if j0_init is None else j0_init
        elif j0_init is None:
            side = "P1"; i = 0 if i0_init is None else i0_init
        else:
            if i0_init <= j0_init:
                side = "P1"; i = i0_init
            else:
                side = "P2"; j = j0_init

        while True:
            i0, i1 = _next_owned_run(p1_seq, owner, "P1", i)
            j0, j1 = _next_owned_run(p2_seq, owner, "P2", j)

            p1_ready = i0 is not None and i0 == i
            p2_ready = j0 is not None and j0 == j

            if i0 is None and j0 is None:
                break

            if p1_ready and p2_ready:
                i, j = clash_handler(out, p1_seq, p2_seq, i, j, i1, j1)
                continue

            if side == "P1":
                if p1_ready:
                    assert i1 is not None
                    while i < i1:
                        out.append(copy.deepcopy(p1_seq[i]))
                        i += 1
                    continue
                side = "P2"
                if j0 is not None and j < j0:
                    j = j0
                continue
            else:
                if p2_ready:
                    assert j1 is not None
                    while j < j1:
                        out.append(copy.deepcopy(p2_seq[j]))
                        j += 1
                    continue
                side = "P1"
                if i0 is not None and i < i0:
                    i = i0
                continue
        return out

    def _weave_sticky(self, owner, p1_seq, p2_seq):
        return self._weave(owner, p1_seq, p2_seq, self._clash_sticky)

    def _weave_lexicographic(self, owner, p1_seq, p2_seq):
        return self._weave(owner, p1_seq, p2_seq, self._clash_lexicographic)

    def cross_stitch(
        self,
        owner: dict[str, str],
        p1_seq: list["Piece"],
        p2_seq: list["Piece"],
        *,
        mode: str = "sticky",
    ) -> list["Piece"]:
        """Pointer-based weave with clash tie-break strategies."""
        strategies = {
            "sticky": self._weave_sticky,
            "lexicographic": self._weave_lexicographic,
        }
        try:
            weave = strategies[mode]
        except KeyError:
            raise ValueError(f"Unknown cross_stitch mode: {mode}")
        return weave(owner, p1_seq, p2_seq)

    def cross_stitch_oxk(
        self,
        other: "Chromosome",
        *,
        k: int = 1,
        mode: str = "sticky",
    ) -> tuple["Chromosome", "Chromosome"]:
        """Perform an OX-k cross-stitch producing two complementary children."""

        order_p1 = self._root_order()
        order_p2 = other._root_order()
        if set(order_p1) != set(order_p2):
            raise RuntimeError("Parents must contain identical root sets (pre-split panels).")

        _, segment_positions = self._select_oxk_root_segments(order_p1, order_p2, k)
        groups = self._conflict_groups(self.design_params, other.design_params, set(order_p1))
        owner = self._propagate_root_ownership(groups, order_p1, segment_positions)

        child1_genes = self.cross_stitch(owner, self.genes, other.genes, mode=mode)
        owner_flip = {r: ("P2" if ow == "P1" else "P1") for r, ow in owner.items()}
        child2_genes = self.cross_stitch(owner_flip, self.genes, other.genes, mode=mode)

        set_roots = set(order_p1)
        child1_rootset = {g.root_id for g in child1_genes}
        child2_rootset = {g.root_id for g in child2_genes}
        if child1_rootset != set_roots or child2_rootset != set_roots:
            raise RuntimeError("Child lost or gained roots unexpectedly.")

        p1_leaf_ids = self._leaf_ids(self.genes)
        p2_leaf_ids = self._leaf_ids(other.genes)
        for ch_genes in (child1_genes, child2_genes):
            child_leaf_ids = self._leaf_ids(ch_genes)
            for r in set_roots:
                cl = child_leaf_ids.get(r, set())
                if cl != p1_leaf_ids.get(r, set()) and cl != p2_leaf_ids.get(r, set()):
                    raise RuntimeError(f"Root {r} leaf set in child differs from both parents.")

        child1_dp = self._merge_design_params(owner, groups, self.design_params, other.design_params)
        child2_dp = self._merge_design_params(owner_flip, groups, self.design_params, other.design_params)

        child1 = Chromosome(
            pieces=child1_genes,
            container=self.container,
            origin="crossover",
            design_params=child1_dp,
            body_params=self.body_params,
            initial_design_params=self.initial_design_params,
        )
        child2 = Chromosome(
            pieces=child2_genes,
            container=self.container,
            origin="crossover",
            design_params=child2_dp,
            body_params=self.body_params,
            initial_design_params=self.initial_design_params,
        )

        child1.split_history = self._propagate_split_history(
            owner,
            getattr(self, "split_history", []),
            getattr(other, "split_history", []),
            order_p1,
        )
        child2.split_history = self._propagate_split_history(
            owner_flip,
            getattr(self, "split_history", []),
            getattr(other, "split_history", []),
            order_p1,
        )

        for ch in (child1, child2):
            try:
                ch.calculate_fitness()
            except Exception:
                pass

        import nesting.config as config
        for ch in (child1, child2):
            try:
                if config.FORCE_MUTATION_ON_CROSSOVER and (ch.fitness in (self.fitness, other.fitness)):
                    before = ch.fitness
                    ch.mutate()
                    ch.calculate_fitness()
                    if getattr(config, "VERBOSE", False):
                        print(f"[DEBUG] Forced mutation: {before} -> {ch.fitness}")
            except Exception:
                pass

        for ch in (child1, child2):
            ch._warn_if_panel_lost(set_roots, "cross_stitch_oxk")

        return child1, child2


    def _signature(self) -> tuple:
        """
        An immutable fingerprint including:
        - Gene signature: ((id, rotation), …) in gene order
        - Design parameters hash (if available)
        - Parent IDs for tracking lineage (if available)
        """
        # Get signature based on genes
        gene_signature = tuple((p.id, p.rotation) for p in self.genes)
        design_params_hash = None
        if self.design_params is not None:
            try:
                design_params_str = json.dumps(self.design_params, sort_keys=True)
                design_params_hash = hash(design_params_str)
            except Exception:
                design_params_hash = None
        return (gene_signature, design_params_hash)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Chromosome):
            return NotImplemented
        return self._signature() == other._signature()

    def __repr__(self) -> str:
        sig = self._signature()
        return f"Chromosome(genes={sig[0]}, design_params_hash={sig[1]})"
