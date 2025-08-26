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

    def _mutate_split(self):
        #from nesting.panel_mapping import dispatch_split

        before = {g.root_id for g in self.genes}

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

        self._warn_if_panel_lost(before, "split")
        return True

    def _mutate_rotate(self):
        before = {g.root_id for g in self.genes}
        for _ in range(random.randint(1, len(self.genes))):
            idx = random.randrange(len(self.genes))
            self.genes[idx].rotate(random.choice(config.ALLOWED_ROTATIONS))
        self._warn_if_panel_lost(before, "rotate")

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

        before = {g.root_id for g in self.genes}

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

        self._warn_if_panel_lost(before, "swap")

    def _mutate_inversion(self):
        before = {g.root_id for g in self.genes}
        i, j = sorted(random.sample(range(len(self.genes)), 2))
        self.genes[i:j + 1] = reversed(self.genes[i:j + 1])
        self._warn_if_panel_lost(before, "inversion")

    def _mutate_insertion(self, k = 1):
        if len(self.genes) < 2:
            return
        
        n = len(self.genes)

        if k > n:
            raise ValueError("Invalid insertion operation: k too big for n")
        
        before = {g.root_id for g in self.genes}

        
        i = random.randrange(len(self.genes))
        insert_at = random.randrange(len(self.genes) + 1)
        gene = self.genes.pop(i)
        self.genes.insert(insert_at, gene)
        self._warn_if_panel_lost(before, "insertion")

    def _mutate_scramble(self):
        before = {g.root_id for g in self.genes}
        i, j = sorted(random.sample(range(len(self.genes)), 2))
        subset = self.genes[i:j + 1]
        random.shuffle(subset)
        self.genes[i:j + 1] = subset
        self._warn_if_panel_lost(before, "scramble")

    # ── design‑parameter mutation ──────────────────────────

    def _mutate_design_params(self):
        if not (self.design_params and self.body_params):
            if config.VERBOSE:
                print("[Chromosome] design‑param mutation skipped - missing design or body params")
            return

        before = {g.root_id for g in self.genes}

        # Ensure fitness is calculated before attempting mutation
        if self.fitness is None:
            self.calculate_fitness()
        original_fitness = self.fitness

        # Backup original state
        original_design_params = copy.deepcopy(self.design_params)
        original_genes = copy.deepcopy(self.genes)

        # Somewhat weird way of doing things incoming
        # It works and I am terrified of breaking it so for now it stays
        # TODOLOW: Refactor this to be less weird

        # Get a shuffled list of parameters to try
        mutatable_params = list(self._mutatable_params)
        random.shuffle(mutatable_params)

        # loop until we find a successful mutation or exhaust all options
        for param in mutatable_params:
            # Restore state for the new attempt
            self.design_params = copy.deepcopy(original_design_params)
            self.genes = copy.deepcopy(original_genes)

            node = nested_get(self.design_params, param.split("."))
            p_type = node.get("type", "float")
            old_val = node["v"]
            base_val = nested_get(self.initial_design_params, param.split("."))["v"]
            new_val = _random_value(old_val, base_val, p_type, node["range"])

            if old_val == new_val:
                continue # Value didn't change, try next parameter

            nested_set(self.design_params, param.split(".") + ["v"], new_val)

            # Regenerate garment and update pieces
            if self._apply_design_param_change(param, old_val, new_val):
                self.calculate_fitness() # Recalculate fitness with the new design
                if self.fitness != original_fitness:
                    print(f"[Chromosome] Successful mutation on {param}: {old_val} -> {new_val}")
                    # Store this specific parameter change for tracking
                    if not hasattr(self, 'param_changes_this_gen'):
                        self.param_changes_this_gen = []
                    self.param_changes_this_gen.append({
                        'param_path': param,
                        'old_value': old_val,
                        'new_value': new_val
                    })
                    self._warn_if_panel_lost(before, "design_params_1")
                    return True # Success

        # If loop completes, no mutation led to a fitness change, restore state
        self.design_params = original_design_params
        self.genes = original_genes
        self.fitness = original_fitness
        if config.VERBOSE:
            print("[Chromosome] No design param mutation resulted in a fitness change.")
        self._warn_if_panel_lost(before, "design_params_2")
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

        # TODO: use a logger instead of this
        if config.LOG_DESIGN_PARAM_PATHS and changed_ids:
            log_path = Path(config.SAVE_LOGS_PATH) / "design_param_paths.csv"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", newline="") as fh:
                writer = csv.writer(fh)
                if fh.tell() == 0:
                    writer.writerow(["piece_id", "param_path", "old_v", "new_v"])
                for pid in extended_changed_ids:
                    if pid in new_piece_ids:  # only log those we actually refreshed
                        writer.writerow([pid, path, old_val, new_val])
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

    # ------------------------- OX-k crossover (with weave) -------------------------

    def crossover_oxk(self, other: "Chromosome", k: int = 1) -> "Chromosome":
        """
        OX-k over *root order* with param-group closure & family integrity,
        preserving non-contiguous interleaving via an owner-stable weave.

        Invariants enforced:
        - Every root appears in child (no loss).
        - For each root, the child's *leaf set* equals exactly one parent's leaf set (no mixing within a root).
        - For any differing-parameter group, *all* its roots come from a single parent.
        - split_history kept only from the owning parent per root.
        """
        import random
        from copy import deepcopy
        from pygarment.garmentcode.utils import nested_set, nested_get, nested_del

        # --- Root orders
        order_p1 = self._root_order()
        order_p2 = other._root_order()

        if not order_p1 and not order_p2:
            raise RuntimeError("Both parents have no roots (empty chromosomes).")
            # return Chromosome([], self.container, origin="crossover",
            #                   design_params=deepcopy(self.design_params),
            #                   body_params=self.body_params)

        # must share the same root set (leaf counts may differ due to splits)
        if set(order_p1) != set(order_p2):
            raise RuntimeError("Parents must contain identical root sets (pre-split panels).")

        n = len(order_p1)

        # --- Step 1: True OX-k on parent-1 root order (to produce child root order)
        if n >= 2:
            # k = max(1, min(k, n // 2))
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
                #if child_roots[pos] is None:
                child_roots[pos] = root
                chosen.add(root)
                segment_positions.add(pos)

        # Fill remaining slots with roots from P2, skipping already chosen ones
        it2 = (root for root in order_p2 if root not in chosen)
        for i in range(n):
            if child_roots[i] is None:
                try:
                    child_roots[i] = next(it2)
                except StopIteration:
                    remaining = [root for root in order_p2 if root not in set(child_roots)]
                    child_roots[i] = remaining[0] if remaining else order_p1[i]

        assert all(r is not None for r in child_roots), "OX-k produced empty slots."

        # --- Step 2: Differing-parameter groups & ownership (closure)
        groups = self._conflict_groups(self.design_params, other.design_params, set(order_p1))

        # owner[root] ∈ {None, "P1", "P2"}
        owner: dict[str, str | None] = {r: None for r in order_p1}

        # seed from P1 segments
        for pos in segment_positions:
            owner[order_p1[pos]] = "P1"

        # --- TRANSITIVE closure over overlapping differing groups ---
        # Build a lightweight adjacency: connect roots that appear together in a differing group.
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
            # BFS from all seed roots to pull entire connected components to owner=P1
            from collections import deque  # local import (already available globally, kept for clarity)
            q = deque(seed_roots)
            seen = set(seed_roots)
            while q:
                r = q.popleft()
                owner[r] = "P1"
                for nb in adj[r]:
                    if nb not in seen:
                        seen.add(nb)
                        q.append(nb)

        # remaining undecided groups: single owner (default P2)
        for g in groups:
            if all(owner[r] is None for r in g["roots"]):
                for r in g["roots"]:
                    owner[r] = "P2"

        # roots not in any differing group: default by segment position
        roots_in_groups = set().union(*(g["roots"] for g in groups)) if groups else set()
        for pos, r in enumerate(order_p1):
            if owner[r] is None and r not in roots_in_groups:
                owner[r] = "P1" if pos in segment_positions else "P2"

        # no mixed ownership inside any differing group
        for g in groups:
            assert len({owner[r] for r in g["roots"]}) == 1, "Mixed ownership in a differing-parameter group."

        # --- Step 3: Build child by owner-stable weave (preserve non-contiguous patterns)
        child_genes: list[Piece] = self._build_child_by_owner_weave(
            order_child_roots=[r for r in child_roots if r is not None],
            owner=owner,
            p1_seq=self.genes,
            p2_seq=other.genes,
        )

        # --- Step 4: Post-build assertions
        # A) no root lost
        child_roots_set = {g.root_id for g in child_genes}
        if child_roots_set != set(order_p1):
            raise RuntimeError(f"Root mismatch: expected {sorted(order_p1)}, got {sorted(child_roots_set)}")

        # B) each root's leaves equal exactly one parent's leaf set
        # collect per-root leaf IDs from parents (only leaves: id not in parent_ids)
        def _leaf_ids(seq: list["Piece"]) -> dict[str, set[str]]:
            parent_ids = {p.parent_id for p in seq if p.parent_id}
            out: dict[str, set[str]] = {}
            for p in seq:
                if p.id not in parent_ids:
                    out.setdefault(p.root_id, set()).add(p.id)
            return out

        p1_leaf_ids = _leaf_ids(self.genes)
        p2_leaf_ids = _leaf_ids(other.genes)
        child_leaf_ids = _leaf_ids(child_genes)

        for r in order_p1:
            cl = child_leaf_ids.get(r, set())
            if cl != p1_leaf_ids.get(r, set()) and cl != p2_leaf_ids.get(r, set()):
                raise RuntimeError(
                    f"Root {r} leaf mismatch: child={sorted(cl)} "
                    f"not equal to P1={sorted(p1_leaf_ids.get(r, set()))} "
                    f"nor P2={sorted(p2_leaf_ids.get(r, set()))}"
                )

        # --- Step 5: Merge design_params per owning group
        child_dp = deepcopy(self.design_params) if self.design_params is not None else None
        if child_dp is not None and other.design_params is not None and groups:
            for g in groups:
                g_owner = next(iter({owner[r] for r in g["roots"]}))  # uniform by assertion
                if g_owner == "P2":
                    for path in g["paths"]:
                        node = None
                        try:
                            node = nested_get(other.design_params, path.split("."))
                        except Exception:
                            node = None
                        if node is None:
                            try:
                                nested_del(child_dp, path.split("."))
                            except Exception:
                                pass
                        else:
                            nested_set(child_dp, path.split("."), deepcopy(node))
                # if g_owner == 1, P1 values already in child_dp via deepcopy

        # --- Step 6: Build child & propagate split_history relevant to the child
        child = Chromosome(
            pieces=child_genes,
            container=self.container,
            origin="crossover",
            design_params=child_dp,
            body_params=self.body_params,
            initial_design_params=self.initial_design_params,
        )

        p1_hist_by_root = self._history_by_root(getattr(self, "split_history", []))
        p2_hist_by_root = self._history_by_root(getattr(other, "split_history", []))

        seen_entries: set[tuple[str, float]] = set()
        child.split_history = []
        for r in order_p1:
            src = p1_hist_by_root if owner[r] == "P1" else p2_hist_by_root
            for entry in src.get(r, []):
                if entry not in seen_entries:
                    seen_entries.add(entry)
                    child.split_history.append(entry)

        # optional: force mutation if configured and child fitness equals a parent
        child.calculate_fitness()
        import nesting.config as config
        # force = getattr(config, "FORCE_MUTATION_ON_CROSSOVER", False)
        if config.FORCE_MUTATION_ON_CROSSOVER and (child.fitness in (self.fitness, other.fitness)):
            before = child.fitness
            child.mutate()
            child.calculate_fitness()
            if getattr(config, "VERBOSE", False):
                print(f"[DEBUG] Forced mutation: {before} -> {child.fitness}")

        # Warn if any root/panel was lost compared to the parents' root set
        child._warn_if_panel_lost(set(order_p1), "crossover_oxk")

        return child


    # --------------------- Cross-Stitch OX ---------------------

    def cross_stitch(
        self,
        owner: dict[str, str],
        p1_seq: list["Piece"],
        p2_seq: list["Piece"],
        *,
        mode: str = "sticky",
    ) -> list["Piece"]:
        """
        Pointer-based weave with clash tie-break modes.

        - Only append genes whose root owner matches that parent.
        - Clash occurs when both sides' next item is owned by that side.
          Tie-break:
            sticky:   emit full run from P1 until gap then continue.
            lexicographic: emit P1[i], then P2[j], then continue normal logic.
        """

        def next_owned_run(seq: list["Piece"], take_owner: str, start: int) -> tuple[int | None, int | None]:
            # find first index >= start that belongs to take_owner
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

        out: list["Piece"] = []
        i, j = 0, 0
        # n1, n2 = len(p1_seq), len(p2_seq)
        # choose initial side by earliest first-owned index; tie -> P1
        i0_init, _ = next_owned_run(p1_seq, "P1", 0)
        j0_init, _ = next_owned_run(p2_seq, "P2", 0)
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
            i0, i1 = next_owned_run(p1_seq, "P1", i)
            j0, j1 = next_owned_run(p2_seq, "P2", j)

            p1_ready = i0 is not None and i0 == i
            p2_ready = j0 is not None and j0 == j

            # no more owned genes on either side -> done
            if (i0 is None) and (j0 is None):
                break

            # clash: both sides ready at current pointers
            if p1_ready and p2_ready:
                if mode == "lexicographic":
                    # emit one from P1 then one from P2
                    out.append(copy.deepcopy(p1_seq[i])); i += 1
                    out.append(copy.deepcopy(p2_seq[j])); j += 1
                    # continue without forcing side; recompute next
                    continue
                else:  # sticky
                    # emit the maximal run from P1, then continue
                    assert i1 is not None
                    while i < i1:
                        out.append(copy.deepcopy(p1_seq[i])); i += 1
                    # after finishing run, side naturally changes according to availability
                    continue

            # non-clash: follow the active side greedily
            if side == "P1":
                if p1_ready:
                    assert i1 is not None
                    while i < i1:
                        out.append(copy.deepcopy(p1_seq[i])); i += 1
                    # keep side=P1 until next gap; loop will recompute
                    continue
                # gap on P1: switch to P2 and jump to its next owned start to avoid oscillation
                side = "P2"
                if j0 is not None and j < j0:
                    j = j0
                continue
            else:  # side == "P2"
                if p2_ready:
                    assert j1 is not None
                    while j < j1:
                        out.append(copy.deepcopy(p2_seq[j])); j += 1
                    # keep side=P2 until next gap
                    continue
                # gap on P2: switch to P1 and jump to its next owned start
                side = "P1"
                if i0 is not None and i < i0:
                    i = i0
                continue

        return out

    def cross_stitch_oxk(
        self,
        other: "Chromosome",
        *,
        k: int = 1,
        mode: str = "sticky",
    ) -> tuple["Chromosome", "Chromosome"]:
        """
        OX-k variant over gene indices with split-root closure and design-parameter closure.

        Steps:
        1) Select k disjoint contiguous segments on parent1's gene sequence.
           Seed ownership: any root appearing in these segments is owned by P1 (root closure).
        2) DP closure: compute differing design parameters; for each param, find affected roots.
           Build an adjacency among roots that co-occur in any differing group. Propagate ownership
           from the seed roots through this graph (BFS). Unassigned roots default to P2.
        3) Weave: build child sequence using a pointer-based weave with clash handling.
           mode in {"sticky", "lexicographic"} controls tie-break behavior.
        4) Build complementary child by flipping owners and re-weaving with the same mode.

        Returns (child_primary, child_complementary).
        """
        # Preconditions: parents must contain identical root sets (pre-split identifiers)
        p1_roots = [g.root_id for g in self.genes]
        p2_roots = [g.root_id for g in other.genes]
        set1, set2 = set(p1_roots), set(p2_roots)
        if set1 != set2:
            raise RuntimeError("Parents must contain identical root sets (pre-split panels).")

        # 1) Choose k random gene segments on parent1 and seed P1 ownership via root closure
        n = len(self.genes)
        if n == 0:
            raise RuntimeError("Empty parent genes.")
        if n == 1:
            k = 1
        else:
            k = max(1, min(k, n // 2))

        cuts = sorted(random.sample(range(n), 2 * k)) if n > 1 else [0, 0]
        segs: list[tuple[int, int]] = []
        for i in range(0, len(cuts), 2):
            a, b = cuts[i], cuts[i + 1]
            if a > b:
                a, b = b, a
            segs.append((a, b))  # inclusive

        # Seed ownership
        owner: dict[str, str | None] = {r: None for r in set1}
        for a, b in segs:
            for idx in range(a, b + 1):
                r = self.genes[idx].root_id
                owner[r] = "P1"

        # 2) DP closure (grouped by differing params) and BFS propagation
        groups = self._conflict_groups(self.design_params, other.design_params, set1)
        if groups:
            # adjacency among roots that appear together in a group
            adj: dict[str, set[str]] = {r: set() for r in set1}
            for g in groups:
                rs = list(g["roots"])  # set[str]
                for u in range(len(rs)):
                    for v in range(u + 1, len(rs)):
                        a, b = rs[u], rs[v]
                        adj[a].add(b)
                        adj[b].add(a)

            # BFS from seed roots owned by P1
            from collections import deque
            seeds = deque([r for r, ow in owner.items() if ow == "P1"])
            seen = set(seeds)
            while seeds:
                cur = seeds.popleft()
                owner[cur] = "P1"
                for nb in adj.get(cur, ()):  # pull whole connected component to P1
                    if nb not in seen:
                        seen.add(nb)
                        seeds.append(nb)

            # Any fully-unassigned differing groups go to P2 wholesale
            for g in groups:
                rs = g["roots"]
                if all(owner[r] is None for r in rs):
                    for r in rs:
                        owner[r] = "P2"

        # Assign any remaining undecided roots to P2
        for r in set1:
            if owner[r] is None:
                owner[r] = "P2"

        # 3) Build child via pointer-based weave
        child_genes = self.cross_stitch(owner, self.genes, other.genes, mode=mode)

        # 3b) Post invariants: root set and leaf ownership integrity
        child_roots_set = {g.root_id for g in child_genes}
        if child_roots_set != set1:
            raise RuntimeError("Child lost or gained roots unexpectedly.")

        def _leaf_ids(seq: list["Piece"]) -> dict[str, set[str]]:
            parent_ids = {p.parent_id for p in seq if p.parent_id}
            out: dict[str, set[str]] = {}
            for p in seq:
                if p.id not in parent_ids:
                    out.setdefault(p.root_id, set()).add(p.id)
            return out

        p1_leaf_ids = _leaf_ids(self.genes)
        p2_leaf_ids = _leaf_ids(other.genes)
        child_leaf_ids = _leaf_ids(child_genes)
        for r in set1:
            cl = child_leaf_ids.get(r, set())
            if cl != p1_leaf_ids.get(r, set()) and cl != p2_leaf_ids.get(r, set()):
                raise RuntimeError(f"Root {r} leaf set in child differs from both parents.")

        # 4) Build complementary owner map and weave again
        owner_flip = {r: ("P2" if ow == "P1" else "P1") for r, ow in owner.items()}
        child2_genes = self.cross_stitch(owner_flip, self.genes, other.genes, mode=mode)

        # 5) Merge design params per owning groups for both children
        def merge_dp(owner_map: dict[str, str]) -> dict | None:
            if self.design_params is None:
                return None
            dp = copy.deepcopy(self.design_params)
            if other.design_params is None:
                return dp
            if not groups:
                return dp
            from pygarment.garmentcode.utils import nested_get, nested_set, nested_del
            for g in groups:
                # owner is uniform per group by construction
                any_r = next(iter(g["roots"]))
                g_owner = owner_map[any_r]
                if g_owner == "P2":
                    for path in g["paths"]:
                        try:
                            node = nested_get(other.design_params, path.split("."))
                        except Exception:
                            node = None
                        if node is None:
                            try:
                                nested_del(dp, path.split("."))
                            except Exception:
                                pass
                        else:
                            nested_set(dp, path.split("."), copy.deepcopy(node))
            return dp

        child1_dp = merge_dp(owner)
        child2_dp = merge_dp(owner_flip)

        # 6) Construct Chromosome objects and propagate split histories per ownership
        def build_child(pieces: list["Piece"], owner_map: dict[str, str], dp: dict | None) -> "Chromosome":
            ch = Chromosome(
                pieces=pieces,
                container=self.container,
                origin="crossover",
                design_params=dp,
                body_params=self.body_params,
                initial_design_params=self.initial_design_params,
            )
            p1_hist_by_root = self._history_by_root(getattr(self, "split_history", []))
            p2_hist_by_root = self._history_by_root(getattr(other, "split_history", []))
            seen_entries: set[tuple[str, float]] = set()
            ch.split_history = []
            for r in set1:
                src = p1_hist_by_root if owner_map[r] == "P1" else p2_hist_by_root
                for entry in src.get(r, []):
                    if entry not in seen_entries:
                        seen_entries.add(entry)
                        ch.split_history.append(entry)
            # optional immediate fitness
            try:
                ch.calculate_fitness()
            except Exception:
                pass

            # Warn if any root/panel was lost compared to the parents' root set
            ch._warn_if_panel_lost(set1, "cross_stitch_oxk")
            return ch

        child1 = build_child(child_genes, owner, child1_dp)
        child2 = build_child(child2_genes, owner_flip, child2_dp)

        # Optional: force mutation if configured and child's fitness equals a parent's
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
                # do not derail crossover if fitness cannot be computed here
                pass

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