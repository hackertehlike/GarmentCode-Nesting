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


def _random_value(old: Any, p_type: str, rng: Sequence[float | int]):
    """Return a *new* value that differs from *old* by ≤ 20 % of *range* width."""
    
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
        candidates = [v for v in vals if abs(v - old) <= max_delta and v != old]
        if not candidates:
            return old  # no valid candidates, return old value
        return random.choice(candidates)

    # float parameters ----------------------------------------------------
    attempts = 0
    while attempts < 10:
        cand = random.uniform(lower, upper)
        if abs(cand - old) <= max_delta and cand != old:
            return cand
        attempts += 1
    # fallback – choose a value slightly different from old but within max_delta
    if old + max_delta <= upper:
        return old + max_delta / 2
    elif old - max_delta >= lower:
        return old - max_delta / 2
    else:
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

    # flatten
    all_paths = _flatten_param_paths(design_params)

    # numeric & not excluded
    patterns = config.EXCLUDED_PARAM_PATHS or []
    numeric_ok = [
        p for p in all_paths
        if _numeric_range_ok(nested_get(design_params, p.split(".")))
        and not any(fnmatch(p, pat) for pat in patterns)
    ]
    #print(f"[DEBUG] After numeric & exclusion filtering: {len(numeric_ok)} paths")
    
    # touches at least one panel in this chromo
    panel_ids = {g.id for g in genes}
    #print(f"[DEBUG] Panel IDs in chromosome: {panel_ids}")
    
    mutatable = []
    for p in numeric_ok:
        panel_patterns = affected_panels([p], design_params)
        matching_panels = [pid for pid in panel_ids if any(fnmatch(pid, pat) for pat in panel_patterns)]
        if matching_panels:
            mutatable.append(p)
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
    return dec.concave_hull_area()

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
        #design_sampler: "DesignSampler" | None = None,
    ):
        # Store a deep copy of each piece
        self._genes = [copy.deepcopy(p) for p in pieces]
        self.container = container
        self.fitness: float | None = None

        self.design_params = copy.deepcopy(design_params) if design_params else None
        self.body_params = body_params
        self._mutatable_params = _collect_mutatable_params(
            self.design_params, self._genes
        )

        # Track origin and last mutation type
        self.origin: str | None = origin
        self.last_mutation: str | None = None
        
        # # For backwards compatibility, if the origin is crossover, set last_mutation to crossover_params
        # if origin == "crossover":
        #     self.last_mutation = "crossover_params"
        
        # Initialize empty list to track parameter changes in the current generation
        self.param_changes_this_gen = []
        
        # Add tracking for mutation statistics
        self.old_fitness: float | None = None
        self.new_fitness: float | None = None
        self.mutation_improvement: float | None = None
        self.split_history: list[tuple[str, float]] = []

        # Keep a MetaGarment instance for incremental operations similar to the GUI
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
            print(f"[Chromosome] WARNING: mutation '{mutation}' lost panels: {sorted(missing)}")

    # ── simple mutations ───────────────────────────────────────────────
    def meta_split(self, piece: Piece, idx: int, piece_mirror: Piece | None = None) -> bool:
        import tempfile
        from nesting.path_extractor import PatternPathExtractor
        from assets.garment_programs.meta_garment import MetaGarment
        from pathlib import Path
        mg = self.meta_garment

        # choose split proportion and perform
        proportion = random.uniform(0.3, 0.7)
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
        """Replace a piece at index *idx* with *new_piece*."""
        idx = self.genes.index(piece)
        genes.remove(piece)
        # insert the first piece at the original index
        genes.insert(idx, new_piece_left)
        # insert the second piece at a random position
        genes.insert(random.randrange(len(genes)), new_piece_right)

        return genes

    def _mutate_split(self):
        """Split a panel using the MetaGarment pipeline, preserving state."""
        #from nesting.panel_mapping import dispatch_split
        

        before = {g.root_id for g in self.genes}

        self.split_set = getattr(self, 'split_set', set())

        max_splits = min(config.NUM_SPLITS, len(self.genes))
        for _ in range(random.randint(0, min(max_splits, len(self.genes)-len(self.split_set)))):
            #unsplit = [i for i, g in enumerate(self.genes) if g.parent_id is None]
            #if not unsplit:
            #    break

            #idx = random.choices(unsplit, weights=[self.genes[i].bbox_area for i in unsplit])[0]
            #idx = random.choices(range(len(self.genes)), weights=[g.bbox_area for g in self.genes])[0]
            idx = random.randrange(len(self.genes))
            piece = self.genes[idx]

            if piece in self.split_set:
                print(f"[Chromosome] Skipping split for {piece.id} - already split")
                continue
            else:
                self.split_set.add(piece)

            #retry until we find a piece that can be split
            # retries = 0
            # while 'split' in piece.id and retries < 10:
            #     print(f"[Chromosome] Skipping split for {piece.id} - already split")
            #     idx = random.randrange(len(self.genes))
            #     piece = self.genes[idx]
            #     retries += 1

            # if 'split' in piece.id:
            #     print(f"Max retries reached for split mutation on {piece.id}, skipping")
            #     continue

            # Ensure a MetaGarment instance is fresh for the split operation
        
            # mg = MetaGarment("chromosome_split", self.body_params, self.design_params)
            # Re-apply existing splits to bring the new instance to the current state
            #for panel_name, proportion in self.split_history:
            #    mg.split_panel(panel_name, proportion)
            # self.meta_garment = mg

            piece_mirror = next((p for p in self.genes if p.parent_id == piece.id.replace("left", "right") or p.parent_id == piece.id.replace("right", "left")), None)

            # None
            # if 'left' in piece.id and piece.parent_id is None:
            #     mirror_id = piece.id.replace("left", "right")
            #     piece_mirror = next((p for p in self.genes if p.id == mirror_id), None)
            # elif 'right' in piece.id and piece.parent_id is None:
            #     mirror_id = piece.id.replace("right", "left")
            #     piece_mirror = next((p for p in self.genes if p.id == mirror_id), None)


            if True:#not self.meta_garment:
                left, right = piece.split()
                # remove the original piece
                # self.genes.remove(piece)
                # # insert the first piece at the original index
                # self.genes.insert(idx, left)
                # # insert the second piece at a random position
                # self.genes.insert(random.randrange(len(self.genes)), right)
                self._replace(piece, left, right, self.genes)
                self.split_set.add(left)
                self.split_set.add(right)
                if piece_mirror and config.SYMMETRIC_SPLITS:
                    self.split_set.add(piece_mirror)
                    # If there's a mirror piece, also split it
                    # Find the current index of the mirror piece (may have shifted)
                    # current_mirror_idx = self.genes.index(piece_mirror)
                    left_mirror, right_mirror = piece_mirror.split()
                    self.genes = self._replace(piece_mirror, left_mirror, right_mirror, self.genes)
                    # # Remove the original mirror piece
                    # self.genes.remove(piece_mirror)
                    # # Insert the split pieces - first at the original mirror position
                    # self.genes.insert(current_mirror_idx, left_mirror)
                    # # Insert the second at a random position
                    # self.genes.insert(random.randrange(len(self.genes)), right_mirror)
                    self.split_set.add(left_mirror)
                    self.split_set.add(right_mirror)

                self._warn_if_panel_lost(before, "split")
                return True  # Indicate successful mutation
                
            else:
                # Use MetaGarment's split_panel pipeline
                success = self.meta_split(piece, idx, piece_mirror)
                if not success:
                    print(f"[Chromosome] MetaGarment split failed for {piece.id}")
                    continue
            self._warn_if_panel_lost(before, "split")
            return True

    # def _no_param_split(self, piece, idx, piece_mirror=None):

    #     left, right = piece.split()
    #     # remove the original piece
    #     self.genes.remove(piece)
    #     # insert the first piece at the original index
    #     self.genes.insert(idx, left)
    #     # insert the second piece at a random position
    #     self.genes.insert(random.randrange(len(self.genes) + 1), right)

    #     if piece_mirror and config.SYMMETRIC_SPLITS:
    #         # If there's a mirror piece, also split it
    #         # Find the current index of the mirror piece (may have shifted)
    #         current_mirror_idx = self.genes.index(piece_mirror)
    #         left_mirror, right_mirror = piece_mirror.split()
    #         # Remove the original mirror piece
    #         self.genes.remove(piece_mirror)
    #         # Insert the split pieces - first at the original mirror position
    #         self.genes.insert(current_mirror_idx, left_mirror)
    #         # Insert the second at a random position
    #         self.genes.insert(random.randrange(len(self.genes) + 1), right_mirror)

    #     self._warn_if_panel_lost(before, "split")
    #     return True  # Indicate successful mutation


    def _mutate_rotate(self):
        before = {g.root_id for g in self.genes}
        for _ in range(random.randint(1, len(self.genes))):
            idx = random.randrange(len(self.genes))
            self.genes[idx].rotate(random.choice(config.ALLOWED_ROTATIONS))
        self._warn_if_panel_lost(before, "rotate")

    def _mutate_swap(self):
        before = {g.root_id for g in self.genes}
        i, j = random.sample(range(len(self.genes)), 2)
        self.genes[i], self.genes[j] = self.genes[j], self.genes[i]
        self._warn_if_panel_lost(before, "swap")

    def _mutate_inversion(self):
        before = {g.root_id for g in self.genes}
        i, j = sorted(random.sample(range(len(self.genes)), 2))
        self.genes[i:j + 1] = reversed(self.genes[i:j + 1])
        self._warn_if_panel_lost(before, "inversion")

    def _mutate_insertion(self):
        if len(self.genes) < 2:
            return
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

        # Get a shuffled list of parameters to try
        mutatable_params = list(self._mutatable_params)
        random.shuffle(mutatable_params)

        for path in mutatable_params: # eh fuck it why not
            # Restore state for the new attempt
            self.design_params = copy.deepcopy(original_design_params)
            self._genes = copy.deepcopy(original_genes)

            node = nested_get(self.design_params, path.split("."))
            p_type = node.get("type", "float")
            old_val = node["v"]
            new_val = _random_value(old_val, p_type, node["range"])

            if old_val == new_val:
                continue # Value didn't change, try next parameter

            nested_set(self.design_params, path.split(".") + ["v"], new_val)

            # Regenerate garment and update pieces
            if self._apply_design_param_change(path, old_val, new_val):
                self.calculate_fitness() # Recalculate fitness with the new design
                if self.fitness != original_fitness:
                    print(f"[Chromosome] Successful mutation on {path}: {old_val} -> {new_val}")
                    # Store this specific parameter change for tracking
                    if not hasattr(self, 'param_changes_this_gen'):
                        self.param_changes_this_gen = []
                    self.param_changes_this_gen.append({
                        'param_path': path,
                        'old_value': old_val,
                        'new_value': new_val
                    })
                    self._warn_if_panel_lost(before, "design_params")
                    return True # Success

        # If loop completes, no mutation led to a fitness change, restore state
        self.design_params = original_design_params
        self._genes = original_genes
        self.fitness = original_fitness
        if config.VERBOSE:
            print("[Chromosome] No design param mutation resulted in a fitness change.")
        self._warn_if_panel_lost(before, "design_params")
        return False

    def _apply_design_param_change(self, path: str, old_val: Any, new_val: Any) -> bool:
        # TODO: fix split handling. it must happen before reassembly
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
        split_cache: dict[str, tuple[Piece, Piece]] = {}
        for i, g in enumerate(self.genes):
            if g.id in changed_ids:
                new_piece = copy.deepcopy(new_pieces[g.id])
                new_piece.rotation, new_piece.translation = g.rotation, g.translation
                self.genes[i] = new_piece
            elif g.root_id in changed_ids:
                if g.root_id not in split_cache:
                    base_piece = copy.deepcopy(new_pieces[g.root_id])
                    split_cache[g.root_id] = base_piece.split()
                left, right = split_cache[g.root_id]
                suffix = g.id[len(g.root_id) + 1:]
                replacement = copy.deepcopy(left if suffix == "split_left" else right)
                replacement.rotation, replacement.translation = g.rotation, g.translation
                self.genes[i] = replacement

        if config.LOG_DESIGN_PARAM_PATHS and changed_ids:
            log_path = Path(config.SAVE_LOGS_PATH) / "design_param_paths.csv"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", newline="") as fh:
                writer = csv.writer(fh)
                if fh.tell() == 0:
                    writer.writerow(["piece_id", "param_path", "old_v", "new_v"])
                for pid in changed_ids:
                    writer.writerow([pid, path, old_val, new_val])
        return True

    def crossover_oxk(
        self,
        other: "Chromosome",
        k: int = 1
    ) -> "Chromosome":
        """
        Perform an order-based crossover (OX-k) with constraints using a single pass.

        This crossover handles gene dependencies related to panel splitting.
        If a descendant of a split panel is selected from parent 1, all its
        sibling leaf-descendants are also taken from parent 1, and the
        corresponding original panel (root) is blocked from being taken from parent 2.
        """
        n = len(self.genes)
        child_genes: List[Piece | None] = [None] * n
        child_gene_ids: Set[str] = set()
        blocked_root_ids: Set[str] = set()

        # ----------------------------- helpers ---------------------------------
        def get_family_leaves(parent: "Chromosome", root_id: str) -> List[Piece]:
            parent_ids = {p.parent_id for p in parent.genes if p.parent_id}
            return [g for g in parent.genes if g.root_id == root_id and g.id not in parent_ids]

        # Index map: piece.id -> index in that parent's gene order
        p1_index = {g.id: i for i, g in enumerate(self.genes)}
        p2_index = {g.id: i for i, g in enumerate(other.genes)}

        free_indices = deque(range(n))  # always place something if we need to relocate

        def occupy(idx: int, piece: Piece):
            """Place piece at idx, relocating any occupant to the next free slot."""
            if child_genes[idx] is None:
                # remove idx from free list if present
                try:
                    free_indices.remove(idx)
                except ValueError:
                    pass
                child_genes[idx] = piece
                child_gene_ids.add(piece.id)
                return True
            else:
                # relocate existing occupant to next free slot
                if not free_indices:
                    return False  # should not happen if logic maintains counts
                new_idx = free_indices.popleft()
                # If new_idx equals idx (rare if idx was still listed), try another
                if new_idx == idx:
                    if not free_indices:
                        return False
                    new_idx = free_indices.popleft()
                displaced = child_genes[idx]
                child_genes[new_idx] = displaced
                # mark displaced id already accounted for
                # Place the desired piece at idx
                child_genes[idx] = piece
                child_gene_ids.add(piece.id)
                return True

        def place_family(parent: "Chromosome", root_id: str) -> bool:
            """Atomically place all leaf descendants of root_id using that parent's indices.
            If a placement collides, we relocate the occupant to next free slot.
            Only blocks the root after full success.
            """
            if root_id in blocked_root_ids:
                return False
            family = get_family_leaves(parent, root_id)
            if not family:
                return False
            # Preferred indices from the appropriate parent's ordering
            index_map = p1_index if parent is self else p2_index
            targets = [index_map[m.id] for m in family]

            # Stage pieces so we can roll back if something goes wrong (shouldn't)
            snapshot = list(child_genes)
            snapshot_free = deque(free_indices)

            try:
                for member, idx in zip(family, targets):
                    if member.id in child_gene_ids:
                        # already placed via a sibling/group earlier
                        continue
                    ok = occupy(idx, copy.deepcopy(member))
                    if not ok:
                        raise RuntimeError("No free slot available during family placement")
            except Exception:
                # rollback on failure
                for i in range(n):
                    child_genes[i] = snapshot[i]
                free_indices.clear()
                free_indices.extend(snapshot_free)
                return False

            # Success → block this root for the other parent
            blocked_root_ids.add(root_id)
            return True

        # --------------------- design-parameter conflict groups -----------------
        from nesting.panel_mapping import affected_panels, select_genes
        from pygarment.garmentcode.utils import nested_get

        def _flatten_param_paths(node: dict, prefix: list[str] | None = None) -> list[str]:
            prefix = prefix or []
            out = []
            for k, v in node.items():
                if isinstance(v, dict):
                    if "v" in v:
                        out.append(".".join(prefix + [k]))
                    else:
                        out += _flatten_param_paths(v, prefix + [k])
            return out

        def _get_val(dp: dict | None, path: str):
            if dp is None:
                return None
            try:
                node = nested_get(dp, path.split("."))
            except Exception:
                return None
            if isinstance(node, dict) and "v" in node:
                return node.get("v")
            return node

        dp_conflicts: list[str] = []
        if self.design_params is not None and other.design_params is not None:
            paths1 = set(_flatten_param_paths(self.design_params))
            paths2 = set(_flatten_param_paths(other.design_params))
            for p in paths1 | paths2:
                if _get_val(self.design_params, p) != _get_val(other.design_params, p):
                    dp_conflicts.append(p)

        root_ids_p1 = {g.root_id for g in self.genes}
        root_ids_p2 = {g.root_id for g in other.genes}
        all_root_ids = root_ids_p1 | root_ids_p2

        groups: list[dict] = []  # {"roots": set, "params": set, "owner": None|1|2}

        for path in dp_conflicts:
            patterns = affected_panels([path], self.design_params)
            affected_roots = select_genes(all_root_ids, patterns)
            if not affected_roots:
                continue
            # merge into overlapping groups
            idxs = [i for i, g in enumerate(groups) if g["roots"] & affected_roots]
            if not idxs:
                groups.append({"roots": set(affected_roots), "params": {path}, "owner": None})
            else:
                first = idxs[0]
                groups[first]["roots"].update(affected_roots)
                groups[first]["params"].add(path)
                for extra in sorted(idxs[1:], reverse=True):
                    groups[first]["roots"].update(groups[extra]["roots"])
                    groups[first]["params"].update(groups[extra]["params"])
                    groups.pop(extra)

        root_to_group: dict[str, int] = {}
        for gi, g in enumerate(groups):
            for r in g["roots"]:
                root_to_group[r] = gi

        for g in groups:
            p1_roots = g["roots"] & root_ids_p1
            p2_roots = g["roots"] & root_ids_p2
            if p1_roots and not p2_roots:
                g["owner"] = 1
            elif p2_roots and not p1_roots:
                g["owner"] = 2
            else:
                g["owner"] = None

        processed_groups: Set[int] = set()

        child_design_params = copy.deepcopy(self.design_params) if self.design_params is not None else None

        from pygarment.garmentcode.utils import nested_set, nested_del

        def assign_group(parent: "Chromosome", group_idx: int, owner: int):
            g = groups[group_idx]
            if g.get("owner") is not None and g["owner"] != owner:
                return False
            # Atomically place every root family
            snapshot = list(child_genes)
            snapshot_free = deque(free_indices)
            placed_roots = []
            for root_id in g["roots"]:
                if not place_family(parent, root_id):
                    # rollback
                    for i in range(n):
                        child_genes[i] = snapshot[i]
                    free_indices.clear(); free_indices.extend(snapshot_free)
                    for r in placed_roots:
                        blocked_root_ids.discard(r)
                    return False
                placed_roots.append(root_id)
            g["owner"] = owner
            processed_groups.add(group_idx)

            # adopt other parent's param values if owner is 2
            if owner == 2 and child_design_params is not None and other.design_params is not None:
                for p in g["params"]:
                    val = _get_val(other.design_params, p)
                    if val is None:
                        try:
                            nested_del(child_design_params, p.split("."))
                        except Exception:
                            pass
                    else:
                        nested_set(child_design_params, p.split("."), copy.deepcopy(nested_get(other.design_params, p.split("."))))
            return True

        # ------------------------- OX-k parent-1 segments ------------------------
        if 2 * k > n:
            k = n // 2
        import random
        sampled = sorted(random.sample(range(n), 2 * k))
        p1_indices = set()
        for i in range(k):
            start, end = sampled[2 * i], sampled[2 * i + 1]
            p1_indices.update(range(start, end + 1))

        # Pre-assign groups that must come from parent 1
        for gi, g in enumerate(groups):
            if g.get("owner") == 1:
                assign_group(self, gi, 1)

        # Mark roots that must come from parent 2 so P1 doesn't take them
        skip_p1_roots = {r for g in groups if g.get("owner") == 2 for r in g["roots"]}

        # Process parent 1 indices; place families atomically
        for i in sorted(p1_indices):
            gene = self.genes[i]
            if gene.root_id in skip_p1_roots or gene.root_id in blocked_root_ids:
                continue
            gi = root_to_group.get(gene.root_id)
            if gi is not None and gi not in processed_groups:
                assign_group(self, gi, 1)
                continue
            place_family(self, gene.root_id)

        # Pre-assign groups that must come from parent 2
        for gi, g in enumerate(groups):
            if g.get("owner") == 2 and gi not in processed_groups:
                assign_group(other, gi, 2)

        # Fill remaining gaps from parent 2 (cycling through list, not a one-pass generator)
        p2_cycle = list(other.genes)
        p2_pos = 0
        for i in range(n):
            if child_genes[i] is not None:
                continue
            # find next eligible root from P2
            attempts = 0
            while attempts < n:
                gene2 = p2_cycle[p2_pos]
                p2_pos = (p2_pos + 1) % n
                attempts += 1
                root = gene2.root_id
                if root in blocked_root_ids:
                    continue
                gi = root_to_group.get(root)
                if gi is not None and gi not in processed_groups:
                    # try to assign the whole group from P2
                    if assign_group(other, gi, 2):
                        break
                    else:
                        continue
                if place_family(other, root):
                    break
            # loop proceeds; if we couldn't place anything, remaining slots will be handled by sanity step

        # -------------------------- Sanity & backfill ----------------------------
        # 1) No None slots
        if any(g is None for g in child_genes):
            # Backfill with any remaining P1 families not yet used, else P2, then random parents
            available_roots = [r for r in (root_ids_p1 | root_ids_p2) if r not in blocked_root_ids]
            for r in available_roots:
                if not any(g is None for g in child_genes):
                    break
                if not place_family(self, r):
                    place_family(other, r)
            # If still None, try placing single leaves (as last resort)
            for idx, slot in enumerate(child_genes):
                if slot is None:
                    # pick from P1 by index, else P2
                    cand = copy.deepcopy(self.genes[idx])
                    if cand.root_id in blocked_root_ids or cand.id in child_gene_ids:
                        cand = copy.deepcopy(other.genes[idx])
                    occupy(idx, cand)

        # 2) Family integrity: if any leaf of root R is present, ensure all leaves are present
        present_roots = {g.root_id for g in child_genes if g is not None}
        for r in list(present_roots):
            leaves = get_family_leaves(self, r) or get_family_leaves(other, r)
            missing = [m for m in leaves if m.id not in child_gene_ids]
            for m in missing:
                # place missing leaf into next free slot while preserving rotation/translation default
                if not free_indices:
                    # relocate some existing piece to make room (swap into first index)
                    # pick the first index not already that root
                    idx_to_relocate = next((j for j, cg in enumerate(child_genes) if cg is not None and cg.root_id != r), None)
                    if idx_to_relocate is not None:
                        new_slot = idx_to_relocate  # will be occupied() relocated
                    else:
                        # as an absolute fallback, append (keeps gene count consistent after final trim)
                        child_genes.append(copy.deepcopy(m))
                        child_gene_ids.add(m.id)
                        continue
                target_idx = p1_index.get(m.id, p2_index.get(m.id, None))
                if target_idx is None:
                    target_idx = free_indices[0] if free_indices else 0
                occupy(target_idx, copy.deepcopy(m))

        # Trim any accidental expansion and assert size
        child_genes = [g for g in child_genes if g is not None][:n]

        # Integrity validation: ensure no panels (roots) are lost and families are consistent
        expected_roots = {g.root_id for g in self.genes} | {g.root_id for g in other.genes}
        child_roots = {g.root_id for g in child_genes}
        missing_roots = sorted(list(expected_roots - child_roots))
        unexpected_roots = sorted(list(child_roots - expected_roots))

        # Build child parent_ids set to identify leaves within child
        child_parent_ids = {p.parent_id for p in child_genes if p is not None and p.parent_id}

        errors: list[str] = []
        if missing_roots:
            errors.append(f"Missing root panels in offspring: {missing_roots}")
        if unexpected_roots:
            errors.append(f"Unexpected roots in offspring: {unexpected_roots}")

        # For each expected root, check that child contains a full leaf family equal to one of the parents
        for r in sorted(expected_roots):
            # Leaves in parents
            p1_leaves = {m.id for m in (get_family_leaves(self, r) or [])}
            p2_leaves = {m.id for m in (get_family_leaves(other, r) or [])}
            # Leaves in child: pieces with same root and whose id is not any parent_id in child
            child_leaves = {g.id for g in child_genes if g.root_id == r and g.id not in child_parent_ids}

            if not child_leaves:
                # if the entire root is missing, it will be covered by missing_roots above
                if r in child_roots:
                    errors.append(f"Root {r} present but has no leaves in offspring")
                continue

            # Family must match exactly one of the parents (IDs), allowing for parents with identical sets
            if child_leaves != p1_leaves and child_leaves != p2_leaves:
                errors.append(
                    f"Root {r} leaf mismatch. child={sorted(child_leaves)} not in (p1={sorted(p1_leaves)}, p2={sorted(p2_leaves)})"
                )

        if errors:
            raise RuntimeError("Crossover produced invalid offspring: " + "; ".join(errors))

        # Build child
        final_genes = [copy.deepcopy(g) for g in child_genes]
        child = Chromosome(
            pieces=final_genes,
            container=self.container,
            origin="crossover",
            design_params=child_design_params,
            body_params=self.body_params,
        )

        # Optional: force mutation if identical fitness to a parent and configured
        child.calculate_fitness()
        force_mutation = False
        if child.fitness is not None:
            if self.fitness is not None and child.fitness == self.fitness:
                force_mutation = True
            if other.fitness is not None and child.fitness == other.fitness:
                force_mutation = True
        import nesting.config as config
        if force_mutation and getattr(config, "FORCE_MUTATION_ON_CROSSOVER", False):
            old_f = child.fitness
            child.mutate()
            child.calculate_fitness()
            if config.VERBOSE:
                print(f"[DEBUG] Forced mutation changed fitness from {old_f} to {child.fitness}")

        return child



    # def sync_order(self) -> None:
    #     """Sync the order dict to reflect the current gene sequence."""
    #     self.order = OrderedDict((p.id, p) for p in self.genes)

    def _signature(self) -> tuple:
        """
        An immutable fingerprint including:
        - Gene signature: ((id, rotation), …) in gene order
        - Design parameters hash (if available)
        - Parent IDs for tracking lineage (if available)
        """
        # Get signature based on genes
        gene_signature = tuple((p.id, p.rotation) for p in self.genes)
        
        # Include design params hash if available
        design_params_hash = None
        if self.design_params is not None:
            
            # Convert design params to a stable string representation and hash it
            design_params_str = json.dumps(self.design_params, sort_keys=True)
            design_params_hash = hash(design_params_str)
    
        # Return signature components
        return (gene_signature, design_params_hash)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Chromosome):
            return NotImplemented
        self_sig = self._signature()
        other_sig = other._signature()
        return self_sig[0] == other_sig[0] and self_sig[1] == other_sig[1]

    # def __hash__(self) -> int:
    #     return hash(self._signature())

    def __repr__(self) -> str:
        signature = self._signature()
        gene_signature = signature[0]
        design_params_hash = signature[1]
        
        # Original gene signature representation
        genes_str = str(gene_signature)
        
        return f"Chromosome(genes={genes_str}, design_params_hash={design_params_hash})"