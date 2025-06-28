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

from .layout import Piece, Container, Layout, LayoutView
from .placement_engine import DECODER_REGISTRY
#from pygarment.garmentcode.params import DesignSampler
from pygarment.garmentcode.utils import nested_get, nested_set
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
    max_delta = config.PARAM_CHANGE_MARGIN * span  # 20 % of the range

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

    # touches at least one panel in this chromo
    panel_ids = {g.id for g in genes}
    mutatable = [
        p for p in numeric_ok
        if any(pid in affected_panels([p]) for pid in panel_ids)
    ]

    # if config.VERBOSE:
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
    cc_height = dec.concave_hull_utilization()
    rest_height = dec.rest_height()
    return cc_height + config.REST_PENALTY * rest_height

@register_metric("cc_with_rest_length")
def fitness_cc_length_combined(chromosome: Chromosome, decoder: str):
    """
    Combined fitness metric for concave hull length and rest length.
    
    This metric returns the sum of the concave hull length and the rest length.
    It is useful for evaluating the overall horizontal space utilization of the layout.
    """
    dec = _run_decoder(chromosome, decoder)
    cc_length = dec.concave_hull_utilization()
    rest_length = dec.rest_length()
    return cc_length + config.REST_PENALTY * rest_length

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
        return self

    # ── simple mutations ───────────────────────────────────────────────

    def _mutate_split(self):
        max_splits = min(config.NUM_SPLITS, len(self.genes))
        for _ in range(random.randint(1, max_splits)):
            unsplit = [i for i, g in enumerate(self.genes) if g.parent_id is None]
            if not unsplit:
                break
            idx = random.choices(unsplit, weights=[self.genes[i].bbox_area for i in unsplit])[0]
            left, right = self.genes[idx].split()
            self.genes[idx:idx + 1] = [left, right]
            # randomly relocate one half
            child = random.choice([left, right])
            self.genes.remove(child)
            self.genes.insert(random.randrange(len(self.genes) + 1), child)

    def _mutate_rotate(self):
        for _ in range(random.randint(1, len(self.genes))):
            idx = random.randrange(len(self.genes))
            self.genes[idx].rotate(random.choice(config.ALLOWED_ROTATIONS))

    def _mutate_swap(self):
        i, j = random.sample(range(len(self.genes)), 2)
        self.genes[i], self.genes[j] = self.genes[j], self.genes[i]

    def _mutate_inversion(self):
        i, j = sorted(random.sample(range(len(self.genes)), 2))
        self.genes[i:j + 1] = reversed(self.genes[i:j + 1])

    def _mutate_insertion(self):
        i = random.randrange(len(self.genes))
        self.genes.insert(random.randrange(len(self.genes)), self.genes.pop(i))

    def _mutate_scramble(self):
        i, j = sorted(random.sample(range(len(self.genes)), 2))
        subset = self.genes[i:j + 1]
        random.shuffle(subset)
        self.genes[i:j + 1] = subset

    # ── design‑parameter mutation ──────────────────────────

    def _mutate_design_params(self):
        if not (self.design_params and self.body_params):
            if config.VERBOSE:
                print("[Chromosome] design‑param mutation skipped - missing design or body params")
            return


        path = random.choice(self._mutatable_params)
        node = nested_get(self.design_params, path.split("."))

        # all_paths = _filter_excluded(_flatten_param_paths(self.design_params))
        # path = _choose_param(self.design_params, all_paths)

        if path is None:  # should not happen, but just in case
            if config.VERBOSE:
                print("[Chromosome] No numeric design‑params left to mutate -> skip")
            return

        node = nested_get(self.design_params, path.split("."))
        p_type = node.get("type", "float")
        old_val = node["v"]
        new_val = _random_value(old_val, p_type, node["range"])
        nested_set(self.design_params, path.split(".") + ["v"], new_val)
        
        # Only record the change if the value actually changed
        if old_val != new_val:
            # Store this specific parameter change for tracking
            if not hasattr(self, 'param_changes_this_gen'):
                self.param_changes_this_gen = []
            
            # Record that this specific parameter was modified in this generation
            self.param_changes_this_gen.append({
                'param_path': path,
                'old_value': old_val,
                'new_value': new_val
            })
            print(f"[Chromosome] {path}: {old_val} -> {new_val}")

        else:
            # If the value did not change, we still log it for completeness
            print(f"[Chromosome] {path}: value unchanged ({old_val})")
            return False

        # regenerate the garment and update affected pieces ----------------
        from assets.garment_programs.meta_garment import MetaGarment
        from nesting.panel_mapping import affected_panels, select_genes
        from nesting.path_extractor import PatternPathExtractor
        import tempfile

        panel_ids = {g.id for g in self.genes}
        affected = affected_panels([path])
        # debug print
        if config.VERBOSE:
            print(f"[Chromosome] Affected panels for {path}: {affected}")

        if not any(fnmatch(pid, pat) for pat in affected for pid in panel_ids):
            # changed param does not influence this chromosome
            if config.VERBOSE:
                print(f"[Chromosome] No affected panels for {path} in this chromosome")
                # throw error because this should not happen
            raise ValueError(f"No affected panels for {path} in this chromosome, how did you pick this param?")

        # regenerate the garment with the new design parameters
        mg = MetaGarment("design_mut", self.body_params, self.design_params)
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


        # update the genes with the new pieces
        # but only for those that are affected by the design parameter change
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
                suffix = g.id[len(g.root_id) + 1:]  # e.g. 's1' or 's2'
                replacement = copy.deepcopy(left if suffix == "s1" else right)
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

        return True  # indicate that the design parameters were successfully mutated

    def crossover_ox1(
        self,
        other: "Chromosome",
        k: int = 1,
        circular_walk: bool = config.OX_CIRCULAR
    ) -> "Chromosome":
        """
        OX-k crossover on the gene order/rotations, PLUS a separate uniform
        crossover on the design-parameters.  After the params are combined we
        regenerate the garment so every piece geometry matches the new design.
        """

        t0 = time.time()
        if config.VERBOSE:
            print(f"[XO]  ▶︎  Starting crossover_ox1  |  k={k}, circular={circular_walk}")

        # ──────────────────────────────────────────────────────────────────
        # OX-k on gene order / rotations
        # ──────────────────────────────────────────────────────────────────
        size = len(self.genes)
        if 2 * k > size:
            raise ValueError("2·k cut points exceed chromosome length")

        cut_pts = sorted(random.sample(range(size), 2 * k))
        segs = [(a, b) for a, b in zip(cut_pts[::2], cut_pts[1::2])]

        if config.VERBOSE:
            print(f"[XO]  Cut points: {cut_pts}")
            print(f"[XO]  Segments : {segs}")

        child_genes: list[Piece] = [None] * size
        taken_ids: set[str] = set()

        # Copy the chosen segments from parent-1
        for a, b in segs:
            for i in range(a, b + 1):
                g = copy.deepcopy(self.genes[i])
                child_genes[i] = g
                taken_ids.add(g.id)
        if config.VERBOSE:
            print(f"[XO]  Copied {len(taken_ids)} genes from P1 segments")

        # Fill the gaps with genes from parent-2
        if circular_walk:
            j = 0
            for i in range(size):
                if child_genes[i] is not None:
                    continue
                while other.genes[j].id in taken_ids:
                    j = (j + 1) % len(other.genes)
                child_genes[i] = copy.deepcopy(other.genes[j])
                taken_ids.add(other.genes[j].id)
            if config.VERBOSE:
                print(f"[XO]  Filled remaining slots with circular walk from P2")
        else:
            other_idx = 0
            for i in range(size):
                if child_genes[i] is not None:
                    continue
                while other_idx < len(other.genes) and other.genes[other_idx].id in taken_ids:
                    other_idx += 1
                if other_idx >= len(other.genes):
                    raise ValueError("Not enough unique genes in parent 2")
                child_genes[i] = copy.deepcopy(other.genes[other_idx])
                taken_ids.add(other.genes[other_idx].id)
                other_idx += 1
            if config.VERBOSE:
                print(f"[XO]  Filled remaining slots with linear walk from P2")

        #print the child genes
        if config.VERBOSE:
            print(f"[XO]  Child genes: {[g.id for g in child_genes]}")

        # ──────────────────────────────────────────────────────────────────
        # 2) Uniform crossover on the design-parameter tree
        # ──────────────────────────────────────────────────────────────────
        if self.design_params is None or other.design_params is None:
            raise ValueError("Both parents must carry design_params for crossover")

        _leaf_choices = 0

        def _recurse(u, v):
            nonlocal _leaf_choices
            if not isinstance(u, dict):
                _leaf_choices += 1
                choice = copy.deepcopy(random.choice([u, v]))
                return choice
            out = {}
            keys = u.keys() | v.keys()
            for kx in keys:
                out[kx] = _recurse(u[kx], v[kx])
            return out

        child_dp = _recurse(self.design_params, other.design_params)

        if config.VERBOSE:
            print(f"[XO]  Uniform-XO picked {_leaf_choices} leaves")

        # ──────────────────────────────────────────────────────────────────
        # 3) Regenerate geometry from the new parameters
        # ──────────────────────────────────────────────────────────────────
        from assets.garment_programs.meta_garment import MetaGarment
        from nesting.path_extractor import PatternPathExtractor
        import tempfile

        mg = MetaGarment("child", self.body_params, child_dp)
        pattern = mg.assembly()
        with tempfile.TemporaryDirectory() as td:
            spec = Path(td) / f"{pattern.name}_specification.json"
            pattern.serialize(
                Path(td),
                to_subfolder=False,
                with_3d=False,
                with_text=False,
                view_ids=False
            )
            extractor = PatternPathExtractor(spec)
            fresh = extractor.get_all_panel_pieces(
                samples_per_edge=config.SAMPLES_PER_EDGE
            )
            for p in fresh.values():
                p.add_seam_allowance(config.SEAM_ALLOWANCE_CM)

        if config.VERBOSE:
            print(f"[XO]  Regenerated {len(fresh)} fresh panel pieces")

        # Replace each gene’s geometry but keep its pose
        for i, g in enumerate(child_genes):
            base = fresh[g.id]
            base.rotation, base.translation = g.rotation, g.translation
            child_genes[i] = base

        # ──────────────────────────────────────────────────────────────────
        # 4) Spawn the offspring
        # ──────────────────────────────────────────────────────────────────
        child = Chromosome(
            child_genes,
            self.container,
            origin="crossover",
            design_params=child_dp,
            body_params=self.body_params
        )
        child.last_mutation = None
        child.param_changes_this_gen = []

        child.calculate_fitness()  # compute fitness of the child

        if config.VERBOSE:
            dt = time.time() - t0
            print(f"[XO]  ▲  crossover_ox1 finished in {dt:.3f}s | child fitness = {child.fitness}")

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