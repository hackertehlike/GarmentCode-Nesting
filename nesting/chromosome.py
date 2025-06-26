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
from pygarment.garmentcode.params import DesignSampler
from pygarment.garmentcode.utils import nested_get, nested_set
import nesting.config as config

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
        if k == "meta":
            continue  # never touch meta‑data
        if isinstance(v, dict):
            if "v" in v:
                paths.append(".".join(prefix + [k]))
            else:
                paths += _flatten_param_paths(v, prefix + [k])
    return paths


def _filter_excluded(paths: Sequence[str]) -> list[str]:
    patterns = config.EXCLUDED_PARAM_PATHS or []
    if not patterns:
        return list(paths)
    return [p for p in paths if not any(fnmatch(p, pat) for pat in patterns)]


def _numeric_range_ok(node: JsonDict) -> bool:
    rng = node.get("range", [])
    return len(rng) >= 2 and all(isinstance(x, (int, float)) for x in rng[:2])


def _choose_numeric_param(params: JsonDict, paths: Sequence[str]) -> str | None:
    """Return a random numeric *path* or *None* if none available."""
    numeric = [p for p in paths if _numeric_range_ok(nested_get(params, p.split(".")))]
    return random.choice(numeric) if numeric else None


def _random_value(old: Any, p_type: str, rng: Sequence[float | int]):
    """Return a *new* value that differs from *old* by ≤ 20 % of *range* width."""
    
    # Handle boolean type explicitly
    if p_type == "bool" or isinstance(old, bool):
        return old  # Don't randomize boolean values
        
    # Handle None value
    if old is None:
        return None
        
    lower, upper = rng[0], rng[1]
    span = upper - lower
    max_delta = config.PARAM_CHANGE_MARGIN * span  # 20 % of the *total* range, not of *old*

    if span <= 0:
        return old  # degenerate range – nothing we can do

    # integer parameters --------------------------------------------------
    if p_type == "int":
        vals = list(range(int(lower), int(upper) + 1))
        candidates = [v for v in vals if abs(v - old) <= max_delta and v != old]
        if not candidates:
            candidates = [v for v in vals if v != int(old)] or vals
        return random.choice(candidates)

    # float parameters ----------------------------------------------------
    attempts = 0
    while attempts < 100:
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
        # If we can't stay within both the bounds and max_delta, just pick a different value
        return lower if abs(old - lower) > 1e-6 else upper

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
        design_sampler: "DesignSampler" | None = None,
    ):
        # Store a deep copy of each piece
        self._genes = [copy.deepcopy(p) for p in pieces]
        self.container = container
        self.fitness: float | None = None

        self.design_params = copy.deepcopy(design_params) if design_params else None
        self.body_params = body_params
        self.design_sampler = design_sampler

        # Track origin and last mutation type
        self.origin: str | None = origin
        self.last_mutation: str | None = None
        
        # For backwards compatibility, if the origin is crossover, set last_mutation to crossover_params
        if origin == "crossover":
            self.last_mutation = "crossover_params"
        
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
        """Pick a mutation based on :pydata:`config.MUTATION_WEIGHTS` and apply it."""
        start = time.time()
        self.last_mutation = mutation = _weighted_choice(config.MUTATION_WEIGHTS)
        # Clear the parameter changes tracking for this generation
        self.param_changes_this_gen = []
        
        if config.VERBOSE:
            print(f"[Chromosome.mutate] → {mutation}")

        # dispatch table maps mutation‑name → handler method
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
        handler()  # perform the actual work

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

    # ── heavy‑weight design‑parameter mutation ──────────────────────────

    def _mutate_design_params(self):
        if not (self.design_params and self.body_params):
            if config.VERBOSE:
                print("[Chromosome] design‑param mutation skipped – missing prerequisites")
            return

        all_paths = _filter_excluded(_flatten_param_paths(self.design_params))
        path = _choose_numeric_param(self.design_params, all_paths)
        if path is None:
            if config.VERBOSE:
                print("[Chromosome] No numeric design‑params left to mutate → skip")
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

        if config.VERBOSE:
            if old_val != new_val:
                print(f"[Chromosome] {path}: {old_val} → {new_val}")
            else:
                print(f"[Chromosome] {path}: value unchanged ({old_val})")

        # regenerate the garment and update affected pieces ----------------
        from assets.garment_programs.meta_garment import MetaGarment
        from nesting.panel_mapping import affected_panels, select_genes, filter_parameters
        from nesting.path_extractor import PatternPathExtractor
        import tempfile

        panel_ids = {g.id for g in self.genes}
        affected = affected_panels([path])
        if not any(fnmatch(pid, pat) for pat in affected for pid in panel_ids):
            # changed param does not influence this chromosome
            return

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

        changed_ids = select_genes(new_pieces.keys(), affected)
        for i, g in enumerate(self.genes):
            if g.id in changed_ids:
                new_piece = copy.deepcopy(new_pieces[g.id])
                new_piece.rotation, new_piece.translation = g.rotation, g.translation
                self.genes[i] = new_piece

        # optional CSV logging --------------------------------------------------
        if config.LOG_DESIGN_PARAM_PATHS and changed_ids:
            log_path = Path(config.SAVE_LOGS_PATH) / "design_param_paths.csv"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", newline="") as fh:
                writer = csv.writer(fh)
                if fh.tell() == 0:
                    writer.writerow(["piece_id", "param_path", "old_v", "new_v"])
                for pid in changed_ids:
                    writer.writerow([pid, path, old_val, new_val])

    
    def crossover_ox1(self, other: "Chromosome", k: int = 1) -> "Chromosome":
        """
        Generalised OX1 crossover that respects component split trees.

        Rules
        -----
        • Parents may contain different numbers of leaves (genes).
        • The child inherits complete split-trees: every original component
        (root_id) is taken entirely from *one* parent, never mixed.
        • Step 1 is still "OX1 style": pick 2·k cut points on *self* and copy
        those segments (plus any additional leaves that belong to the same
        components) into the child, preserving order.
        • Design parameters and corresponding panel geometry are inherited together
        from the same parent to maintain consistency.

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
        from nesting.panel_mapping import PARAM_TO_PATTERNS

        gene_ids = [g.id for g in self.genes]

        def _build_groups(ids):
            adj = {i: set() for i in ids}

            # group by root_id
            roots = {}
            for i in ids:
                rid = i.split("_")[0]
                roots.setdefault(rid, []).append(i)
            for lst in roots.values():
                for a in lst:
                    for b in lst:
                        if a != b:
                            adj[a].add(b); adj[b].add(a)

            # parameter groups - only if design parameters are available
            param_group = {}
            if self.design_params is not None or other.design_params is not None:
                for param, patterns in PARAM_TO_PATTERNS.items():
                    matched = [i for i in ids if any(fnmatch(i, p) for p in patterns)]
                    for a in matched:
                        for b in matched:
                            if a != b:
                                adj[a].add(b); adj[b].add(a)
                    if matched:
                        param_group[param] = 0  # Will be updated properly later

            visited = set(); groups = []
            for i in ids:
                if i in visited:
                    continue
                comp=set([i]); stack=[i]; visited.add(i)
                while stack:
                    cur=stack.pop()
                    for nb in adj[cur]:
                        if nb not in visited:
                            visited.add(nb); comp.add(nb); stack.append(nb)
                groups.append(comp)

            gmap={}
            for idx, comp in enumerate(groups):
                for i in comp:
                    gmap[i]=idx
                    
            # Update param_group with correct group indices
            if self.design_params is not None or other.design_params is not None:
                for param, patterns in PARAM_TO_PATTERNS.items():
                    matched=[i for i in ids if any(fnmatch(i,p) for p in patterns)]
                    if matched:
                        param_group[param]=gmap[matched[0]]
                        
            return gmap, param_group

        group_map, param_group = _build_groups(gene_ids)

        def get_group_id(piece):
            return group_map.get(piece.id, -1)

        # Map of group IDs to genes from both parents
        parent_genes = {
            "self": {},
            "other": {}
        }
        
        # Group genes by their group ID for both parents
        for g in self.genes:
            gid = get_group_id(g)
            if gid not in parent_genes["self"]:
                parent_genes["self"][gid] = []
            parent_genes["self"][gid].append(g)
        
        for g in other.genes:
            gid = get_group_id(g)
            if gid not in parent_genes["other"]:
                parent_genes["other"][gid] = []
            parent_genes["other"][gid].append(g)

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
        chosen_source: dict[int, str] = {}     # group_id → "self" | "other"

        # Copy the chosen segments from self (plus all leaves of their components)
        for seg_start, seg_end in segments:
            for idx in range(seg_start, seg_end + 1):
                g = self.genes[idx]
                gid = get_group_id(g)
                if gid in chosen_source:
                    continue
                chosen_source[gid] = "self"
                
                # Add all genes from this group
                if gid in parent_genes["self"]:
                    for gene in parent_genes["self"][gid]:
                        if gene.id not in placed_ids:
                            child_genes.append(copy.deepcopy(gene))
                            placed_ids.add(gene.id)

        # ------------------------------------------------------------------
        # 2) walk through parent 2 and take components not chosen yet --------
        # ------------------------------------------------------------------
        for gid in parent_genes["other"]:
            if gid in chosen_source:
                continue
            chosen_source[gid] = "other"
            
            # Add all genes from this group
            for gene in parent_genes["other"][gid]:
                if gene.id not in placed_ids:
                    child_genes.append(copy.deepcopy(gene))
                    placed_ids.add(gene.id)

        # ------------------------------------------------------------------
        # finally, copy any still-missing components from parent 1 --------
        # ------------------------------------------------------------------
        for gid in parent_genes["self"]:
            if gid in chosen_source:
                continue
            chosen_source[gid] = "self"
            
            # Add all genes from this group
            for gene in parent_genes["self"][gid]:
                if gene.id not in placed_ids:
                    child_genes.append(copy.deepcopy(gene))
                    placed_ids.add(gene.id)

        # ------------------------------------------------------------------
        # finished -------------------------------------------------------
        # ------------------------------------------------------------------
        if config.LOG_TIME:
            dt = time.time() - t0
            print(f"[Chromosome.crossover_ox1] k={k} produced "
                f"{len(child_genes)} genes in {dt:.4f} s, segments={segments}")

        child_dp = None
        if self.design_params is not None and other.design_params is not None:
            # Create a deep copy of the first parent's design_params
            child_dp = copy.deepcopy(self.design_params)
            
            # For each parameter group, copy from the appropriate parent
            for param, gid in param_group.items():
                source = chosen_source.get(gid, "self")
                parent_dp = self.design_params if source == "self" else other.design_params
                val = nested_get(parent_dp, param.split("."))
                # Ensure we do a deep copy of the parameter value
                nested_set(child_dp, param.split("."), copy.deepcopy(val))

        # Create the offspring chromosome with the combined genes and design parameters
        # Ensure we pass a deep copy of child_dp to avoid reference sharing
        offspring = Chromosome(child_genes, self.container, origin="crossover",
                          design_params=copy.deepcopy(child_dp), 
                          body_params=self.body_params,
                          design_sampler=self.design_sampler)
        
        # Set the last_mutation field to indicate that design parameters were inherited during crossover
        offspring.last_mutation = "crossover_params"
        
        # Initialize an empty param_changes_this_gen list to avoid inheriting parent's changes
        offspring.param_changes_this_gen = []
        
        return offspring


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
            try:
                # Convert design params to a stable string representation and hash it
                design_params_str = json.dumps(self.design_params, sort_keys=True)
                design_params_hash = hash(design_params_str)
            except (TypeError, ValueError) as e:
                # If design params can't be serialized to JSON, fallback to a simple id
                if config.VERBOSE:
                    print(f"[Chromosome._signature] Warning: Could not hash design params: {e}")
                # Return just the gene signature with a dummy hash
                design_params_hash = hash(id(self.design_params))
        
        # Return signature components
        return (gene_signature, design_params_hash)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Chromosome):
            return NotImplemented
        self_sig = self._signature()
        other_sig = other._signature()
        return self_sig[0] == other_sig[0] and self_sig[1] == other_sig[1]

    def __hash__(self) -> int:
        return hash(self._signature())

    def __repr__(self) -> str:
        signature = self._signature()
        gene_signature = signature[0]
        design_params_hash = signature[1]
        
        # Original gene signature representation
        genes_str = str(gene_signature)
        
        if design_params_hash is not None:
            return f"Chromosome(genes={genes_str}, design_params_hash={design_params_hash})"
        else:
            return f"Chromosome(genes={genes_str})"