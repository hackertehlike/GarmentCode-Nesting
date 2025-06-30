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
                    return True # Success

        # If loop completes, no mutation led to a fitness change, restore state
        # THIS SHOULD NEVER NEVER NEVER HAPPEN
        # If it does something is very wrong with the logic
        self.design_params = original_design_params
        self._genes = original_genes
        self.fitness = original_fitness
        if config.VERBOSE:
            print("[Chromosome] No design param mutation resulted in a fitness change.")
        return False

    def _apply_design_param_change(self, path: str, old_val: Any, new_val: Any) -> bool:
        """Helper to regenerate garment pieces after a design param change and update genes."""
        from assets.garment_programs.meta_garment import MetaGarment
        from nesting.panel_mapping import affected_panels, select_genes
        from nesting.path_extractor import PatternPathExtractor
        import tempfile

        panel_ids = {g.id for g in self.genes}
        affected = affected_panels([path])

        if not any(fnmatch(pid, pat) for pat in affected for pid in panel_ids):
            if config.VERBOSE:
                print(f"[Chromosome] No affected panels for {path} in this chromosome")
            return False

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
        child_genes: list[Piece | None] = [None] * n
        child_gene_ids: set[str] = set()
        blocked_root_ids: set[str] = set()

        # ------------------------------------------------------------------
        # Handle design parameter conflicts
        # ------------------------------------------------------------------
        from nesting.panel_mapping import affected_panels, select_genes

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
            all_paths = paths1 | paths2
            for p in all_paths:
                if _get_val(self.design_params, p) != _get_val(other.design_params, p):
                    dp_conflicts.append(p)

        # Build groups of root ids affected by differing params
        root_ids_p1 = {g.root_id for g in self.genes}
        root_ids_p2 = {g.root_id for g in other.genes}
        all_root_ids = root_ids_p1 | root_ids_p2

        groups: list[dict] = []  # each: {"roots": set, "params": set, "owner": None|1|2}

        for path in dp_conflicts:
            patterns = affected_panels([path])
            affected_roots = select_genes(all_root_ids, patterns)
            if not affected_roots:
                continue
            # merge with existing groups if overlapping
            idxs = [i for i, g in enumerate(groups) if g["roots"] & affected_roots]
            if not idxs:
                groups.append({"roots": set(affected_roots), "params": {path}})
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

        processed_groups: set[int] = set()

        child_design_params = copy.deepcopy(self.design_params) if self.design_params is not None else None

        def assign_group(parent: "Chromosome", group_idx: int, owner: int):
            """Assign an entire design group from *parent* to the child."""
            g = groups[group_idx]
            if g.get("owner") is not None and g["owner"] != owner:
                return
            g["owner"] = owner
            for root_id in g["roots"]:
                family = get_family_leaves(root_id, parent)
                for member in family:
                    member_idx = next(idx for idx, gg in enumerate(parent.genes) if gg.id == member.id)
                    if member_idx >= len(child_genes):
                        child_genes.append(None)
                    if child_genes[member_idx] is None:
                        child_genes[member_idx] = copy.deepcopy(member)
                        child_gene_ids.add(member.id)
                blocked_root_ids.add(root_id)

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

            processed_groups.add(group_idx)


        # Define the k segments from parent 1
        if 2 * k > n:
            k = n // 2
        
        sampled_points = sorted(random.sample(range(n), 2 * k))
        p1_indices = set()
        for i in range(k):
            start, end = sampled_points[2*i], sampled_points[2*i+1]
            p1_indices.update(range(start, end + 1))

        # Helper to find all leaf-node descendants of a root in a parent
        def get_family_leaves(root_id: str, parent: "Chromosome") -> list[Piece]:
            parent_ids = {p.parent_id for p in parent.genes if p.parent_id}
            return [
                g for g in parent.genes
                if g.root_id == root_id and g.id not in parent_ids
            ]

        # Pre-assign any groups that must come from parent 1
        for gi, g in enumerate(groups):
            if g.get("owner") == 1:
                assign_group(self, gi, 1)

        # Mark any groups that must come from parent 2 so parent 1 doesn't use them
        skip_p1_roots = {r for g in groups if g.get("owner") == 2 for r in g["roots"]}

        # Process parent 1 in a single pass
        for i in p1_indices:
            if child_genes[i] is not None:
                continue  # Already filled by a family member from a previous iteration

            gene = self.genes[i]

            if gene.root_id in skip_p1_roots:
                continue

            gi = root_to_group.get(gene.root_id)
            if gi is not None and gi not in processed_groups:
                assign_group(self, gi, 1)
                continue

            # If gene is a descendant, take the whole family
            if gene.root_id != gene.id:
                blocked_root_ids.add(gene.root_id)
                family = get_family_leaves(gene.root_id, self)

                for member in family:
                    # Find the original index of the family member to place it correctly
                    
                    member_idx = next(
                        idx for idx, g in enumerate(self.genes) if g.id == member.id
                    )
                    if child_genes[member_idx] is None:
                        child_genes[member_idx] = copy.deepcopy(member)
                        child_gene_ids.add(member.id)

            # If it's a simple leaf gene
            else:
                parent_ids = {p.parent_id for p in self.genes if p.parent_id}
                if gene.id not in parent_ids: # It's a leaf
                    if gene.id not in child_gene_ids and gene.root_id not in blocked_root_ids:
                        child_genes[i] = copy.deepcopy(gene)
                        child_gene_ids.add(gene.id)
                        blocked_root_ids.add(gene.root_id)

        # Process parent 2, filling in the gaps
        other_genes_iter = (g for g in other.genes)

        # Pre-assign groups that must come from parent 2
        for gi, g in enumerate(groups):
            if g.get("owner") == 2 and gi not in processed_groups:
                assign_group(other, gi, 2)

        for i in range(n):
            if child_genes[i] is None:
                for gene_from_other in other_genes_iter:
                    # Determine if the gene from the other parent and its family are eligible
                    family = get_family_leaves(gene_from_other.root_id, other)

                    gi = root_to_group.get(gene_from_other.root_id)
                    if gi is not None and gi not in processed_groups:
                        if groups[gi].get("owner") == 1:
                            continue
                        assign_group(other, gi, 2)
                        break

                    # Check if the root is blocked or any family member is already in the child
                    if (gene_from_other.root_id not in blocked_root_ids and
                            not any(m.id in child_gene_ids for m in family)):

                        # Add the entire family to the child
                        for member in family:
                            # Find the next available slot
                            try:
                                insert_idx = child_genes.index(None)
                                child_genes[insert_idx] = copy.deepcopy(member)
                                child_gene_ids.add(member.id)
                            except ValueError:
                                # No more empty slots, append to the end
                                child_genes.append(copy.deepcopy(member))
                                child_gene_ids.add(member.id)

                        blocked_root_ids.add(gene_from_other.root_id)
                        break  # Move to the next empty slot in the child

        # Create the new chromosome
        final_genes = [g for g in child_genes if g is not None]
        
        child = Chromosome(
            pieces=final_genes,
            container=self.container,
            origin="crossover",
            design_params=child_design_params,
            body_params=self.body_params,
        )
        
        # Debug: Check if offspring has same fitness as a parent
        child.calculate_fitness()
        if child.fitness is not None:
            if self.fitness is not None and child.fitness == self.fitness:
                print(f"[DEBUG] Crossover offspring has same fitness as parent 1 ({self.fitness:.4f}).")
                print(f"  Parent 1 signature: {self._signature()}")
                print(f"  Child signature: {child._signature()}")
            if other.fitness is not None and child.fitness == other.fitness:
                print(f"[DEBUG] Crossover offspring has same fitness as parent 2 ({other.fitness:.4f}).")
                print(f"  Parent 2 signature: {other._signature()}")
                print(f"  Child signature: {child._signature()}")

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