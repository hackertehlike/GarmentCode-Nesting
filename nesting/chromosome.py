# chromosome.py

from __future__ import annotations
from collections import OrderedDict
from itertools import chain
import time
import random
import copy
from typing import Callable
from pathlib import Path
import csv
import json

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
        
        # Always print which mutation types are available with their weights
        if config.VERBOSE:
            print(f"[Chromosome.mutate] Available mutations: {list(zip(mutation_types, weights))}")
            
        mutation = random.choices(mutation_types, weights=weights, k=1)[0]
        self.last_mutation = mutation
        
        # Always print which mutation was selected
        if config.VERBOSE:
            print(f"[Chromosome.mutate] Selected mutation: {mutation}")
            
        # For design_param, check if data is available early and print more info
        if mutation == "design_param":
            if self.design_params is None:
                print("[Chromosome.mutate] Missing design_params")
            if self.design_sampler is None:
                print("[Chromosome.mutate] Missing design_sampler")
            if self.body_params is None:
                print("[Chromosome.mutate] Missing body_params")

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

        elif mutation == "design_param":
            if self.design_params is None or self.design_sampler is None or self.body_params is None:
                if config.VERBOSE:
                    print("[Chromosome.mutate] design_param skipped: missing data")
            else:
                if config.VERBOSE:
                    print("[Chromosome.mutate] Starting design_param mutation")
                
                # flatten parameter paths
                param_paths: list[str] = []

                def _collect(node, prefix=[]):
                    for k, v in node.items():
                        if isinstance(v, dict) and "v" in v:
                            param_paths.append(".".join(prefix + [k]))
                        elif isinstance(v, dict):
                            _collect(v, prefix + [k])

                _collect(self.design_params)

                if config.VERBOSE:
                    print(f"[Chromosome.mutate] Found {len(param_paths)} possible parameters to mutate")
                
                # Always log whether we can find parameters to mutate
                if not param_paths:
                    print("[Chromosome.mutate] ERROR: No mutable parameters found in design_params")
                    print("[Chromosome.mutate] design_params structure:")
                    import json
                    print(json.dumps(self.design_params, indent=2)[:500] + "..." if len(json.dumps(self.design_params)) > 500 else json.dumps(self.design_params, indent=2))
                
                if param_paths:
                    chosen = random.choice(param_paths)
                    if config.VERBOSE:
                        old_value = nested_get(self.design_params, chosen.split("."))
                        print(f"[Chromosome.mutate] Selected parameter: '{chosen}' with current value: {old_value}")
                    
                    new_params = copy.deepcopy(self.design_params)
                    self.design_sampler._randomize_value(new_params, chosen.split("."))
                    
                    if config.VERBOSE:
                        new_value = nested_get(new_params, chosen.split("."))
                        print(f"[Chromosome.mutate] Changed '{chosen}' from {old_value} to {new_value}")
                    
                    self.design_params = new_params

                    from assets.garment_programs.meta_garment import MetaGarment
                    from nesting.path_extractor import PatternPathExtractor
                    from nesting.panel_mapping import affected_panels, select_genes
                    import tempfile

                    if config.VERBOSE:
                        print(f"[Chromosome.mutate] Regenerating garment with MetaGarment")
                    
                    mg = MetaGarment("design_mut", self.body_params, self.design_params)
                    pattern = mg.assembly()
                    with tempfile.TemporaryDirectory() as td:
                        out_dir = Path(td)
                        pattern.serialize(out_dir, to_subfolder=False, with_3d=False, with_text=False, view_ids=False)
                        spec_path = out_dir / f"{pattern.name}_specification.json"
                        extractor = PatternPathExtractor(spec_path)
                        new_pieces = extractor.get_all_panel_pieces(samples_per_edge=config.SAMPLES_PER_EDGE)
                        for p in new_pieces.values():
                            p.add_seam_allowance(config.SEAM_ALLOWANCE_CM)
                    
                    if config.VERBOSE:
                        print(f"[Chromosome.mutate] Generated {len(new_pieces)} new panel pieces")

                    changed = affected_panels([chosen])
                    if config.VERBOSE:
                        print(f"[Chromosome.mutate] Parameter '{chosen}' affects panel patterns: {changed}")
                    
                    affected_ids = select_genes(new_pieces.keys(), changed)
                    if config.VERBOSE:
                        print(f"[Chromosome.mutate] Will update {len(affected_ids)} panel pieces: {affected_ids}")

                    update_count = 0
                    update_records = []
                    for i, g in enumerate(self.genes):
                        if g.id in affected_ids:
                            old_path = copy.deepcopy(g.outer_path)
                            new_piece = copy.deepcopy(new_pieces[g.id])
                            new_piece.rotation = g.rotation
                            new_piece.translation = g.translation
                            self.genes[i] = new_piece
                            update_count += 1
                            update_records.append((g.id, old_path, new_piece.outer_path))

                    if config.VERBOSE:
                        print(f"[Chromosome.mutate] Successfully updated {update_count} panel pieces")

                    if update_records and (config.VERBOSE or config.LOG_DESIGN_PARAM_PATHS):
                        log_file = Path(config.SAVE_LOGS_PATH) / "design_param_paths.csv"
                        header_needed = not log_file.exists()
                        with log_file.open("a", newline="") as fh:
                            writer = csv.writer(fh)
                            if header_needed:
                                writer.writerow(["piece_id", "old_outer_path", "new_outer_path"])
                            for rec in update_records:
                                piece_id, old_p, new_p = rec
                                writer.writerow([
                                    piece_id,
                                    json.dumps(old_p),
                                    json.dumps(new_p),
                                ])

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
        from nesting.panel_mapping import PARAM_TO_PATTERNS, select_genes

        gene_ids = [g.id for g in self.genes]

        def _build_groups(ids):
            from fnmatch import fnmatch

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
        # 3) finally, copy any still-missing components from parent 1 --------
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
        # 4) finished -------------------------------------------------------
        # ------------------------------------------------------------------
        if config.LOG_TIME:
            dt = time.time() - t0
            print(f"[Chromosome.crossover_ox1] k={k} produced "
                f"{len(child_genes)} genes in {dt:.4f} s, segments={segments}")

        child_dp = None
        if self.design_params is not None and other.design_params is not None:
            child_dp = copy.deepcopy(self.design_params)
            for param, gid in param_group.items():
                source = chosen_source.get(gid, "self")
                parent_dp = self.design_params if source == "self" else other.design_params
                val = nested_get(parent_dp, param.split("."))
                nested_set(child_dp, param.split("."), copy.deepcopy(val))

        return Chromosome(child_genes, self.container, origin="crossover",
                           design_params=child_dp, body_params=self.body_params,
                           design_sampler=self.design_sampler)


    # def sync_order(self) -> None:
    #     """Sync the order dict to reflect the current gene sequence."""
    #     self.order = OrderedDict((p.id, p) for p in self.genes)

    def _signature(self) -> tuple:
        """
        An immutable fingerprint including:
        - Gene signature: ((id, rotation), …) in gene order
        - Design parameters hash (if available)
        """
        # Get signature based on genes
        gene_signature = tuple((p.id, p.rotation) for p in self.genes)
        
        # Include design params hash if available
        if self.design_params is not None:
            try:
                # Convert design params to a stable string representation and hash it
                design_params_str = json.dumps(self.design_params, sort_keys=True)
                design_params_hash = hash(design_params_str)
                # Return a tuple containing both the gene signature and design params hash
                return (gene_signature, design_params_hash)
            except (TypeError, ValueError) as e:
                # If design params can't be serialized to JSON, fallback to a simple id
                if config.VERBOSE:
                    print(f"[Chromosome._signature] Warning: Could not hash design params: {e}")
                # Return just the gene signature with a dummy hash
                return (gene_signature, hash(id(self.design_params)))
        
        # If no design params, just return the gene signature
        return (gene_signature, None)

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
