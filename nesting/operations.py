import random
import copy
from typing import Callable, List, Any, Sequence, Optional, Tuple, Union
from fnmatch import fnmatch
from pathlib import Path
from .layout import Piece, LayoutView
from .placement_engine import DECODER_REGISTRY
from pygarment.garmentcode.utils import nested_get, nested_set, nested_del
import nesting.config as config
from nesting.panel_mapping import affected_panels

JsonDict = dict[str, Any]

# Unified metric registry
METRIC_REGISTRY: dict[str, Callable] = {}

def register_metric(name: str):
    def deco(fn):
        METRIC_REGISTRY[name] = fn
        return fn
    return deco

def run_decoder(pieces: List[Piece], decoder: str, container=None):
    """Helper to instantiate & run the placement decoder."""
    view = LayoutView(pieces)
    Decoder = DECODER_REGISTRY[decoder]
    if container:
        dec = Decoder(view, container)
    else:
        dec = Decoder(view)
    dec.decode()
    return dec

# ---------------------------------- FITNESS METRICS --------------------------------- #

@register_metric("usage_bb")
def fitness_usage_bb(pieces: List[Piece], decoder: str, container=None):
    dec = run_decoder(pieces, decoder, container)
    return dec.usage_BB()

@register_metric("concave_hull")
def fitness_concave_hull(pieces: List[Piece], decoder: str, container=None):
    dec = run_decoder(pieces, decoder, container)
    return dec.concave_hull_utilization()

@register_metric("concave_hull_area")
# TODO: change the constant to be the container size
def fitness_concave_hull_area(pieces: List[Piece], decoder: str, container=None):
    dec = run_decoder(pieces, decoder, container)
    area = dec.concave_hull_area()
    if area < 1e-6:
        return 0
    return 10000/area

@register_metric("rest_length")
def fitness_rest_length(pieces: List[Piece], decoder: str, container=None):
    dec = run_decoder(pieces, decoder, container)
    return dec.rest_length()

@register_metric("rest_height")
def fitness_rest_height(pieces: List[Piece], decoder: str, container=None):
    dec = run_decoder(pieces, decoder, container)
    return dec.rest_height()

@register_metric("cc_with_rest_height")
def fitness_cc_height_combined(pieces: List[Piece], decoder: str, container=None):
    """Combined fitness metric for concave hull height and rest height."""
    dec = run_decoder(pieces, decoder, container)
    cc = dec.concave_hull_utilization()
    rest_height = dec.rest_height()
    if cc == 0:
        return 0
    return cc + rest_height / config.CONTAINER_HEIGHT_CM

@register_metric("cc_with_rest_length")
def fitness_cc_length_combined(pieces: List[Piece], decoder: str, container=None):
    """Combined fitness metric for concave hull length and rest length."""
    dec = run_decoder(pieces, decoder, container)
    cc = dec.concave_hull_utilization()
    rest_length = dec.rest_length()
    if cc == 0:
        return 0
    return cc + rest_length / config.CONTAINER_WIDTH_CM

@register_metric("bb_cc")
def fitness_bb_cc(pieces: List[Piece], decoder: str, container=None):
    """Combined fitness metric for bounding box area and concave hull area."""
    dec = run_decoder(pieces, decoder, container)
    bb_area = dec.usage_BB()
    cc_area = dec.concave_hull_utilization()
    if bb_area == 0 or cc_area == 0:
        return 0
    return config.BB_WEIGHT * bb_area + config.CC_WEIGHT * cc_area

@register_metric("bb_area")
# TODO: change the constant to be the container size
def fitness_bb_area(pieces: List[Piece], decoder: str, container=None):
    dec = run_decoder(pieces, decoder, container)
    bb_area = dec.bbox_area()
    if bb_area == 0:
        return 0
    return 10000 / bb_area

@register_metric("bb_cc_area")
# TODO: change the constant to be the container size
def fitness_bb_cc_area(pieces: List[Piece], decoder: str, container=None):
    dec = run_decoder(pieces, decoder, container)
    bb_area = dec.bbox_area()
    cc_area = dec.concave_hull_area()
    if bb_area == 0 or cc_area == 0:
        return 0
    return 10000 / (config.BB_WEIGHT * bb_area + config.CC_WEIGHT * cc_area)

@register_metric("bb_with_rest_length")
def fitness_bb_length_combined(pieces: List[Piece], decoder: str, container=None):
    """Combined fitness metric for bounding box length and rest length."""
    dec = run_decoder(pieces, decoder, container)
    bb = dec.usage_BB()
    rest_length = dec.rest_length()
    if bb == 0:
        return 0
    return bb + rest_length / config.CONTAINER_WIDTH_CM

# ---------------------------------- CHROMOSOME-COMPATIBLE WRAPPERS --------------------------------- #
# These maintain backward compatibility with code that passes Chromosome objects

def fitness_rest_length_chromosome(chromosome, decoder: str):
    """Wrapper for backward compatibility with Chromosome objects."""
    return fitness_rest_length(chromosome.genes, decoder, getattr(chromosome, 'container', None))

def fitness_usage_bb_chromosome(chromosome, decoder: str):
    """Wrapper for backward compatibility with Chromosome objects."""
    return fitness_usage_bb(chromosome.genes, decoder, getattr(chromosome, 'container', None))

def fitness_concave_hull_area_chromosome(chromosome, decoder: str):
    """Wrapper for backward compatibility with Chromosome objects."""
    return fitness_concave_hull_area(chromosome.genes, decoder, getattr(chromosome, 'container', None))

def fitness_concave_hull_chromosome(chromosome, decoder: str):
    """Wrapper for backward compatibility with Chromosome objects."""
    return fitness_concave_hull(chromosome.genes, decoder, getattr(chromosome, 'container', None))

def fitness_bb_area_chromosome(chromosome, decoder: str):
    """Wrapper for backward compatibility with Chromosome objects."""
    return fitness_bb_area(chromosome.genes, decoder, getattr(chromosome, 'container', None))

def fitness_bb_cc_area_chromosome(chromosome, decoder: str):
    """Wrapper for backward compatibility with Chromosome objects."""
    return fitness_bb_cc_area(chromosome.genes, decoder, getattr(chromosome, 'container', None))

def fitness_bb_length_combined_chromosome(chromosome, decoder: str):
    """Wrapper for backward compatibility with Chromosome objects."""
    return fitness_bb_length_combined(chromosome.genes, decoder, getattr(chromosome, 'container', None))

# ---------------------------------- MUTATION OPERATIONS --------------------------------- #

class Operators:
    """Shared mutation operations for both Chromosome and SimulatedAnnealing classes."""
    
    @staticmethod
    def rotate(pieces: List[Piece]) -> List[Piece]:
        """Rotate a random non-empty subset of pieces ensuring at least one actual rotation change.

        Falls back to forcing a change on one piece if random sampling produces no net change.
        """
        new_pieces = copy.deepcopy(pieces)
        if len(new_pieces) == 0:
            return new_pieces
        count = random.randint(1, len(new_pieces))
        chosen_indices = random.sample(range(len(new_pieces)), count)
        changed = False
        for idx in chosen_indices:
            piece = new_pieces[idx]
            current_rot = getattr(piece, 'rotation', 0)
            possible = [r for r in config.ALLOWED_ROTATIONS if r != current_rot]
            if not possible:  # all rotations equivalent / only one allowed
                continue
            piece.rotate(random.choice(possible))
            changed = True
        if not changed:
            # Force rotate a single piece (if possible)
            for piece in new_pieces:
                current_rot = getattr(piece, 'rotation', 0)
                possible = [r for r in config.ALLOWED_ROTATIONS if r != current_rot]
                if possible:
                    piece.rotate(random.choice(possible))
                    break
        return new_pieces

    @staticmethod
    def swap(pieces: List[Piece], k=None) -> List[Piece]:
        """Swap two non-overlapping contiguous blocks of length k."""
        new_pieces = copy.deepcopy(pieces)
        n = len(new_pieces)
        
        if n < 2:
            return new_pieces
            
        max_k = n // 2  # must allow two non-overlapping blocks
        if max_k < 1:
            return new_pieces
            
        if k is None:
            k = config.SWAP_MUTATION_K if hasattr(config, 'SWAP_MUTATION_K') and config.SWAP_MUTATION_K else random.randint(1, max_k)
        
        k = min(k, max_k) if k else 1
        
        # Build all valid non-overlapping (i, j) with i < j and j >= i + k
        pairs: List[tuple[int, int]] = []
        last_start = n - k
        for i in range(0, n - 2 * k + 1):
            for j in range(i + k, last_start + 1):
                pairs.append((i, j))

        if not pairs:
            return new_pieces

        i, j = random.choice(pairs)

        # Splice: pieces = pre + B + mid + A + post
        pre = new_pieces[:i]
        A = new_pieces[i:i + k]
        mid = new_pieces[i + k:j]
        B = new_pieces[j:j + k]
        post = new_pieces[j + k:]
        
        return pre + B + mid + A + post

    @staticmethod
    def local_swap(pieces: List[Piece], max_distance: int = 3) -> List[Piece]:
        """Swap two individual pieces whose indices differ by at most ``max_distance``.

        Guarantees an order change if at least one valid pair exists. Falls back to
        the original list copy if n < 2 (no swap possible).

        Args:
            pieces: Current ordered list of pieces.
            max_distance: Maximum allowed index gap (inclusive) between the two
                pieces selected for swapping. Default = 3 per user request.
        Returns:
            A new list of pieces with the selected pair swapped (deep copied list).
        """
        new_pieces = copy.deepcopy(pieces)
        n = len(new_pieces)
        if n < 2:
            return new_pieces

        # Collect all valid (i, j) pairs with 1 <= j - i <= max_distance
        pairs: list[tuple[int, int]] = []
        for i in range(n - 1):
            upper = min(n - 1, i + max_distance)
            for j in range(i + 1, upper + 1):
                pairs.append((i, j))

        if not pairs:
            return new_pieces

        i, j = random.choice(pairs)
        new_pieces[i], new_pieces[j] = new_pieces[j], new_pieces[i]
        return new_pieces

    @staticmethod
    def inversion(pieces: List[Piece]) -> List[Piece]:
        """Reverse a random subsequence with length >= 2; retry limited times to ensure change."""
        new_pieces = copy.deepcopy(pieces)
        n = len(new_pieces)
        if n < 2:
            return new_pieces
        for _ in range(5):
            i, j = sorted(random.sample(range(n), 2))
            if j - i >= 1:  # length at least 2
                slice_before = [(p.id, getattr(p, 'rotation', None)) for p in new_pieces[i:j+1]]
                reversed_slice = list(reversed(new_pieces[i:j+1]))
                slice_after = [(p.id, getattr(p, 'rotation', None)) for p in reversed_slice]
                if slice_after != slice_before:
                    new_pieces[i:j+1] = reversed_slice
                    break
        return new_pieces

    @staticmethod
    def insertion(pieces: List[Piece]) -> List[Piece]:
        """Remove a piece and insert it at a different position (guaranteed)."""
        new_pieces = copy.deepcopy(pieces)
        n = len(new_pieces)
        if n < 2:
            return new_pieces
        i = random.randrange(n)
        # Choose a different insertion point
        choices = [pos for pos in range(n) if pos != i and pos != i+1]
        if not choices:
            # fallback: any different index
            choices = [pos for pos in range(n) if pos != i]
        insert_at = random.choice(choices)
        piece = new_pieces.pop(i)
        if insert_at > i:
            insert_at -= 1  # list shrank by one
        new_pieces.insert(insert_at, piece)
        return new_pieces

    @staticmethod
    def scramble(pieces: List[Piece]) -> List[Piece]:
        """Randomly shuffle a subsequence (length >=2) ensuring an order change or retry."""
        new_pieces = copy.deepcopy(pieces)
        n = len(new_pieces)
        if n < 2:
            return new_pieces
        for _ in range(5):
            i, j = sorted(random.sample(range(n), 2))
            if j - i < 1:
                continue
            before = [p.id for p in new_pieces[i:j+1]]
            subset = new_pieces[i:j+1]
            random.shuffle(subset)
            after = [p.id for p in subset]
            if after != before:
                new_pieces[i:j+1] = subset
                break
        return new_pieces

    @staticmethod
    def design_params(
        pieces: List[Piece],
        design_params: Optional[JsonDict],
        body_params: Optional[Any],
        initial_design_params: Optional[JsonDict],
        split_history: Optional[List[Any]] = None,
        fitness_fn: Optional[Callable] = None
    ) -> Tuple[List[Piece], Optional[JsonDict], bool]:
        """
        Mutate design parameters and return updated pieces, params, and success status.
        
        Returns:
            (new_pieces, new_design_params, success)
        """
        if not (design_params and body_params):
            return pieces, design_params, False

        mutatable_params = _collect_mutatable_params(design_params, pieces)
        if not mutatable_params:
            return pieces, design_params, False

        # Get current fitness if fitness function provided
        original_fitness = fitness_fn(pieces) if fitness_fn else None
        original_design_params = copy.deepcopy(design_params)
        original_pieces = copy.deepcopy(pieces)

        # Shuffle and try parameters
        params_to_try = list(mutatable_params)
        random.shuffle(params_to_try)

        for param in params_to_try:
            # Reset to original state for each attempt
            current_design_params = copy.deepcopy(original_design_params)
            current_pieces = copy.deepcopy(original_pieces)

            node = nested_get(current_design_params, param.split("."))
            p_type = node.get("type", "float")
            old_val = node["v"]
            base_val = nested_get(initial_design_params, param.split("."))["v"]
            new_val = _random_value(old_val, base_val, p_type, node["range"])

            if old_val == new_val:
                continue

            # Apply the parameter change
            nested_set(current_design_params, param.split(".") + ["v"], new_val)

            # Apply design parameter change and regenerate pieces
            success, updated_pieces = _apply_design_param_change(
                param, old_val, new_val, current_pieces, current_design_params, 
                body_params, split_history or []
            )

            if success:
                # Check if fitness changed (if we have a fitness function)
                if fitness_fn:
                    new_fitness = fitness_fn(updated_pieces)
                    if new_fitness != original_fitness:
                        return updated_pieces, current_design_params, True
                else:
                    # No fitness function, assume success
                    return updated_pieces, current_design_params, True

        # No successful parameter change
        return pieces, design_params, False

def _apply_design_param_change(
    path: str,
    old_val: Any,
    new_val: Any, 
    pieces: List[Piece],
    design_params: JsonDict,
    body_params: Any,
    split_history: List[Any]
) -> Tuple[bool, List[Piece]]:
    """Helper to regenerate garment pieces after a design param change."""
    try:
        from assets.garment_programs.meta_garment import MetaGarment
        from nesting.panel_mapping import select_genes
        from nesting.path_extractor import PatternPathExtractor
        import tempfile

        panel_ids = {g.id for g in pieces}
        affected = affected_panels([path], design_params)

        if not any(fnmatch(pid, pat) for pat in affected for pid in panel_ids):
            if config.VERBOSE:
                print(f"[DesignParams] No affected panels for {path}")
            return False, pieces

        # Debug: Log MetaGarment initialization parameters
        print(f"[DesignParams] DEBUG: Creating MetaGarment with body_params type: {type(body_params)}")
        print(f"[DesignParams] DEBUG: design_params keys: {list(design_params.keys()) if design_params else None}")
        if design_params and 'meta' in design_params:
            meta_params = design_params['meta']
            print(f"[DesignParams] DEBUG: meta params: upper={meta_params.get('upper', {}).get('v')}, bottom={meta_params.get('bottom', {}).get('v')}, wb={meta_params.get('wb', {}).get('v')}")
        
        mg = MetaGarment("design_mut", body_params, design_params)
        
        # Debug: Check MetaGarment composition after creation
        print(f"[DesignParams] DEBUG: MetaGarment created - upper_name: {getattr(mg, 'upper_name', 'MISSING')}, lower_name: {getattr(mg, 'lower_name', 'MISSING')}, belt_name: {getattr(mg, 'belt_name', 'MISSING')}")
        print(f"[DesignParams] DEBUG: MetaGarment subs count: {len(getattr(mg, 'subs', []))}")

        # Debug: Check what panels exist in regenerated MetaGarment BEFORE reapplying splits
        try:
            temp_pattern = mg.assembly()
            available_panels = list(temp_pattern.pattern['panels'].keys())
            print(f"[DesignParams] DEBUG: Panels in regenerated MetaGarment: {available_panels}")
            print(f"[DesignParams] DEBUG: Split history to reapply: {split_history}")
        except Exception as debug_e:
            print(f"[DesignParams] DEBUG: Could not get panel list from regenerated MetaGarment: {debug_e}")
            print(f"[DesignParams] DEBUG: Exception type: {type(debug_e)}, message: {str(debug_e)}")

        # Restore splits from history
        for panel_name, proportion in split_history:
            if config.VERBOSE:
                print(f"[DesignParams] Reapplying split for {panel_name} with proportion {proportion}")
            try:
                mg.split_panel(panel_name, proportion)
                print(f"[DesignParams] DEBUG: Successfully split {panel_name}")
            except Exception as split_error:
                print(f"[DesignParams] DEBUG: Failed to split {panel_name}: {split_error}")
                # Continue trying other splits rather than failing completely
                continue

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
        new_piece_ids = set(new_pieces.keys())

        # Extend changed ids for split fragments
        extended_changed_ids = set(changed_ids)
        root_ids_changed = {cid for cid in changed_ids if cid in new_piece_ids}
        for root_id in root_ids_changed:
            for g in pieces:
                if g.root_id == root_id and g.id in new_piece_ids:
                    extended_changed_ids.add(g.id)

        # Update pieces
        updated_pieces = copy.deepcopy(pieces)
        for i, g in enumerate(updated_pieces):
            if g.id in extended_changed_ids and g.id in new_pieces:
                replacement = copy.deepcopy(new_pieces[g.id])
                replacement.rotation = g.rotation
                replacement.parent_id = g.parent_id
                replacement.root_id = g.root_id
                updated_pieces[i] = replacement

        return True, updated_pieces

    except Exception as e:
        if config.VERBOSE:
            print(f"[DesignParams] Failed to apply design parameter change: {e}")
        return False, pieces

def weighted_choice(options: dict[str, float]) -> str:
    """Return a key from *options* using their values as weights."""
    choices, weights = zip(*options.items())
    return random.choices(choices, weights)[0]

# ---------------------------------- DESIGN PARAMETER HELPERS --------------------------------- #

def _flatten_param_paths(node: JsonDict, prefix: Optional[List[str]] = None) -> List[str]:
    """Return all *non‑meta* leaf‑paths that contain a ``v`` key."""
    prefix = prefix or []
    paths: list[str] = []
    for k, v in node.items():
        if isinstance(v, dict):
            if "v" in v:
                paths.append(".".join(prefix + [k]))
            else:
                paths += _flatten_param_paths(v, prefix + [k])
    return paths

def _numeric_range_ok(node: JsonDict) -> bool:
    rng = node.get("range", [])
    return len(rng) >= 2 and all(isinstance(x, (int, float)) for x in rng[:2])

def _random_value(old: Any, base: Any, p_type: str, rng: Sequence[Union[float, int]]):
    """Return a *new* value that differs from *base* by ≤ margin param % of *range* width
    while still being different from *old*.

    The *base* value represents the original design parameter value before any
    mutations, which allows us to limit cumulative drift across generations.
    """
    lower, upper = rng[0], rng[1]
    span = upper - lower
    max_delta = config.PARAM_CHANGE_MARGIN * span

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
    genes: List[Piece]
) -> List[str]:
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
    
    # filter out the ones that don't affect any panel in this chromosome
    panel_ids = {g.id for g in genes}
    mutatable = [p for p in numeric_ok if (panel_patterns := affected_panels([p], design_params)) 
                 and any(fnmatch(pid, pat) for pat in panel_patterns for pid in panel_ids)]

    return mutatable