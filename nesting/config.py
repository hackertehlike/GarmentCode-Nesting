# nesting/config.py

import math
import json
import hashlib
from typing import Literal, Mapping

# GUI STUFF
NUM_COPIES = 0

CONTAINER_WIDTH_CM  = 800
CONTAINER_HEIGHT_CM = 140

SEAM_ALLOWANCE_CM = 1

# ——— general settings —————————————————————————————————————
MULTITHREADING: bool = True
VERBOSE: bool = True
# DEFAULT_PATTERN_PATH: str = "nesting-assets/pattern_files/circle_skirt/circle_skirt_specification.json"
# DEFAULT_DESIGN_PARAM_PATH: str = "nesting-assets/pattern_files/circle_skirt/circle_skirt_design_params.yaml"
# DEFAULT_BODY_PARAM_PATH: str = "nesting-assets/pattern_files/circle_skirt/circle_skirt_body_measurements.yaml"

DEFAULT_PATTERN_PATH: str = "nesting-assets/pattern_files/rand_04ANOD2PBA/rand_04ANOD2PBA_specification.json"
DEFAULT_DESIGN_PARAM_PATH: str = "nesting-assets/pattern_files/rand_04ANOD2PBA/rand_04ANOD2PBA_design_params.yaml"
# DEFAULT_DESIGN_PARAM_PATH: str = ""
DEFAULT_BODY_PARAM_PATH: str = "nesting-assets/pattern_files/rand_04ANOD2PBA/rand_04ANOD2PBA_body_measurements.yaml"
# DEFAULT_BODY_PARAM_PATH: str = ""

DecoderName = Literal["BL", "Greedy", "NFP", "Random"]
MetricName  = Literal["usage_bb", "concave_hull", "concave_hull_area", "rest_length", "rest_height", "cc_with_rest_height", "cc_with_rest_length", "bb_cc", "bb_cc_area"]
#CrossoverName = Literal["pmx", "ox1"]
CrossoverName = Literal["oxk", "cross_stitch_oxk"]
SortKey = Literal["bbox_area", "hull_area", "aspect_ratio"]
CrossStitchMode = Literal["sticky", "lexicographic"]


# ——— algorithm settings —————————————————————————————————————
SELECTED_DECODER       : DecoderName = "NFP"
PRESERVE_HOLES: bool = True  # whether to preserve holes in the layout
SELECTED_FITNESS_METRIC: MetricName  = "usage_bb"
SELECTED_CROSSOVER      : CrossoverName = "cross_stitch_oxk"  # can be "pmx" or "ox1" or "oxk" or "cross_stitch_oxk"
# When True, crossover operations will produce only one child per mating.
# When False, crossover can return two children and the GA will consume both.
#SINGLE_CHILD_CROSSOVER: bool = True
CROSS_STITCH_MODE = "sticky"
OX_K = 1
NUM_SPLITS = 1  # number of splits for the split mutation operator
SPLIT_LOWER_BOUND = 0.3  # lower bound for split proportion
SPLIT_UPPER_BOUND = 0.7  # upper bound for split proportion


# ——— decoder settings —————————————————————————————————————
# BL
GRAVITATE_ONCE: bool = False  # whether to gravitate the pattern once or continuously
GRAVITATE_STEP = 1
# Greedy
SORT_BY = "hull_area"  # can be "bbox_area", "hull_area", or "aspect_ratio"
REST_PENALTY = 0.01  # penalty for rest length, since rest length is in centimeters and cc is in percentage, this should be a small value
BB_WEIGHT = 0.5  # weight for bounding box utilization in combined fitness metric
CC_WEIGHT = 0.5  # weight for concave hull utilization in combined fitness metric

# NFP
NFP_GRAVITATE_ON: bool = True  # whether to gravitate after NFP placement

# ——— genetic algorithm —————————————————————————————————————
POPULATION_SIZE       = 100
NUM_GENERATIONS       = 20
MUTATION_RATE         = 0.2
FORCE_MUTATION_ON_CROSSOVER: bool = False  # force mutation if offspring has same fitness as a parent
SWAP_MUTATION_K = None

POPULATION_WEIGHTS: Mapping[str, float] = {
    "elites": 0.25,  # weight for elite population
    "offspring": 0.25,  # weight for offspring population
    "mutants": 0.25,  # weight for mutants population
    "randoms": 0.25,  # weight for random population
}


# Parameters that should be excluded from design parameter mutations
EXCLUDED_PARAM_PATHS = [
    # ─── high-level garment selectors ──────────────────────────────────────
    "*meta*",                # meta.upper / meta.wb / meta.bottom
    "*.base",                # levels-skirt.base, godet-skirt.base
    "*.level",               # levels-skirt.level

    # ─── categorical style picks (select / select_null) ───────────────────
    "*_collar",              # collar.{f_,b_}collar  (+ left.* equivalents)
    "*.component.style",     # collar.component.style
    "*.cuff.type",           # sleeve / pants cuff style
    "*.panel_curve",         # flare-skirt.skirt-many-panels.panel_curve
    "*.num_inserts",         # godet-skirt.num_inserts
    "*style_side_cut*",      # pencil-skirt.style_side_cut

    # ─── boolean feature toggles / flags ───────────────────────────────────
    "*strapless*",           # shirt & left.shirt strapless flags
    "*sleeveless*",          # sleeve & left.sleeve
    "*enable_asym*",         # left.enable_asym
    "*flip*",                # collar & left.collar flip curve flags
    "*standing_shoulder*",   # sleeve + left.sleeve standing shoulder flags
    "*lapel_standing*",      # collar.component.lapel_standing
    "*sleeve.armhole_shape*",# armhole shape selector
    "*cut.add*",             # flare-skirt.cut.add boolean

    # ─── misc. discrete counts / selectors that drive structure ───────────
    "*.n_panels",            # number of panels (discrete structural change)
    "*panel_curve*",         # shape of each panel (categorical)
]


# Parameter change margin for design parameter mutations (percentage)
# Can be a single number (symmetric margin) or a tuple/list of (min_change, max_change)
# Example: 0.2 means parameter can change by up to ±20% of its value/range
# Example: (-0.1, 0.3) means parameter can decrease by up to 10% and increase by up to 30%
# Note: For parameters that can be both positive and negative (like opening_dir_mix),
# special handling is applied to prevent excessive changes when values are close to zero
PARAM_CHANGE_MARGIN = 0.2

SYMMETRIC_SPLITS: bool = False  # whether to split the pattern symmetrically (e.g., left and right halves of a skirt)
ALLOW_RECURSIVE_SPLITS: bool = False  # whether to allow recursive splits (e.g., splitting a split pattern again)
WEIGHT_BY_BBOX: bool = False  # whether to weight the split selection by bounding box area


# mutation weights
MUTATION_WEIGHTS = {
    "rotate":    0.14,
    "swap":      0.14,
    "inversion": 0.14,
    "insertion": 0.14,
    "scramble":  0.14,
    "split":     0.15,
    "design_params": 0.15
}


# dynamic stopping and extension for GA
ENABLE_DYNAMIC_STOPPING: bool = True
EARLY_STOP_WINDOW: int         = 10
EARLY_STOP_TOLERANCE: float    = 0.01
ENABLE_EXTENSION: bool         = False
EXTEND_WINDOW: int             = 10
EXTEND_THRESHOLD: float        = 0.1
MAX_GENERATIONS: int    = 20
GENERATION_PER_FLUSH: int = max(1, min(math.ceil(100 / POPULATION_SIZE), 10))

# How often to flush log lines to disk (in generations)
LOG_FLUSH_INTERVAL: int = GENERATION_PER_FLUSH

# log
SAVE_LOGS = True
# Unified experiments directory structure (replaces separate run_logs/results/aggregate_stats)
EXPERIMENTS_ROOT = "nesting/experiments"
RUNS_DIR = f"{EXPERIMENTS_ROOT}/runs"
AGGREGATE_DIR = f"{EXPERIMENTS_ROOT}/aggregate"
# Primary logs & per-run artifacts directory
SAVE_LOGS_PATH = f"{RUNS_DIR}/"
LOG_TIME = True
LOG_DESIGN_PARAM_PATHS = False
SAVE_GENERATION_SVGS = True

# ——— concave hull —————————————————————————————————————
HULL_TRIM_RATIO = 10 # higher number -> more convex
INTERIOR_SAMPLE_SPACING = 5 # how many cm between sampled interior points, tradeoff between speed and accuracy of the hull
BOUNDARY_SAMPLE_SPACING = 3 # how many cm between sampled boundary points, tradeoff between speed and accuracy of the hull
SNAP = False
SNAP_TOLERANCE = 0.1 # how close points must be to snap to the hull, in percentage of the container size


# ——— sampling (path extractor) —————————————————————————————————————
SAMPLES_PER_EDGE  = 7
ENABLE_ROTATIONS  = True
ALLOWED_ROTATIONS = [0, 180]  # allowed rotations in degrees, if ENABLE_ROTATIONS is True


_EXPORT_KEYS = [
    # general / algorithm
    "MULTITHREADING",
    "SELECTED_DECODER",
    "SELECTED_FITNESS_METRIC",
    "SELECTED_CROSSOVER",
    "CROSS_STITCH_MODE",

    # container / sampling
    "CONTAINER_WIDTH_CM",
    "CONTAINER_HEIGHT_CM",
    "SEAM_ALLOWANCE_CM",
    "SAMPLES_PER_EDGE",
    "ENABLE_ROTATIONS",
    "ALLOWED_ROTATIONS",
    "GRAVITATE_STEP",

    # GA
    "POPULATION_SIZE",
    "NUM_GENERATIONS",
    "MUTATION_RATE",
    "ENABLE_DYNAMIC_STOPPING",
    "EARLY_STOP_WINDOW",
    "EARLY_STOP_TOLERANCE",
    "ENABLE_EXTENSION",
    "EXTEND_WINDOW",
    "EXTEND_THRESHOLD",
    "MAX_GENERATIONS",
    "MUTATION_WEIGHTS",
    "POPULATION_WEIGHTS",

    # logging / experiments
    "SAVE_LOGS",
    "LOG_FLUSH_INTERVAL",
    "SAVE_LOGS_PATH",
    "LOG_TIME",
    "LOG_DESIGN_PARAM_PATHS",
    "EXPERIMENTS_ROOT",
    "RUNS_DIR",
    "AGGREGATE_DIR",

    # geometry / hull
    "HULL_TRIM_RATIO",
    "INTERIOR_SAMPLE_SPACING",
    "BOUNDARY_SAMPLE_SPACING",
    "SNAP",
    "SNAP_TOLERANCE",

    # UI / misc
    "NUM_COPIES",
    "NUM_SPLITS",
    "OX_K",
    "PRESERVE_HOLES",
    "EXCLUDED_PARAM_PATHS",
    "PARAM_CHANGE_MARGIN",
]

def as_dict() -> dict:
    """Return the selected configuration values as a dictionary.

    Uses an explicit allow-list of keys for clarity and stability.
    """
    g = globals()
    return {k: g[k] for k in _EXPORT_KEYS}


def __dict__() -> dict:
    return as_dict()


# Stable JSON and hash helpers for config
def stable_config_json() -> str:
    """Return a deterministic JSON string of the exported config (uses as_dict()).

    Keys are sorted and compact separators are used for stable output.
    """
    return json.dumps(as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_config_hash() -> str:
    """Return a stable SHA256 hex digest for the exported config values."""
    return hashlib.sha256(stable_config_json().encode("utf-8")).hexdigest()
