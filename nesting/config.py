# nesting/config.py

import math
from typing import Literal, Mapping

# ——— general settings —————————————————————————————————————
MULTITHREADING: bool = False
VERBOSE: bool = True
# DEFAULT_PATTERN_PATH: str = "nesting-assets/pattern_files/circle_skirt/circle_skirt_specification.json"
# DEFAULT_DESIGN_PARAM_PATH: str = "nesting-assets/pattern_files/circle_skirt/circle_skirt_design_params.yaml"
# DEFAULT_BODY_PARAM_PATH: str = "nesting-assets/pattern_files/circle_skirt/circle_skirt_body_measurements.yaml"

DEFAULT_PATTERN_PATH: str = "nesting-assets/pattern_files/rand_0A36YXPNV0/rand_0A36YXPNV0_specification.json"
DEFAULT_DESIGN_PARAM_PATH: str = "nesting-assets/pattern_files/rand_0A36YXPNV0/rand_0A36YXPNV0_design_params.yaml"
DEFAULT_BODY_PARAM_PATH: str = "nesting-assets/pattern_files/rand_0A36YXPNV0/rand_0A36YXPNV0_body_measurements.yaml"


DecoderName = Literal["BL", "Greedy", "NFP", "Random"]
MetricName  = Literal["usage_bb", "concave_hull", "rest_length", "rest_height", "cc_with_rest_height", "cc_with_rest_length", "bb_cc"]
#CrossoverName = Literal["pmx", "ox1"]
CrossoverName = Literal["oxk"]
SortKey = Literal["bbox_area", "hull_area", "aspect_ratio"]

# ——— algorithm settings —————————————————————————————————————
SELECTED_DECODER       : DecoderName = "NFP"
PRESERVE_HOLES: bool = True  # whether to preserve holes in the layout
SELECTED_FITNESS_METRIC: MetricName  = "concave_hull"  # can be "usage_bb", "concave_hull", "rest_length", "rest_height", "cc_with_rest_height", "cc_with_rest_length", "bb_cc"
SELECTED_CROSSOVER      : CrossoverName = "ox1"
OX_CIRCULAR: bool = False  # whether to use circular walk in OX crossover
OX_K = 1
NUM_SPLITS = 1  # number of splits for the split mutation operator


# ——— decoder settings —————————————————————————————————————
# BL
GRAVITATE_ONCE: bool = False  # whether to gravitate the pattern once or continuously
GRAVITATE_STEP = 2
# Greedy
SORT_BY = "hull_area"  # can be "bbox_area", "hull_area", or "aspect_ratio"
REST_PENALTY = 0.01  # penalty for rest length in greedy placement, since rest length is in centimeters and cc is in percentage, this should be a small value
BB_WEIGHT = 0.5  # weight for bounding box utilization in combined fitness metric
CC_WEIGHT = 0.5  # weight for concave hull utilization in combined fitness metric

# NFP
NFP_GRAVITATE_ON: bool = True  # whether to gravitate after NFP placement

# ——— genetic algorithm —————————————————————————————————————
POPULATION_SIZE       = 100
NUM_GENERATIONS       = 20
MUTATION_RATE         = 0.2


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

SYMMETRIC_SPLITS: bool = True  # whether to split the pattern symmetrically (e.g., left and right halves of a skirt)


# mutation weights
MUTATION_WEIGHTS = {
    "rotate":    0.1,
    "swap":      0.1,
    "inversion": 0.1,
    "insertion": 0.1,
    "scramble":  0.1,
    "split":     0.25,
    "design_params": 0.25
}


# dynamic stopping and extension for GA
ENABLE_DYNAMIC_STOPPING: bool = True
EARLY_STOP_WINDOW: int         = 10
EARLY_STOP_TOLERANCE: float    = 0.01
ENABLE_EXTENSION: bool         = True
EXTEND_WINDOW: int             = 10
EXTEND_THRESHOLD: float        = 0.1
MAX_GENERATIONS: int    = 200
GENERATION_PER_FLUSH: int = max(1, min(math.ceil(100 / POPULATION_SIZE), 10))

# log
SAVE_LOGS = True
SAVE_LOGS_PATH = "nesting/run_logs/"
LOG_TIME = True
LOG_DESIGN_PARAM_PATHS = False
SAVE_GENERATION_SVGS = True

# ——— concave hull —————————————————————————————————————
HULL_TRIM_RATIO = 20 # higher number -> more convex
INTERIOR_SAMPLE_SPACING = 5 # how many cm between sampled interior points, tradeoff between speed and accuracy of the hull
BOUNDARY_SAMPLE_SPACING = 3 # how many cm between sampled boundary points, tradeoff between speed and accuracy of the hull
SNAP = False
SNAP_TOLERANCE = 0.1 # how close points must be to snap to the hull, in percentage of the container size


# ——— sampling (path extractor) —————————————————————————————————————
SAMPLES_PER_EDGE  = 7
ENABLE_ROTATIONS  = True
ALLOWED_ROTATIONS = [0, 90, 180, 270]  # allowed rotations in degrees, if ENABLE_ROTATIONS is True


# GUI STUFF
NUM_COPIES = 0

CONTAINER_WIDTH_CM  = 300
CONTAINER_HEIGHT_CM = 140

SEAM_ALLOWANCE_CM = 1

def __dict__() -> dict:
    """
    Return the configuration as a dictionary.
    """
    return {
        "MULTITHREADING": MULTITHREADING,
        "SELECTED_DECODER": SELECTED_DECODER,
        "SELECTED_FITNESS_METRIC": SELECTED_FITNESS_METRIC,
        "SELECTED_CROSSOVER": SELECTED_CROSSOVER,
        "CONTAINER_WIDTH_CM": CONTAINER_WIDTH_CM,
        "CONTAINER_HEIGHT_CM": CONTAINER_HEIGHT_CM,
        "SEAM_ALLOWANCE_CM": SEAM_ALLOWANCE_CM,
        "SAMPLES_PER_EDGE": SAMPLES_PER_EDGE,
        "ENABLE_ROTATIONS": ENABLE_ROTATIONS,
        "ALLOWED_ROTATIONS": ALLOWED_ROTATIONS,
        "GRAVITATE_STEP": GRAVITATE_STEP,
        "POPULATION_SIZE": POPULATION_SIZE,
        "NUM_GENERATIONS": NUM_GENERATIONS,
        "MUTATION_RATE": MUTATION_RATE,
        "ENABLE_DYNAMIC_STOPPING": ENABLE_DYNAMIC_STOPPING,
        "EARLY_STOP_WINDOW": EARLY_STOP_WINDOW,
        "EARLY_STOP_TOLERANCE": EARLY_STOP_TOLERANCE,
        "ENABLE_EXTENSION": ENABLE_EXTENSION,
        "EXTEND_WINDOW": EXTEND_WINDOW,
        "EXTEND_THRESHOLD": EXTEND_THRESHOLD,
        "MAX_GENERATIONS": MAX_GENERATIONS,
        "MUTATION_WEIGHTS": MUTATION_WEIGHTS,
        "SAVE_LOGS": SAVE_LOGS,
        "SAVE_LOGS_PATH": SAVE_LOGS_PATH,
        "LOG_TIME": LOG_TIME,
        "LOG_DESIGN_PARAM_PATHS": LOG_DESIGN_PARAM_PATHS,
        "HULL_TRIM_RATIO": HULL_TRIM_RATIO,
        "INTERIOR_SAMPLE_SPACING": INTERIOR_SAMPLE_SPACING,
        "BOUNDARY_SAMPLE_SPACING": BOUNDARY_SAMPLE_SPACING,
        "SNAP": SNAP,
        "SNAP_TOLERANCE": SNAP_TOLERANCE,
        "NUM_COPIES": NUM_COPIES,
        "NUM_SPLITS": NUM_SPLITS,
        "OX_K": OX_K,
        "POPULATION_WEIGHTS": POPULATION_WEIGHTS,
        "PRESERVE_HOLES": PRESERVE_HOLES,
        "EXCLUDED_PARAM_PATHS": EXCLUDED_PARAM_PATHS,
        "PARAM_CHANGE_MARGIN": PARAM_CHANGE_MARGIN
    }
