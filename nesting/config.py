# nesting/config.py

import math
from typing import Literal, Mapping

# ——— general settings —————————————————————————————————————
MULTITHREADING: bool = True
VERBOSE: bool = True
DEFAULT_PATTERN_PATH: str = "nesting-assets/default_pattern.json"
DEFAULT_DESIGN_PARAM_PATH: str = "nesting-assets/default_design_params.yaml"
DEFAULT_BODY_PARAM_PATH: str = "nesting-assets/default_body_measurements.yaml"

DecoderName = Literal["BL", "Greedy", "NFP", "Random", "Jostle"]
MetricName  = Literal["usage_bb", "concave_hull", "rest_length"]
CrossoverName = Literal["pmx", "ox1"]
SortKey = Literal["bbox_area", "hull_area", "aspect_ratio"]

# ——— algorithm settings —————————————————————————————————————
SELECTED_DECODER       : DecoderName = "NFP"
PRESERVE_HOLES: bool = True  # whether to preserve holes in the layout
SELECTED_FITNESS_METRIC: MetricName  = "concave_hull"
SELECTED_CROSSOVER      : CrossoverName = "ox1"
OX_K = 1
NUM_SPLITS = 1  # number of splits for the split mutation operator


# ——— placement settings —————————————————————————————————————
# BL
GRAVITATE_ONCE: bool = True  # whether to gravitate the pattern once before starting the GA
GRAVITATE_STEP = 2
# Greedy
SORT_BY = "hull_area"  # can be "bbox_area", "hull_area", or "aspect_ratio"


# ——— genetic algorithm —————————————————————————————————————
POPULATION_SIZE       = 40
NUM_GENERATIONS       = 15
MUTATION_RATE         = 0.5




POPULATION_WEIGHTS: Mapping[str, float] = {
    "elites": 0.25,  # weight for elite population
    "offspring": 0.25,  # weight for offspring population
    "mutants": 0.25,  # weight for mutants population
    "randoms": 0.25,  # weight for random population
}


# Parameters that should be excluded from design parameter mutations
EXCLUDED_PARAM_PATHS = [
    "*component*",  # Exclude all component style mutations
    "*range*",      # Exclude range modifications
    "*enable_asym*",
    "*strapless*",
    "*flip*",
    "*n_panels*"
]

# Parameter change margin for design parameter mutations (percentage)
# Can be a single number (symmetric margin) or a tuple/list of (min_change, max_change)
# Example: 0.2 means parameter can change by up to ±20% of its value/range
# Example: (-0.1, 0.3) means parameter can decrease by up to 10% and increase by up to 30%
# Note: For parameters that can be both positive and negative (like opening_dir_mix),
# special handling is applied to prevent excessive changes when values are close to zero
PARAM_CHANGE_MARGIN = 0.2


# mutation weights
MUTATION_WEIGHTS = {
    "rotate":    0.2,
    "swap":      0.1,
    "inversion": 0.1,
    "insertion": 0.1,
    "scramble":  0.1,
    "split":     0.2,
    "design_params": 0.2
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
HULL_TRIM_RATIO = 10 # higher number -> more convex
INTERIOR_SAMPLE_SPACING = 5 # how many cm between sampled interior points, tradeoff between speed and accuracy of the hull
BOUNDARY_SAMPLE_SPACING = 3 # how many cm between sampled boundary points, tradeoff between speed and accuracy of the hull
SNAP = False
SNAP_TOLERANCE = 0.1 # how close points must be to snap to the hull, in percentage of the container size


# ——— sampling (path extractor) —————————————————————————————————————
SAMPLES_PER_EDGE  = 4
ENABLE_ROTATIONS  = True
ALLOWED_ROTATIONS = [0, 90, 180, 270]  # allowed rotations in degrees, if ENABLE_ROTATIONS is True


# GUI STUFF
NUM_COPIES = 0

CONTAINER_WIDTH_CM  = 140
CONTAINER_HEIGHT_CM = 400

SEAM_ALLOWANCE_CM = 0

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
