# nesting/config.py

import math
from typing import Literal, Mapping

# ——— general settings —————————————————————————————————————
MULTITHREADING: bool = True
VERBOSE: bool = True


DecoderName = Literal["BL", "Greedy", "NFP", "Random"]
MetricName  = Literal["usage_bb", "concave_hull", "rest_length"]
CrossoverName = Literal["pmx", "ox1", "ox1k"]
# ——— algorithm settings —————————————————————————————————————
SELECTED_DECODER       : DecoderName = "NFP"
SELECTED_FITNESS_METRIC: MetricName  = "concave_hull"
SELECTED_CROSSOVER      : CrossoverName = "ox1k"


# ——— placement settings —————————————————————————————————————
GRAVITATE_STEP = 2

# ——— genetic algorithm —————————————————————————————————————
POPULATION_SIZE       = 100
NUM_GENERATIONS       = 100
MUTATION_RATE         = 0.1

POPULATION_WEIGHTS: Mapping[str, float] = {
    "elites": 0.1,  # weight for elite population
    "offspring": 0.2,  # weight for offspring population
    "mutants": 0.4,  # weight for mutants population
    "randoms": 0.3,  # weight for random population
}


# mutation weights
MUTATION_WEIGHTS = {
    "rotate":    0.3,
    "swap":      0.2,
    "inversion": 0,
    "insertion": 0.1,
    "scramble":  0.4,
    "split":     0,   # keep zero until you implement it
}


# dynamic stopping and extension for GA
ENABLE_DYNAMIC_STOPPING: bool = False
EARLY_STOP_WINDOW: int         = 10
EARLY_STOP_TOLERANCE: float    = 1e-4
ENABLE_EXTENSION: bool         = True
EXTEND_WINDOW: int             = 10
EXTEND_THRESHOLD: float        = 0.1
MAX_GENERATIONS: int    = 200
GENERATION_PER_FLUSH: int = max(1, min(math.ceil(100 / POPULATION_SIZE), 10))

# log
SAVE_LOGS = True
SAVE_LOGS_PATH = "nesting/run_logs"
LOG_TIME = True

# ——— concave hull —————————————————————————————————————
HULL_TRIM_RATIO = 10 # higher number -> more convex
INTERIOR_SAMPLE_SPACING = 5 # how many cm between sampled interior points, tradeoff between speed and accuracy of the hull
BOUNDARY_SAMPLE_SPACING = 3 # how many cm between sampled boundary points, tradeoff between speed and accuracy of the hull
SNAP_TOLERANCE = 10 # how close points must be to snap to the hull, in cm


# ——— sampling (path extractor) —————————————————————————————————————
SAMPLES_PER_EDGE  = 10
ENABLE_ROTATIONS  = True
ALLOWED_ROTATIONS = [0, 90, 180, 270]


# GUI STUFF
SAMPLES_PER_EDGE = 5
NUM_COPIES = 0

CONTAINER_WIDTH_CM  = 500
CONTAINER_HEIGHT_CM = 500.0

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
        "LOG_TIME": LOG_TIME
    }
    