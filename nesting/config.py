# nesting/config.py

from typing import Literal, Mapping

DecoderName = Literal["BL", "Greedy", "NFP", "Random"]
MetricName  = Literal["usage_bb", "concave_hull", "rest_length"]
CrossoverName = Literal["pmx", "ox1", "ox1k"]

# just the names + defaults; no import‐strings here
SELECTED_DECODER       : DecoderName = "BL"
SELECTED_FITNESS_METRIC: MetricName  = "usage_bb"
SELECTED_CROSSOVER      : CrossoverName = "pmx"


# ——— container defaults —————————————————————————————————————
CONTAINER_WIDTH_CM  = 140.0
CONTAINER_HEIGHT_CM = 500.0

# ——— sampling (path extractor) —————————————————————————————————————
SAMPLES_PER_EDGE  = 5
ENABLE_ROTATIONS  = True
ALLOWED_ROTATIONS = [0, 90, 180, 270]

# ——— placement settings —————————————————————————————————————
BL_STEP = 1.0

# ——— genetic algorithm —————————————————————————————————————
POPULATION_SIZE       = 10
NUM_GENERATIONS       = 2
ELITE_POPULATION_SIZE = 5
MUTATION_RATE         = 0.1

# dynamic stopping and extension for GA
ENABLE_DYNAMIC_STOPPING: bool = True
EARLY_STOP_WINDOW: int         = 20
EARLY_STOP_TOLERANCE: float    = 1e-4
ENABLE_EXTENSION: bool         = True
EXTEND_WINDOW: int             = 10
EXTEND_THRESHOLD: float        = 0.1
MAX_GENERATIONS: int    = 200  # or None for “no hard cap”

# mutation weights
MUTATION_WEIGHTS: Mapping[str, float] = {
    "split": 0.3,
    "rotate": 0.2,
    "swap":   0.5,
}


# log
SAVE_LOGS = False
SAVE_LOGS_PATH = "nesting/run_logs"
LOG_TIME = True

SAMPLES_PER_EDGE = 5