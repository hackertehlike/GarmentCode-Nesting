# nesting/config.py

from typing import Literal

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




# log
SAVE_LOGS = False



