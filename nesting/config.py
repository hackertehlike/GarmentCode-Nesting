# nesting/config.py
"""
Organized configuration system for nesting algorithms.
"""

import math
import json
import hashlib
import os
from dataclasses import dataclass, asdict, field
from typing import Literal, Mapping, Dict, Any, Optional, List
from enum import Enum

# ——— Type Definitions ————————————————————————————————————————————————
DecoderName = Literal["BL", "Greedy", "NFP", "Random"]
MetricName = Literal["usage_bb", "concave_hull", "concave_hull_area", "rest_length", "rest_height", "cc_with_rest_height", "cc_with_rest_length", "bb_cc", "bb_cc_area", "bb_with_rest_length"]
CrossoverName = Literal["oxk", "cross_stitch_oxk"]
SortKey = Literal["bbox_area", "hull_area", "aspect_ratio"]
CrossStitchMode = Literal["sticky", "lexicographic"]

class Environment(Enum):
    DEV = "development"
    TEST = "testing" 
    PROD = "production"

# ——— Configuration Classes ————————————————————————————————————————————

@dataclass
class ActiveConfig:
    """Configuration that affects algorithm behavior and reproducibility.
    
    Only these settings influence the stable hash. Changes here require
    new experiment runs for proper comparison.
    """
    # Core algorithm settings
    selected_decoder: DecoderName = "NFP"
    selected_fitness_metric: MetricName = "bb_with_rest_length" 
    selected_crossover: CrossoverName = "cross_stitch_oxk"
    cross_stitch_mode: CrossStitchMode = "lexicographic"
    
    # Container & physical constraints  
    container_width_cm: float = 400
    container_height_cm: float = 51
    seam_allowance_cm: float = 0.001
    samples_per_edge: int = 7
    
    # Genetic algorithm parameters
    population_size: int = 100
    num_generations: int = 20
    mutation_rate: float = 0.2
    force_mutation_on_crossover: bool = False

    # Population composition weights
    population_weights: Dict[str, float] = field(default_factory=lambda: {
        "elites": 0.25, "offspring": 0, "mutants": 0.5, "randoms": 0.25
    })
    
    # Mutation operation weights
    mutation_weights: Dict[str, float] = field(default_factory=lambda: {
        "rotate": 1, "swap": 1, "inversion": 1, "insertion": 1,
        "scramble": 1, "split": 0, "design_params": 0
    })

    zero_gen_rots: bool = True
    
    # Decoder-specific settings
    preserve_holes: bool = True
    nfp_gravitate_on: bool = True
    nfp_edge_samples: int = 5
    gravitate_once: bool = False
    gravitate_step: int = 1
    sort_by: SortKey = "hull_area"
    
    # Fitness metric weights & penalties
    bb_weight: float = 0.5 
    cc_weight: float = 0.5
    
    # Split & design parameter settings
    ox_k: int = 1
    num_splits: int = 1
    split_lower_bound: float = 0.3
    split_upper_bound: float = 0.7
    symmetric_splits: bool = False
    allow_recursive_splits: bool = False
    weight_by_bbox: bool = False
    param_change_margin: float = 0.6
    swap_mutation_k: Optional[int] = None
    
    # Dynamic stopping criteria
    enable_dynamic_stopping: bool = True
    early_stop_window: int = 10
    early_stop_tolerance: float = 0.01
    enable_extension: bool = False
    extend_window: int = 10
    extend_threshold: float = 0.1
    max_generations: int = 20
    
    # Concave hull parameters
    hull_trim_ratio: int = 10
    interior_sample_spacing: int = 5
    boundary_sample_spacing: int = 3
    snap: bool = False
    snap_tolerance: float = 0.1
    
    # Rotation settings
    enable_rotations: bool = True
    allowed_rotations: List[int] = field(default_factory=lambda: [0,180])
    
    # Parameter exclusions for mutations
    excluded_param_paths: List[str] = field(default_factory=lambda: [
        "*meta*", "*.base", "*.level", "*_collar", "*.component.style", 
        "*.cuff.type", "*.panel_curve", "*.num_inserts", "*style_side_cut*",
        "*strapless*", "*sleeveless*", "*enable_asym*", "*flip*", 
        "*standing_shoulder*", "*lapel_standing*", "*sleeve.armhole_shape*",
        "*cut.add*", "*.n_panels", "*panel_curve*"
    ])
    
    def get_stable_hash(self) -> str:
        """Generate reproducible hash based only on algorithmic settings."""
        config_dict = asdict(self)
        # Sort nested dicts for consistency
        for key, value in config_dict.items():
            if isinstance(value, dict):
                config_dict[key] = dict(sorted(value.items()))
        
        config_str = json.dumps(config_dict, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

@dataclass
class SystemConfig:
    """System-level settings that don't affect algorithm results."""
    # System & performance
    multithreading: bool = True
    verbose: bool = True
    
    # File paths
    default_pattern_path: str = "nesting-assets/pattern_files/rand_04ANOD2PBA/rand_04ANOD2PBA_specification.json"
    default_design_param_path: str = "nesting-assets/pattern_files/rand_04ANOD2PBA/rand_04ANOD2PBA_design_params.yaml"  
    default_body_param_path: str = "nesting-assets/pattern_files/rand_04ANOD2PBA/rand_04ANOD2PBA_body_measurements.yaml"
    
    # Logging & experiments
    save_logs: bool = True
    experiments_root: str = "nesting/experiments"
    log_time: bool = True
    log_design_param_paths: bool = False
    save_generation_svgs: bool = True
    
    # Fitness metric parameters (not part of active config hash)
    rest_penalty: float = 0.001
    
    # GUI
    num_copies: int = 0

# ——— Global Configuration Instances ——————————————————————————————————————

ACTIVE = ActiveConfig()
SYSTEM = SystemConfig()

# ——— Configuration Management Functions ——————————————————————————————————

def load_config(env: Environment = Environment.DEV, **overrides) -> ActiveConfig:
    """Load configuration based on environment with optional overrides."""
    global ACTIVE, SYSTEM
    
    if env == Environment.TEST:
        ACTIVE.population_size = 10
        ACTIVE.num_generations = 5
        SYSTEM.save_logs = False
        
    elif env == Environment.PROD:
        ACTIVE.population_size = 200
        ACTIVE.num_generations = 500
        ACTIVE.enable_dynamic_stopping = True
        
    # Apply environment variables
    if metric := os.getenv('NESTING_FITNESS_METRIC'):
        ACTIVE.selected_fitness_metric = metric
    if decoder := os.getenv('NESTING_DECODER'):
        ACTIVE.selected_decoder = decoder
    if pop_size := os.getenv('NESTING_POPULATION_SIZE'):
        ACTIVE.population_size = int(pop_size)
    if generations := os.getenv('NESTING_GENERATIONS'):
        ACTIVE.num_generations = int(generations)
    
    # Apply direct overrides
    for key, value in overrides.items():
        if hasattr(ACTIVE, key):
            setattr(ACTIVE, key, value)
        elif hasattr(SYSTEM, key):
            setattr(SYSTEM, key, value)
        else:
            raise ValueError(f"Unknown configuration setting: {key}")
    
    # Update all backward compatibility variables
    _update_backward_compatibility_vars()
    
    return ACTIVE

def get_stable_hash() -> str:
    """Get stable hash of current active configuration."""
    return ACTIVE.get_stable_hash()

def get_stable_json() -> str:
    """Get stable JSON of current active configuration."""
    config_dict = asdict(ACTIVE)
    return json.dumps(config_dict, sort_keys=True, separators=(',', ':'))

# ——— Configuration Profiles ——————————————————————————————————————————————

def create_profile(**settings) -> ActiveConfig:
    """Create a configuration profile with custom settings."""
    profile = ActiveConfig()
    for key, value in settings.items():
        if hasattr(profile, key):
            setattr(profile, key, value)
        else:
            raise ValueError(f"Unknown configuration setting: {key}")
    return profile

# Pre-defined profiles
PROFILES = {
    "test": create_profile(
        population_size=20,
        num_generations=2
    ),
    
    "rest_length": create_profile(
        population_size=100,
        num_generations=20,
        selected_fitness_metric="rest_length"
    ),
    
    "usage_bb": create_profile(
        population_size=100,
        num_generations=20,
        selected_fitness_metric="usage_bb"
    ),
    
    "bb_with_rest_length": create_profile(
        population_size=100,
        num_generations=20,
        selected_fitness_metric="bb_with_rest_length"
    )
}

def load_profile(name: str) -> ActiveConfig:
    """Load a pre-defined configuration profile."""
    if name not in PROFILES:
        available = ", ".join(PROFILES.keys())
        raise ValueError(f"Unknown profile '{name}'. Available: {available}")
    
    global ACTIVE
    ACTIVE = PROFILES[name]
    _update_backward_compatibility_vars()
    return ACTIVE

def _update_backward_compatibility_vars():
    """Update all backward compatibility variables after config changes."""
    global SELECTED_DECODER, SELECTED_FITNESS_METRIC, SELECTED_CROSSOVER, CROSS_STITCH_MODE
    global OX_K, NUM_SPLITS, SPLIT_LOWER_BOUND, SPLIT_UPPER_BOUND, PRESERVE_HOLES
    global CONTAINER_WIDTH_CM, CONTAINER_HEIGHT_CM, SEAM_ALLOWANCE_CM, SAMPLES_PER_EDGE
    global POPULATION_SIZE, NUM_GENERATIONS, MUTATION_RATE, FORCE_MUTATION_ON_CROSSOVER
    global SWAP_MUTATION_K, POPULATION_WEIGHTS, MUTATION_WEIGHTS
    global GRAVITATE_ONCE, GRAVITATE_STEP, SORT_BY, REST_PENALTY, BB_WEIGHT, CC_WEIGHT
    global NFP_GRAVITATE_ON, NFP_EDGE_SAMPLES, EXCLUDED_PARAM_PATHS, PARAM_CHANGE_MARGIN
    global SYMMETRIC_SPLITS, ALLOW_RECURSIVE_SPLITS, WEIGHT_BY_BBOX
    global ENABLE_DYNAMIC_STOPPING, EARLY_STOP_WINDOW, EARLY_STOP_TOLERANCE
    global ENABLE_EXTENSION, EXTEND_WINDOW, EXTEND_THRESHOLD, MAX_GENERATIONS
    global HULL_TRIM_RATIO, INTERIOR_SAMPLE_SPACING, BOUNDARY_SAMPLE_SPACING
    global SNAP, SNAP_TOLERANCE, ENABLE_ROTATIONS, ALLOWED_ROTATIONS
    global GENERATION_PER_FLUSH, LOG_FLUSH_INTERVAL
    #global MAX_DUPLICATE_RETRIES
    
    # Algorithm settings
    SELECTED_DECODER = ACTIVE.selected_decoder
    SELECTED_FITNESS_METRIC = ACTIVE.selected_fitness_metric  
    SELECTED_CROSSOVER = ACTIVE.selected_crossover
    CROSS_STITCH_MODE = ACTIVE.cross_stitch_mode
    OX_K = ACTIVE.ox_k
    NUM_SPLITS = ACTIVE.num_splits
    SPLIT_LOWER_BOUND = ACTIVE.split_lower_bound
    SPLIT_UPPER_BOUND = ACTIVE.split_upper_bound
    PRESERVE_HOLES = ACTIVE.preserve_holes
    
    # Container & Physical
    CONTAINER_WIDTH_CM = ACTIVE.container_width_cm
    CONTAINER_HEIGHT_CM = ACTIVE.container_height_cm
    SEAM_ALLOWANCE_CM = ACTIVE.seam_allowance_cm
    SAMPLES_PER_EDGE = ACTIVE.samples_per_edge
    
    # Genetic Algorithm
    POPULATION_SIZE = ACTIVE.population_size
    NUM_GENERATIONS = ACTIVE.num_generations
    MUTATION_RATE = ACTIVE.mutation_rate
    FORCE_MUTATION_ON_CROSSOVER = ACTIVE.force_mutation_on_crossover
    SWAP_MUTATION_K = ACTIVE.swap_mutation_k
    POPULATION_WEIGHTS = ACTIVE.population_weights
    MUTATION_WEIGHTS = ACTIVE.mutation_weights
    ZERO_GEN_ROTS = ACTIVE.zero_gen_rots
    # NO_DUPLICATES = ACTIVE.no_duplicates
    # MAX_DUPLICATE_RETRIES = ACTIVE.max_duplicate_retries
    
    # Decoder settings
    GRAVITATE_ONCE = ACTIVE.gravitate_once
    GRAVITATE_STEP = ACTIVE.gravitate_step
    SORT_BY = ACTIVE.sort_by
    REST_PENALTY = SYSTEM.rest_penalty
    BB_WEIGHT = ACTIVE.bb_weight
    CC_WEIGHT = ACTIVE.cc_weight
    NFP_GRAVITATE_ON = ACTIVE.nfp_gravitate_on
    NFP_EDGE_SAMPLES = ACTIVE.nfp_edge_samples
    
    # Design parameters
    EXCLUDED_PARAM_PATHS = ACTIVE.excluded_param_paths
    PARAM_CHANGE_MARGIN = ACTIVE.param_change_margin
    SYMMETRIC_SPLITS = ACTIVE.symmetric_splits
    ALLOW_RECURSIVE_SPLITS = ACTIVE.allow_recursive_splits
    WEIGHT_BY_BBOX = ACTIVE.weight_by_bbox
    
    # Dynamic stopping
    ENABLE_DYNAMIC_STOPPING = ACTIVE.enable_dynamic_stopping
    EARLY_STOP_WINDOW = ACTIVE.early_stop_window
    EARLY_STOP_TOLERANCE = ACTIVE.early_stop_tolerance
    ENABLE_EXTENSION = ACTIVE.enable_extension
    EXTEND_WINDOW = ACTIVE.extend_window
    EXTEND_THRESHOLD = ACTIVE.extend_threshold
    MAX_GENERATIONS = ACTIVE.max_generations
    
    # Concave hull & sampling
    HULL_TRIM_RATIO = ACTIVE.hull_trim_ratio
    INTERIOR_SAMPLE_SPACING = ACTIVE.interior_sample_spacing
    BOUNDARY_SAMPLE_SPACING = ACTIVE.boundary_sample_spacing
    SNAP = ACTIVE.snap
    SNAP_TOLERANCE = ACTIVE.snap_tolerance
    ENABLE_ROTATIONS = ACTIVE.enable_rotations
    ALLOWED_ROTATIONS = ACTIVE.allowed_rotations
    
    # Calculated values
    GENERATION_PER_FLUSH = max(1, min(math.ceil(100 / ACTIVE.population_size), 10))
    LOG_FLUSH_INTERVAL = GENERATION_PER_FLUSH

# ——— Backward Compatibility Layer ————————————————————————————————————————
# These maintain compatibility with existing code that imports variables directly

# Algorithm settings
SELECTED_DECODER = ACTIVE.selected_decoder
PRESERVE_HOLES = ACTIVE.preserve_holes
SELECTED_FITNESS_METRIC = ACTIVE.selected_fitness_metric
SELECTED_CROSSOVER = ACTIVE.selected_crossover
CROSS_STITCH_MODE = ACTIVE.cross_stitch_mode
OX_K = ACTIVE.ox_k
NUM_SPLITS = ACTIVE.num_splits
SPLIT_LOWER_BOUND = ACTIVE.split_lower_bound
SPLIT_UPPER_BOUND = ACTIVE.split_upper_bound
# NO_DUPLICATES = ACTIVE.no_duplicates
# MAX_DUPLICATE_RETRIES = ACTIVE.max_duplicate_retries

# Container & Physical
CONTAINER_WIDTH_CM = ACTIVE.container_width_cm
CONTAINER_HEIGHT_CM = ACTIVE.container_height_cm
SEAM_ALLOWANCE_CM = ACTIVE.seam_allowance_cm
SAMPLES_PER_EDGE = ACTIVE.samples_per_edge

# System settings
MULTITHREADING = SYSTEM.multithreading
VERBOSE = SYSTEM.verbose
DEFAULT_PATTERN_PATH = SYSTEM.default_pattern_path
DEFAULT_DESIGN_PARAM_PATH = SYSTEM.default_design_param_path
DEFAULT_BODY_PARAM_PATH = SYSTEM.default_body_param_path
NUM_COPIES = SYSTEM.num_copies


# More backward compatibility assignments
GRAVITATE_ONCE = ACTIVE.gravitate_once
GRAVITATE_STEP = ACTIVE.gravitate_step
SORT_BY = ACTIVE.sort_by
REST_PENALTY = SYSTEM.rest_penalty
BB_WEIGHT = ACTIVE.bb_weight
CC_WEIGHT = ACTIVE.cc_weight
NFP_GRAVITATE_ON = ACTIVE.nfp_gravitate_on
NFP_EDGE_SAMPLES = ACTIVE.nfp_edge_samples

POPULATION_SIZE = ACTIVE.population_size
NUM_GENERATIONS = ACTIVE.num_generations
MUTATION_RATE = ACTIVE.mutation_rate
FORCE_MUTATION_ON_CROSSOVER = ACTIVE.force_mutation_on_crossover
SWAP_MUTATION_K = ACTIVE.swap_mutation_k
POPULATION_WEIGHTS = ACTIVE.population_weights


# More backward compatibility
EXCLUDED_PARAM_PATHS = ACTIVE.excluded_param_paths
PARAM_CHANGE_MARGIN = ACTIVE.param_change_margin
SYMMETRIC_SPLITS = ACTIVE.symmetric_splits
ALLOW_RECURSIVE_SPLITS = ACTIVE.allow_recursive_splits
WEIGHT_BY_BBOX = ACTIVE.weight_by_bbox
MUTATION_WEIGHTS = ACTIVE.mutation_weights
ZERO_GEN_ROTS = ACTIVE.zero_gen_rots

ENABLE_DYNAMIC_STOPPING = ACTIVE.enable_dynamic_stopping
EARLY_STOP_WINDOW = ACTIVE.early_stop_window
EARLY_STOP_TOLERANCE = ACTIVE.early_stop_tolerance
ENABLE_EXTENSION = ACTIVE.enable_extension
EXTEND_WINDOW = ACTIVE.extend_window
EXTEND_THRESHOLD = ACTIVE.extend_threshold
MAX_GENERATIONS = ACTIVE.max_generations

# Calculated values
GENERATION_PER_FLUSH = max(1, min(math.ceil(100 / ACTIVE.population_size), 10))
LOG_FLUSH_INTERVAL = GENERATION_PER_FLUSH

# System settings
SAVE_LOGS = SYSTEM.save_logs
EXPERIMENTS_ROOT = SYSTEM.experiments_root
RUNS_DIR = f"{SYSTEM.experiments_root}/runs"
AGGREGATE_DIR = f"{SYSTEM.experiments_root}/aggregate"
SAVE_LOGS_PATH = f"{RUNS_DIR}/"
LOG_TIME = SYSTEM.log_time
LOG_DESIGN_PARAM_PATHS = SYSTEM.log_design_param_paths
SAVE_GENERATION_SVGS = SYSTEM.save_generation_svgs

# Concave hull & sampling
HULL_TRIM_RATIO = ACTIVE.hull_trim_ratio
INTERIOR_SAMPLE_SPACING = ACTIVE.interior_sample_spacing
BOUNDARY_SAMPLE_SPACING = ACTIVE.boundary_sample_spacing
SNAP = ACTIVE.snap
SNAP_TOLERANCE = ACTIVE.snap_tolerance
ENABLE_ROTATIONS = ACTIVE.enable_rotations
ALLOWED_ROTATIONS = ACTIVE.allowed_rotations


# ——— Legacy Functions (Updated to use new system) ————————————————————————

def as_dict() -> dict:
    """Return the active configuration values as a dictionary.
    
    Updated to use the new stable configuration system.
    """
    return asdict(ACTIVE)

def __dict__() -> dict:
    """Backward compatibility alias."""
    return as_dict()

def stable_config_json() -> str:
    """Return a deterministic JSON string of the active config.
    
    Updated to use new stable configuration system instead of exported keys.
    """
    return get_stable_json()

def stable_config_hash() -> str:
    """Return a stable hash digest for the active config values.
    
    Updated to use the new stable hashing system.
    """
    return get_stable_hash()
