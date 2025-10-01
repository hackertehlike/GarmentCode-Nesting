"""random_search.py

Random Search Heuristic
=======================

Generates a specified number of random candidate layouts by optionally shuffling
the order of pieces and randomizing rotations, evaluates each using the existing
fitness/decoder pipeline, and returns the best.

Design Goals:
-------------
* Minimal dependency surface – reuses METRIC_REGISTRY + config settings.
* Purely stochastic; no neighborhood logic.
* Optional rotation perturbations (disable for order‑only randomization).
* Lightweight result object for downstream reporting or GUI integration.

Usage Example:
--------------
    from nesting.random_search import RandomSearch
    rs = RandomSearch(pieces, container, num_samples=500,
                      randomize_rotations=True, shuffle_order=True)
    best_pieces, best_fitness = rs.run()
    print(best_fitness)

Notes:
------
* Each sample deep‑copies pieces before applying rotations so the original list
  remains unchanged.
* Fitness evaluation uses the same decoder/metric as GA & SA for consistency.
* If ``include_initial`` is True the unmodified input order/rotations are
  evaluated first and counted toward the sample budget.
"""

from __future__ import annotations
import random
import copy
from dataclasses import dataclass, field
from typing import List, Optional

from .layout import Piece, Container
from .operations import METRIC_REGISTRY
import nesting.config as config


@dataclass
class RandomSearchResult:
    best_pieces: List[Piece]
    best_fitness: float
    samples_evaluated: int
    history: list[tuple[int, float]] = field(default_factory=list)  # (sample_idx, fitness)


class RandomSearch:
    """Random sampling of permutations (and optional rotations) of pieces.

    Args:
        pieces: Original list of pieces (not mutated)
        container: Container instance for decoding / metrics
        num_samples: Number of random candidate layouts to evaluate (>=1)
        shuffle_order: If True (default) randomly permute order each sample
        randomize_rotations: If True apply random allowed rotation per piece
        allowed_rotations: Optional explicit rotation set; defaults to config.ALLOWED_ROTATIONS
        include_initial: If True evaluate original ordering/rotations before sampling
        track_history: If True retain (sample_idx, fitness) for each evaluation
        verbose: If True print progress
        seed: Optional RNG seed for reproducibility
    """

    def __init__(
        self,
        pieces: List[Piece],
        container: Container,
        num_samples: int = 100,
        *,
        shuffle_order: bool = True,
        randomize_rotations: bool = True,
        allowed_rotations: Optional[List[int]] = None,
        include_initial: bool = True,
        track_history: bool = False,
        verbose: bool = True,
        seed: Optional[int] = None,
    ):
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1")
        self.original_pieces = pieces  # external list; do NOT mutate
        self.container = container
        self.num_samples = num_samples
        self.shuffle_order = shuffle_order
        self.randomize_rotations = randomize_rotations
        self.allowed_rotations = allowed_rotations or getattr(config, 'ALLOWED_ROTATIONS', [0])
        self.include_initial = include_initial
        self.track_history = track_history
        self.verbose = verbose
        if seed is not None:
            random.seed(seed)

        self.metric_fn = METRIC_REGISTRY[config.SELECTED_FITNESS_METRIC]
        self.decoder_name = config.SELECTED_DECODER

        self.best_fitness = float('-inf')
        self.best_state: List[Piece] = []
        self.samples_evaluated = 0
        self.history: list[tuple[int, float]] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _log(self, msg: str):
        if self.verbose:
            print(f"[RandomSearch] {msg}")

    def _sample_candidate(self, base: List[Piece], apply_shuffle: bool) -> List[Piece]:
        candidate = [copy.deepcopy(p) for p in base]
        if apply_shuffle:
            random.shuffle(candidate)
        if self.randomize_rotations and len(self.allowed_rotations) > 1:
            for p in candidate:
                # Keep current rotation possibility included; ensure change opportunistically
                p.rotate(random.choice(self.allowed_rotations))
        return candidate

    def _evaluate(self, pieces: List[Piece]) -> float:
        return self.metric_fn(pieces, self.decoder_name, self.container)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> tuple[List[Piece], float]:
        remaining = self.num_samples

        # Optionally evaluate the original arrangement first
        if self.include_initial:
            initial_copy = [copy.deepcopy(p) for p in self.original_pieces]
            fit = self._evaluate(initial_copy)
            self.samples_evaluated += 1
            if self.track_history:
                self.history.append((self.samples_evaluated, fit))
            self.best_fitness = fit
            self.best_state = initial_copy
            remaining -= 1
            self._log(f"Initial fitness: {fit:.6f}")

        # Sample loop
        for _ in range(max(0, remaining)):
            candidate = self._sample_candidate(self.original_pieces, self.shuffle_order)
            fit = self._evaluate(candidate)
            self.samples_evaluated += 1
            if self.track_history:
                self.history.append((self.samples_evaluated, fit))
            if fit > self.best_fitness:
                self.best_fitness = fit
                self.best_state = candidate
                self._log(f"New best fitness {fit:.6f} at sample {self.samples_evaluated}")

        if not self.best_state:  # fallback (should not happen unless no samples)
            self.best_state = [copy.deepcopy(p) for p in self.original_pieces]
            self.best_fitness = self._evaluate(self.best_state)

        return [copy.deepcopy(p) for p in self.best_state], self.best_fitness

    def result(self) -> RandomSearchResult:
        return RandomSearchResult(
            best_pieces=[copy.deepcopy(p) for p in self.best_state],
            best_fitness=self.best_fitness,
            samples_evaluated=self.samples_evaluated,
            history=list(self.history),
        )

__all__ = ["RandomSearch", "RandomSearchResult"]
