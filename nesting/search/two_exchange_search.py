"""two_exchange_search.py

2-Exchange (Local Swap) Search Heuristic
========================================

This module implements a simple deterministic (or randomized) hill-climbing style
local search using only 2-exchange (swap) moves constrained to a local neighborhood
window defined by a maximum index distance (``max_distance``).

Neighborhood Definition:
    A neighbor is any layout obtainable by swapping two pieces whose indices i, j
    satisfy: 0 <= i < j < n and 1 <= (j - i) <= max_distance.

Search Strategies Supported:
    - "first_better": Iterate neighbors in a randomized order and accept the first
        strictly improving swap. (First-improvement / stochastic steepest ascent)
    - "best": Evaluate the entire neighborhood, select the best improving swap
        (steepest-ascent). If no improvement exists, terminate.
    - "random_better": Evaluate the full neighborhood, collect all improving swaps,
        then pick one uniformly at random to apply. If none, terminate.

Termination:
    The search stops when no improving neighbor exists (i.e., local optimum under
    the 2-exchange neighborhood) or when ``max_iterations`` (safety cap) is reached.

Integration Notes:
    Fitness evaluation leverages the existing METRIC_REGISTRY and decoder selection
    from ``nesting.config`` to ensure consistency with Simulated Annealing and GA.

Usage Example:
    from nesting.search.two_exchange_search import TwoExchangeSearch
    tes = TwoExchangeSearch(initial_pieces, container, strategy="first_better", max_distance=3)
    best_pieces, best_fitness = tes.run()

Author: Auto-generated helper (requested by user)
"""

from __future__ import annotations
import random
import copy
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Literal, Any

from nesting.core.layout import Piece, Container
from .operations import METRIC_REGISTRY
import nesting.config as config

StrategyName = Literal["first_better", "best", "random_better"]


def _all_local_pairs(n: int, max_distance: int) -> List[Tuple[int, int]]:
    """Return all (i, j) pairs with 0 <= i < j < n and 1 <= j-i <= max_distance."""
    pairs: List[Tuple[int, int]] = []
    for i in range(n - 1):
        upper = min(n - 1, i + max_distance)
        for j in range(i + 1, upper + 1):
            pairs.append((i, j))
    return pairs


@dataclass
class TwoExchangeResult:
    best_pieces: List[Piece]
    best_fitness: float
    iterations: int
    improvements: int
    history: list[tuple[int, float, tuple[int, int]] | tuple[int, float, None]] = field(default_factory=list)


class TwoExchangeSearch:
    """2-Exchange local search restricted to swapping nearby pieces.

    Args:
        pieces: Initial list of pieces (will be deep-copied internally)
        container: Container object for decoding
        strategy: One of {"first_better", "best", "random_better"}
        max_distance: Maximum index gap for a valid swap (default 3)
        max_iterations: Safety upper bound on iterations (default 10_000)
        shuffle_neighbors: If True (default for first_better), randomize neighbor order
            each iteration for diversification.
        verbose: If True, print progress logs.
    """

    def __init__(
        self,
        pieces: List[Piece],
        container: Container,
        strategy: StrategyName = "first_better",
        max_distance: int = 3,
        max_iterations: int = 10_000,
        shuffle_neighbors: Optional[bool] = None,
        verbose: bool = True,
    ):
        if strategy not in ("first_better", "best", "random_better"):
            raise ValueError(f"Unknown strategy '{strategy}'")

        self.strategy = strategy
        self.max_distance = max_distance
        self.max_iterations = max_iterations
        if shuffle_neighbors is None:
            shuffle_neighbors = (strategy == "first_better")
        self.shuffle_neighbors = shuffle_neighbors
        self.verbose = verbose

        # Deep copy initial pieces to avoid mutating caller's list
        self.current_state = [copy.deepcopy(p) for p in pieces]
        self.container = container

        self.metric_fn = METRIC_REGISTRY[config.SELECTED_FITNESS_METRIC]
        self.decoder_name = config.SELECTED_DECODER

        self.current_fitness = self.metric_fn(self.current_state, self.decoder_name, self.container)
        self.best_state = [copy.deepcopy(p) for p in self.current_state]
        self.best_fitness = self.current_fitness

        self.iterations = 0
        self.improvements = 0
        self.history: list[tuple[int, float, tuple[int, int]] | tuple[int, float, None]] = []
        self.history.append((self.iterations, self.current_fitness, None))

    # ------------------------------------------------------------------
    # Core search logic
    # ------------------------------------------------------------------
    def _evaluate_swap(self, i: int, j: int) -> float:
        """Return fitness for state with pieces i and j swapped (non-destructive)."""
        state = self.current_state
        swapped = state[:]
        swapped[i], swapped[j] = swapped[j], swapped[i]
        return self.metric_fn(swapped, self.decoder_name, self.container)

    def _neighbors(self):
        n = len(self.current_state)
        if n < 2:
            return []
        return _all_local_pairs(n, self.max_distance)

    def _log(self, msg: str):
        if self.verbose:
            print(f"[TwoExchangeSearch] {msg}")

    def step(self) -> bool:
        """Perform one iteration. Returns True if improvement applied, else False."""
        pairs = self._neighbors()
        if not pairs:
            return False

        if self.shuffle_neighbors:
            random.shuffle(pairs)

        if self.strategy == "first_better":
            for (i, j) in pairs:
                fit = self._evaluate_swap(i, j)
                if fit > self.current_fitness:
                    self._apply_swap(i, j, fit)
                    return True
            return False

        # For strategies needing full neighborhood evaluation -----------------
        improvements: list[tuple[float, int, int]] = []  # (fitness, i, j)
        for (i, j) in pairs:
            fit = self._evaluate_swap(i, j)
            if fit > self.current_fitness:
                improvements.append((fit, i, j))

        if not improvements:
            return False

        if self.strategy == "best":
            # Pick highest fitness
            fit, i, j = max(improvements, key=lambda t: t[0])
            self._apply_swap(i, j, fit)
            return True

        if self.strategy == "random_better":
            fit, i, j = random.choice(improvements)
            self._apply_swap(i, j, fit)
            return True

        raise RuntimeError("Unreachable strategy path")

    def _apply_swap(self, i: int, j: int, new_fitness: float):
        state = self.current_state
        state[i], state[j] = state[j], state[i]
        self.current_fitness = new_fitness
        self.improvements += 1
        if new_fitness > self.best_fitness:
            self.best_fitness = new_fitness
            self.best_state = [copy.deepcopy(p) for p in state]
        self.history.append((self.iterations, new_fitness, (i, j)))
        self._log(f"Improved via swap ({i},{j}) -> fitness {new_fitness:.6f}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> tuple[List[Piece], float]:
        """Execute the 2-exchange search until local optimum or iteration cap.

        Returns:
            (best_pieces, best_fitness)
        """
        self._log(
            f"Starting search | strategy={self.strategy} | max_distance={self.max_distance} | initial_fitness={self.current_fitness:.6f}"
        )
        while self.iterations < self.max_iterations:
            self.iterations += 1
            improved = self.step()
            if not improved:
                self._log(
                    f"Terminating: local optimum reached after {self.iterations} iterations with best_fitness={self.best_fitness:.6f}"
                )
                break
        return [copy.deepcopy(p) for p in self.best_state], self.best_fitness

    def result(self) -> TwoExchangeResult:
        """Return a detailed result object after running."""
        return TwoExchangeResult(
            best_pieces=[copy.deepcopy(p) for p in self.best_state],
            best_fitness=self.best_fitness,
            iterations=self.iterations,
            improvements=self.improvements,
            history=list(self.history),
        )

__all__ = ["TwoExchangeSearch", "TwoExchangeResult"]
