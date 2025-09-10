import math
import time
import random
from typing import Optional, Union, List

from .layout import Piece, Container, Layout
from .operations import METRIC_REGISTRY, Operators, weighted_choice
import nesting.config as config

class SimulatedAnnealing:
    def __init__(
            self, 
            pieces: List[Piece], 
            cooling_rate, 
            initial_temperature,
            design_params: Optional[dict] = None,
            body_params: Optional[object] = None,
            initial_design_params: Optional[dict] = None,
        ):
        self.current_state = pieces
        self.cooling_rate = cooling_rate
        self.temperature = initial_temperature
        self.design_params = design_params
        self.body_params = body_params
        self.initial_design_params = initial_design_params

        self.last_operation = None
        self.best_state = pieces
        self.best_fitness = float('-inf')

        self.meta_garment = None
        if self.design_params and self.body_params:
            try:
                from assets.garment_programs.meta_garment import MetaGarment
                self.meta_garment = MetaGarment("metagarment", self.body_params, self.design_params)
            except Exception as exc:
                print(f"[Chromosome] Failed to create MetaGarment: {exc}")

    # evaluate fitness of a state
    def fitness(self, state: List[Piece]) -> float:
        metric_fn = METRIC_REGISTRY[config.SELECTED_FITNESS_METRIC]
        return metric_fn(state, config.SELECTED_DECODER)
        
    # generate a neighboring state
    # analog of mutation
    def neighbor(self):

        start = time.time() if config.LOG_TIME else None

        chosen_operation = weighted_choice(config.MUTATION_WEIGHTS)
        self.last_operation = chosen_operation

        handler = {
            "split": self._split,
            "rotate": self._rotate,
            "swap": self._swap,
            "inversion": self._inversion,
            "insertion": self._insertion,
            "scramble": self._scramble,
            "design_params": self._design_params,
        }.get(chosen_operation)

        if handler is None:
            raise ValueError(f"Unknown operation: {chosen_operation}")

        generated_neighbor = handler()
        new_fitness = self.fitness(generated_neighbor)
        curr_fitness = self.fitness(self.current_state)
        if self.accept(curr_fitness, new_fitness):
            self.current_state = generated_neighbor
            if new_fitness > self.best_fitness:
                self.best_fitness = new_fitness
                self.best_state = self.current_state

        if config.LOG_TIME:
            end = time.time()
            elapsed = end - start
            print(f"Neighbor generation ({chosen_operation}) took {elapsed:.6f} seconds")

    def accept(self, old_fitness, new_fitness):
        ap = self._acceptance_probability(old_fitness, new_fitness)
        return ap > random.random()

    # acceptance probability
    def _acceptance_probability(self, old_fitness, new_fitness):
        if new_fitness > old_fitness:
            return 1.0
        return math.exp((new_fitness - old_fitness) / self.temperature)
    
    def termination_condition(self):
        return self.temperature < 1e-3
    
    def cool_down(self):
        self.temperature *= self.cooling_rate

    # operations to generate neighbors

    def _split(self) -> List[Piece]:
        pass

    def _rotate(self) -> List[Piece]:
        return Operators.rotate(self.current_state)

    def _swap(self, k=None) -> List[Piece]:
        k = k or getattr(config, 'SWAP_MUTATION_K', None)
        return Operators.swap(self.current_state, k)

    def _inversion(self) -> List[Piece]:
        return Operators.inversion(self.current_state)

    def _insertion(self) -> List[Piece]:
        return Operators.insertion(self.current_state)

    def _scramble(self) -> List[Piece]:
        return Operators.scramble(self.current_state)

    def _design_params(self) -> List[Piece]:
        if not (self.design_params and self.body_params):
            return self.current_state

        # Create fitness function for the shared operation
        def fitness_fn(pieces):
            metric_fn = METRIC_REGISTRY[config.SELECTED_FITNESS_METRIC]
            return metric_fn(pieces, config.SELECTED_DECODER)

        new_pieces, new_design_params, success = Operators.design_params(
            self.current_state,
            self.design_params,
            self.body_params,
            self.initial_design_params,
            getattr(self, 'split_history', []),
            fitness_fn
        )

        if success:
            # Update the design params (pieces will be returned and set by caller)
            self.design_params = new_design_params
            return new_pieces
        
        return self.current_state

    def run(self):
        while not self.termination_condition():
            self.neighbor()
            self.cool_down()