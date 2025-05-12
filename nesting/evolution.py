"""
Genetic algorithm evolving a population of nesting layouts.
This module implements a genetic algorithm to evolve a population of nesting layouts.

Randomly generates a starting population of layouts.

Each generation, the algorithm evaluates the fitness of each layout, selects the elite layouts,
performs crossover and mutation to create new layouts, and replaces the old population with the new one.
"""

from __future__ import annotations

from nesting.path_extractor import PatternPathExtractor
from .layout import Piece, Container
from .chromosome import Chromosome
from collections import OrderedDict
import random
import time
import matplotlib.pyplot as plt
import os


# TODO: make into a config file or parameters to the class
# # META VARIABLES
# NUM_GENERATIONS = 10
# POPULATION_SIZE = 10
# ELITE_POPULATION_SIZE = 5
# MUTATION_RATE = 0.2
# PMX = True
# SINGLE_CELL_STYLE = False

class Evolution:
    """
    A class representing the evolution of a population of layouts.
    """

    def __init__(
        self,
        pieces: dict[str, Piece],
        container: Container,
        num_generations: int = 10,
        population_size: int = 10,
        elite_population_size: int = 5,
        mutation_rate: float = 0.2,
        pmx: bool = True,
        allow_duplicate_genes: bool = False,
    ):
        self.generation = 0
        self.container = container
        self.pieces = pieces
        self.population = []
        
        # Meta variables as instance attributes
        self.num_generations = num_generations
        self.population_size = population_size
        self.elite_population_size = elite_population_size
        self.mutation_rate = mutation_rate
        self.pmx = pmx
        self.allow_duplicate_genes = allow_duplicate_genes
        self._log("Evolution instance created.", divider=True)

        # Lists to store per-generation metrics for optional plotting
        self.survival_rates = []
        self.avg_child_fitnesses = []
        self.log_lines = []

    def _log(self, msg: str = "", divider: bool = False):
        """Helper for consistent logging and log collection."""
        if divider:
            line = "-" * 60
            print(line)
            self.log_lines.append(line)
        if msg:
            print(msg)
            self.log_lines.append(msg)

    def random_sample(self, k):
        """
        Randomly samples k elements from the population and returns their fitness values.
        """
        piece_ids = list(self.pieces.keys())
        samples = []
        rotations = {0, 90, 180, 270}

        num_generated = 0
        while num_generated < k:
            chromosome = self.generate_random_chromosome()
            if chromosome not in samples:
                # TODO: generate random rotation

                samples.append(chromosome)
                num_generated += 1
        
        # sort the samples by fitness
        samples.sort(key=lambda x: x.fitness, reverse=True)
        
        # print the populations (by id) and rotation and fitness
        print()
        for i, chromosome in enumerate(samples):
            print(f"Layout {i}: {[piece.id for piece in chromosome.genes]} with fitness {chromosome.fitness}")
        print("-" * 50)
        

            
    def generate_random_chromosome(self) -> Chromosome:
        """
        Generates a random chromosome (layout) by shuffling the pieces.
        """
        piece_ids = list(self.pieces.keys())
        random.shuffle(piece_ids)
        # hand the *pieces* to Chromosome
        shuffled_pieces = [self.pieces[pid] for pid in piece_ids]
        chromosome = Chromosome(shuffled_pieces, self.container)
        chromosome.calculate_fitness()
        return chromosome

    def generate_population(self) -> None:
        while len(self.population) < self.population_size:
            chromosome = self.generate_random_chromosome()
            self.population.append(chromosome)
        self._log(f"Initial population of {self.population_size} layouts generated.", divider=True)
        for i, chromosome in enumerate(self.population):
            self._log(f"Layout {i}: {[piece.id for piece in chromosome.genes]} | Fitness: {chromosome.fitness:.4f}")
        self._log("Population generation completed.", divider=True)

    def get_elite(self) -> list[Chromosome]:
        # self._log("Evaluating fitness of the population...", divider=False)
        # for chromosome in self.population:
        #     chromosome.calculate_fitness()
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        self._log(f"Returning elite layouts. Best fitness: {self.population[0].fitness:.4f}", divider=True)
        return self.population[:self.elite_population_size]
    
    def next_generation(self) -> None:
        self._log("Next generation...", divider=True)
        old_elite = self.get_elite()  
        old_elite_ids = {id(chromo) for chromo in old_elite}
        new_population = old_elite[:]

        # Produce children
        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(self.population, 2)
            if self.pmx:
                child = parent1.crossover_pmx(parent2)
            else:
                child = parent1.crossover_ox1_k(parent2)
            if random.random() < self.mutation_rate:
                child.mutate()
            child.calculate_fitness()
            new_population.append(child)

        self.population = new_population

        #print the new population
        for i, chromosome in enumerate(self.population):
            self._log(f"Layout {i}: {[piece.id for piece in chromosome.genes]} | Fitness: {chromosome.fitness:.4f}")
        self._log(divider=True)

        self.generation += 1

        new_elite = self.get_elite()
        new_elite_ids = {id(chromo) for chromo in new_elite}
        # 1) Survival Rate
        intersection_count = len(old_elite_ids & new_elite_ids)
        survival_rate = intersection_count / len(old_elite_ids)
        self._log(f"Elite survival rate: {survival_rate:.2%}")

        # 2) Average child fitness
        children = new_population[len(old_elite):]  # exclude the elite parents
        if children:
            avg_child_fitness = sum(ch.fitness for ch in children) / len(children)
        else:
            avg_child_fitness = 0
        self._log(f"Average child fitness: {avg_child_fitness:.4f}")

        # Store metrics for plotting later
        self.survival_rates.append(survival_rate)
        self.avg_child_fitnesses.append(avg_child_fitness)

        self._log(f"Generation {self.generation} completed.", divider=True)


    def run(self, results_dir=None, plot_results: bool = False) -> Chromosome:
        """
        Runs the genetic algorithm for a specified number of generations.
        If plot_results=True, produces a simple plot of survival rate
        and average child fitness vs. generation at the end.
        """
        self.log_lines = []  # Reset log for this run
        self._log("Starting evolution...", divider=True)
        self.generate_population()

        for _ in range(self.num_generations):
            self.next_generation()

        self._log("Evolution completed.", divider=True)
        best_layout = self.get_elite()[0]
        self._log(f"Best layout fitness: {best_layout.fitness:.4f}", divider=True)

        # Save log to file if results_dir is provided
        if results_dir:
            log_path = os.path.join(results_dir, "run_log.txt")
            with open(log_path, "w", encoding="utf-8") as f:
                for line in self.log_lines:
                    f.write(line + "\n")

        if plot_results:
            self._plot_metrics()

        return best_layout

    def _plot_metrics(self):
        """Plots survival rate and average child fitness per generation."""
        gens = range(1, self.generation + 1)

        plt.figure()
        plt.plot(gens, self.survival_rates, marker='o', label='Survival Rate')
        plt.plot(gens, self.avg_child_fitnesses, marker='x', label='Avg Child Fitness')
        plt.title("GA Per‑Generation Metrics")
        plt.xlabel("Generation")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()
    
if __name__ == "__main__":
    default_container = Container(140, 200)
    # default_path = "/Users/aysegulbarlas/codestuff/GarmentCode/nesting-assets/Configured_design_specification.json"
    default_path = "/Users/aysegulbarlas/codestuff/GarmentCode/nesting-assets/Configured_design_specification_asym_dress.json"
    extractor = PatternPathExtractor(default_path)
    all_pieces = extractor.get_all_panel_pieces(samples_per_edge=20)

    # Create an instance of the Evolution class
    evolution = Evolution(all_pieces, default_container)
    # Run the evolution process
    # best_layout = evolution.run()
    # Print the best layout
    # print("Best layout:")
    # for piece in best_layout.genes:
    #     print(piece.id)
    # print(f"Fitness: {best_layout.fitness}")

    #log time
    start_time = time.time()
    samples = evolution.random_sample(200)
    end_time = time.time()
    
    print(f"Time taken: {end_time - start_time} seconds")
