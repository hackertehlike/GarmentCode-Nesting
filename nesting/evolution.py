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
        pmx: bool = True
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

        # Lists to store per-generation metrics for optional plotting
        self.survival_rates = []
        self.avg_child_fitnesses = []

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
                # generate random rotation
                #rotations.sample()

                samples.append(chromosome)
                num_generated += 1
        
        # sort the samples by fitness
        samples.sort(key=lambda x: x.fitness, reverse=True)
        
        # print the populations (by id) and rotation and fitness
        for i, chromosome in enumerate(samples):
            print(f"Layout {i}: {[piece.id for piece in chromosome.genes]} with fitness {chromosome.fitness}")

        

            
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
        print(f"Initial population of {self.population_size} layouts generated.")
        # print the populations (by id) and rotation
        for i, chromosome in enumerate(self.population):
            print(f"Layout {i}: {[piece.id for piece in chromosome.genes]} with fitness {chromosome.fitness}")
        print("Population generation completed.")
        # print divider
        print("-" * 50)
        print(" " * 50)
        print("-" * 50)
        print(" " * 50)


    def get_elite(self) -> list[Chromosome]:
        """
        Evaluates the fitness of each layout in the population.
        The fitness function is a placeholder and should be replaced with a real one.
        """
        print("Evaluating fitness of the population...")
        for chromosome in self.population:
            chromosome.calculate_fitness()
        # sort the population by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # return the top N layouts
        print(f"Returning elite layouts best {self.population[0].fitness}...")
        return self.population[:self.elite_population_size]
    
    def next_generation(self) -> None:
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
        self.generation += 1

        new_elite = self.get_elite()
        new_elite_ids = {id(chromo) for chromo in new_elite}
        # 1) Survival Rate
        intersection_count = len(old_elite_ids & new_elite_ids)
        survival_rate = intersection_count / len(old_elite_ids)
        print(f"Elite survival rate: {survival_rate:.2%}")

        # 2) Average child fitness
        children = new_population[len(old_elite):]  # exclude the elite parents
        if children:
            avg_child_fitness = sum(ch.fitness for ch in children) / len(children)
        else:
            avg_child_fitness = 0
        print(f"Average child fitness: {avg_child_fitness:.2f}")

        # Store metrics for plotting later
        self.survival_rates.append(survival_rate)
        self.avg_child_fitnesses.append(avg_child_fitness)

        print(f"Generation {self.generation} completed.")

    def run(self, plot_results: bool = False) -> Chromosome:
        """
        Runs the genetic algorithm for a specified number of generations.
        If plot_results=True, produces a simple plot of survival rate
        and average child fitness vs. generation at the end.
        """
        self.generate_population()
        print("Starting evolution...")

        for _ in range(self.num_generations):
            self.next_generation()

        print("Evolution completed.")
        best_layout = self.get_elite()[0]
        print(f"Best layout fitness: {best_layout.fitness}")

        # Optional plotting at the end
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
