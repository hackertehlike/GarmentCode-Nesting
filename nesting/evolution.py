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

# META VARIABLES
NUM_GENERATIONS = 15
POPULATION_SIZE = 20
ELITE_POPULATION_SIZE = 5
MUTATION_RATE = 0.2
# SINGLE_CELL_STYLE = False

class Evolution:
    """
    A class representing the evolution of a population of layouts.
    """

    def __init__(self, pieces : dict[str, Piece], container : Container):
        self.generation = 0
        self.container = container
        self.pieces = pieces
        self.population = []
        self.generate_population()

    def generate_population(self) -> None:
        piece_ids = list(self.pieces.keys())

        for _ in range(POPULATION_SIZE):
            random.shuffle(piece_ids)
            # hand the *pieces* to Chromosome
            shuffled_pieces = [self.pieces[pid] for pid in piece_ids]
            chromosome = Chromosome(shuffled_pieces, self.container)
            chromosome.calculate_fitness()
            self.population.append(chromosome)

        print(f"Initial population of {POPULATION_SIZE} layouts generated.")
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
        return self.population[:ELITE_POPULATION_SIZE]
    
    def next_generation(self) -> None:
        """
        Generates the next generation of layouts.
        """
        print(f"Generating next generation from generation {self.generation}...")
        new_population = self.get_elite()

        # crossover and mutation
        while len(new_population) < POPULATION_SIZE:
            # print child number
            print(f"Creating child {len(new_population) + 1}...")
            parent1, parent2 = random.sample(self.population, 2)
            child = parent1.crossover_pmx(parent2)
            if random.random() < MUTATION_RATE:
                child.mutate()
            new_population.append(child)

        self.population = new_population
        self.generation += 1
        print(f"Generation {self.generation} completed.")
        print("New population:")
        for i, chromosome in enumerate(self.population):
            print(f"Layout {i}: {[(piece.id, piece.rotation) for piece in chromosome.genes]} with fitness {chromosome.fitness}")

        # print divider
        print("-" * 50)
        print(" " * 50)
        print("-" * 50)
        print(" " * 50)
        print(" " * 50)

    def run(self) -> None:
        """
        Runs the genetic algorithm for a specified number of generations.
        """
        for _ in range(NUM_GENERATIONS):
            self.next_generation()
        print("Evolution completed.")
        # return the best layout
        best_layout = self.get_elite()[0]
        print(f"Best layout fitness: {best_layout.fitness}")
        return best_layout
    
if __name__ == "__main__":
    default_container = Container(140, 200)
    default_path = "/Users/aysegulbarlas/codestuff/GarmentCode/nesting-assets/Configured_design_specification.json"
    extractor = PatternPathExtractor(default_path)
    all_pieces = extractor.get_all_panel_pieces(samples_per_edge=20)

    # Create an instance of the Evolution class
    evolution = Evolution(all_pieces, default_container)
    # Run the evolution process
    best_layout = evolution.run()
    # Print the best layout
    print("Best layout:")
    for piece in best_layout.genes:
        print(piece.id)
    print(f"Fitness: {best_layout.fitness}")
