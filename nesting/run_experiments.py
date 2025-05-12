import os
import matplotlib.pyplot as plt
from itertools import product
import pandas as pd
import seaborn as sns

from nesting.path_extractor import PatternPathExtractor
from .evolution import Evolution
from .layout import Container

def param_sweep(pieces, container):
    # Example parameter lists
    population_sizes = [10, 20]
    elite_sizes = [3, 5]
    mutation_rates = [0.1, 0.2, 0.3]
    pmx_values = [True, False]
    num_generations = [5, 10, 15]

    for (pop_size, elite_size, mut_rate, pmx, num_gen) in product(
        population_sizes, elite_sizes, mutation_rates, pmx_values, num_generations
    ):
        param_label = (
            f"pop{pop_size}_elite{elite_size}_mut{mut_rate}_"
            f"crossover_{'pmx' if pmx else 'ox'}_gen{num_gen}"
        )
        results_dir = os.path.join("sweep_results", param_label)
        os.makedirs(results_dir, exist_ok=True)

        print(f"\nRunning GA with parameters: {param_label}")
        print("-" * 50)
        print("\n")

        # Lists to track fitness across generations
        gen_all_fitnesses = []
        gen_best_fitness = []
        gen_avg_fitness = []

        # Create & initialize the Evolution object
        evol = Evolution(
            pieces,
            container,
            num_generations=num_gen,
            population_size=pop_size,
            elite_population_size=elite_size,
            mutation_rate=mut_rate,
            pmx=pmx
        )
        evol.generate_population()

        # Loop over each generation and store fitness data
        for g in range(num_gen):
            evol.next_generation()
            
            # Collect fitnesses from the current population
            fitness_list = [chrom.fitness for chrom in evol.population]
            gen_all_fitnesses.append(fitness_list)
            
            # Record best & average fitness
            best_fit = max(fitness_list)
            avg_fit = sum(fitness_list) / len(fitness_list)
            gen_best_fitness.append(best_fit)
            gen_avg_fitness.append(avg_fit)

        # Build a DataFrame of all fitness data for a swarm plot
        all_data = []
        for gen_idx, fitness_list in enumerate(gen_all_fitnesses):
            for fit in fitness_list:
                all_data.append({"Generation": gen_idx + 1, "Fitness": fit})
        df_fitness = pd.DataFrame(all_data)

        # 1) Swarm plot of all fitnesses per generation
        plt.figure(figsize=(7, 5))
        sns.swarmplot(data=df_fitness, x="Generation", y="Fitness", size=3, alpha=0.7)
        plt.title(f"All Fitnesses Swarm Plot – {param_label}")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        swarm_path = os.path.join(results_dir, "all_fitnesses_swarm.png")
        plt.savefig(swarm_path, dpi=300)
        plt.close()

        # 2) Plot average child fitness from Evolution
        gens = range(1, len(evol.avg_child_fitnesses) + 1)
        plt.figure()
        plt.plot(gens, evol.avg_child_fitnesses, marker='o', label='Avg Child Fitness')
        plt.title(f"Average Child Fitness – {param_label}")
        plt.xlabel("Generation")
        plt.ylabel("Average Child Fitness")
        plt.legend()
        avg_child_path = os.path.join(results_dir, "avg_child_fitness.png")
        plt.savefig(avg_child_path, dpi=300)
        plt.close()

        # 3) Survival rate
        plt.figure()
        plt.plot(gens, evol.survival_rates, marker='x', color='green', label='Survival Rate')
        plt.title(f"Survival Rate – {param_label}")
        plt.xlabel("Generation")
        plt.ylabel("Survival Rate")
        plt.legend()
        survival_path = os.path.join(results_dir, "survival_rate.png")
        plt.savefig(survival_path, dpi=300)
        plt.close()

        # 4) Best & average fitness
        plt.figure()
        plt.plot(range(1, num_gen + 1), gen_best_fitness, marker='o', label='Best Fitness')
        plt.plot(range(1, num_gen + 1), gen_avg_fitness, marker='x', label='Avg Fitness')
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title(f"Best & Average Fitness – {param_label}")
        plt.legend()
        plt.savefig(os.path.join(results_dir, "best_avg_fitness.png"), dpi=300)
        plt.close()

        # Save CSV of per-generation data
        csv_path = os.path.join(results_dir, "generation_data.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("Generation,BestFitness,AvgFitness,SurvivalRate,AvgChildFitness,AllFitnesses\n")
            for i in range(num_gen):
                best_val = gen_best_fitness[i]
                avg_val = gen_avg_fitness[i]
                survival_val = evol.survival_rates[i] if i < len(evol.survival_rates) else None
                child_avg_val = evol.avg_child_fitnesses[i] if i < len(evol.avg_child_fitnesses) else None
                all_fits = gen_all_fitnesses[i] if i < len(gen_all_fitnesses) else []
                row = (
                    f"{i+1},{best_val},{avg_val},"
                    f"{survival_val},{child_avg_val},{all_fits}\n"
                )
                f.write(row)

        print(f"Completed all {num_gen} generations for {param_label}.")
        print(f"Results (plots, CSV, logs) in {results_dir}.\n\n")



if __name__ == "__main__":
    default_container = Container(140, 200)
    default_path = "/Users/aysegulbarlas/codestuff/GarmentCode/nesting-assets/Configured_design_specification_asym_dress.json"
    extractor = PatternPathExtractor(default_path)
    all_pieces = extractor.get_all_panel_pieces(samples_per_edge=20)
    param_sweep(all_pieces, default_container)