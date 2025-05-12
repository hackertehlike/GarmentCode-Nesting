import os
import matplotlib.pyplot as plt
from itertools import product

from nesting.path_extractor import PatternPathExtractor
from .evolution import Evolution
from .layout import Container

def run_experiments(pieces, container, num_gen):
    """
    Runs multiple GA experiments with different population sizes.
    Plots best fitness vs. generation for each experiment.
    """
    population_settings = [10, 20, 30]  # example values
    generation_records = {}

    for pop_size in population_settings:
        # Collect best fitness per generation for a single experiment
        best_fitness_per_gen = []
        
        # Create or update an Evolution instance
        evol = Evolution(pieces, container)
        
        # Adjust your code to allow customizing population size, e.g.,
        # evol.population_size = pop_size  # if you have such a property

        evol.generate_population()
        for gen in range(num_gen):
            evol.next_generation()
            current_best = evol.get_elite()[0].fitness
            best_fitness_per_gen.append(current_best)
        
        generation_records[pop_size] = best_fitness_per_gen

    # Plot the results
    for pop_size, fitness_data in generation_records.items():
        plt.plot(fitness_data, label=f"Pop Size {pop_size}")

    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("GA Runs with Different Population Sizes")
    plt.legend()
    plt.savefig("my_ga_plot.png", dpi=300)  # Replace with your desired filepath
    # plt.show()  # Omit if you only want to save and not display

def run_ablation_study(pieces, container):
    """
    Runs an ablation study over various GA parameters, saving
    plots (best fitness, average fitness, all fitnesses,
    survival rate, and average child fitness) in a directory
    structure based on each parameter combination.
    """

    # Example parameter grids (adjust as needed)
    population_sizes = [10, 20]
    elite_sizes = [3, 5]
    mutation_rates = [0.1, 0.2, 0.3]
    pmx_values = [True, False]
    num_generations = [10, 15]  # or adjust as needed

    # Loop over all combinations of parameters
    for (pop_size, elite_size, mut_rate, pmx, num_gen) in product(
        population_sizes, elite_sizes, mutation_rates, pmx_values, num_generations
    ):
        print(f"Running GA with parameters: pop_size={pop_size}, "
                f"elite_size={elite_size}, mut_rate={mut_rate}, "
                f"pmx={pmx}, num_gen={num_gen}")
        
        # Create a results directory for this combination
        param_label = (
            f"pop{pop_size}_elite{elite_size}_mut{mut_rate}_"
            f"crossover_{'pmx' if pmx else 'ox'}_gen{num_gen}"
        )
        results_dir = os.path.join("ablation_results", param_label)
        os.makedirs(results_dir, exist_ok=True)

        # Track fitness metrics for each generation
        gen_best_fitness = []
        gen_avg_fitness = []
        gen_all_fitnesses = []

        # Create the Evolution object with the chosen parameters
        evol = Evolution(
            pieces,
            container,
            num_generations=num_gen,
            population_size=pop_size,
            elite_population_size=elite_size,
            mutation_rate=mut_rate,
            pmx=pmx
        )
        print(f"Running GA with parameters: {param_label}")
        # Generate the initial population
        evol.generate_population()

        # For each generation, track best, average, and all fitnesses
        for gen in range(num_gen):
            print(f"Running generation {gen + 1}/{num_gen} for {param_label}...")
            evol.next_generation()

            # Sort population after generation so index 0 is best
            evol.get_elite()

            fitnesses = [chrom.fitness for chrom in evol.population]
            best_fitness = fitnesses[0]
            avg_fitness = sum(fitnesses) / len(fitnesses)

            gen_best_fitness.append(best_fitness)
            gen_avg_fitness.append(avg_fitness)
            gen_all_fitnesses.append(fitnesses)

        # At this point, evol.survival_rates and evol.avg_child_fitnesses
        # each have num_generations entries
        gen_survival_rates = evol.survival_rates
        gen_child_fitness = evol.avg_child_fitnesses

        # Plot best & average fitness
        plt.figure()
        plt.plot(gen_best_fitness, marker='o', label='Best Fitness')
        plt.plot(gen_avg_fitness, marker='x', label='Average Fitness')
        plt.title(f"Best & Average Fitness: {param_label}")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend()
        plt.savefig(os.path.join(results_dir, "fitness_plot.png"), dpi=300)
        plt.close()

        # Plot all fitnesses per generation (boxplot)
        plt.figure()
        plt.boxplot(gen_all_fitnesses, positions=range(len(gen_all_fitnesses)))
        plt.title(f"All Fitnesses per Generation (Boxplot): {param_label}")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.xticks(range(len(gen_all_fitnesses)), range(len(gen_all_fitnesses)))  # label gens
        plt.savefig(os.path.join(results_dir, "all_fitnesses_boxplot.png"), dpi=300)
        plt.close()

        # Plot survival rate
        plt.figure()
        plt.plot(gen_survival_rates, marker='o', label='Elite Survival Rate')
        plt.title(f"Elite Survival Rate: {param_label}")
        plt.xlabel("Generation")
        plt.ylabel("Survival Rate (fraction)")
        plt.legend()
        plt.savefig(os.path.join(results_dir, "survival_rate.png"), dpi=300)
        plt.close()

        # Plot average child fitness
        plt.figure()
        plt.plot(gen_child_fitness, marker='x', label='Avg Child Fitness')
        plt.title(f"Average Child Fitness: {param_label}")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend()
        plt.savefig(os.path.join(results_dir, "avg_child_fitness.png"), dpi=300)
        plt.close()

        # Optionally, save raw data (including new metrics) to CSV
        data_file = os.path.join(results_dir, "fitness_data.csv")
        with open(data_file, "w", encoding="utf-8") as f:
            f.write(
                "Generation,BestFitness,AverageFitness,"
                "SurvivalRate,ChildFitness,AllFitnesses\n"
            )
            for g in range(num_gen):
                f.write(
                    f"{g},"
                    f"{gen_best_fitness[g]},"
                    f"{gen_avg_fitness[g]},"
                    f"{gen_survival_rates[g]},"
                    f"{gen_child_fitness[g]},"
                    f"{gen_all_fitnesses[g]}\n"
                )

        print(f"Ablation results saved under {results_dir}.\n")

if __name__ == "__main__":
    """
    Main function to run the ablation study.
    """
    # Example usage:
    default_container = Container(140, 200)
    default_path = "/Users/aysegulbarlas/codestuff/GarmentCode/nesting-assets/Configured_design_specification_asym_dress.json"
    extractor = PatternPathExtractor(default_path)
    print("Extracting all panel pieces...")
    all_pieces = extractor.get_all_panel_pieces(samples_per_edge=20)

    # Run the experiments
    #num_generations = 10  # Example value
    #run_experiments(all_pieces, default_container, num_generations)
    print("Running ablation study...")
    # Run the ablation study
    run_ablation_study(all_pieces, default_container)

# Example usage:
# from .path_extractor import PatternPathExtractor
# from .layout import Container
# def main():
#     container = Container(140, 200)
#     path = "/path/to/your/pattern.json"
#     extractor = PatternPathExtractor(path)
#     pieces = extractor.get_all_panel_pieces(samples_per_edge=20)
#     run_ablation_study(pieces, container)