
import os
from itertools import product
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from nesting.path_extractor import PatternPathExtractor

# –– local import: Evolution class just defined above –––––––––––
from .evolution import Evolution  # noqa: E402
from .layout import Container  # noqa: E402


# ------------------------------------------------------------------
# Parameter sweep driver
# ------------------------------------------------------------------


def param_sweep(pieces, container) -> None:
    """Run a grid search over GA hyper‑parameters and plot rich metrics."""

    population_sizes   = [10, 20]
    elite_sizes        = [3, 5]
    mutation_rates     = [0.1, 0.2, 0.3]
    pmx_values         = [True, False]
    num_generations_ls = [5, 10, 15]
    duplicate_policy   = [True, False]

    # ------------------------------------------------------------------
    for pop, elite, mut_r, pmx, gens, allow_dups in product(
        population_sizes, elite_sizes, mutation_rates, pmx_values, num_generations_ls, duplicate_policy
    ):
        label = (
            f"pop{pop}_elite{elite}_mut{mut_r}_"
            f"crossover_{'pmx' if pmx else 'ox'}_gen{gens}_dups_{'yes' if allow_dups else 'no'}"
        )
        out_dir = os.path.join("sweep_results_2", label)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\nRunning GA with parameters: {label}\n{'-'*50}\n")
        start = time.time()

        # ------------------ run GA ------------------

        evo = Evolution(
            pieces,
            container,
            num_generations=gens,
            population_size=pop,
            elite_population_size=elite,
            mutation_rate=mut_r,
            pmx=pmx,
            allow_duplicate_genes=allow_dups,
        )

        # ------------------ run & capture per‑generation fitnesses ------------------
        gen_all_fitnesses: list[list[float]] = []
        evo.generate_population()
        for _ in range(gens):
            evo.next_generation()
            gen_all_fitnesses.append([c.fitness for c in evo.population])

        # ------------------ plotting helpers ------------------
        def save_line_plot(data, title, fname, ylabel="Value", marker="o"):
            plt.figure(); plt.plot(range(1, gens + 1), data, marker=marker)
            plt.title(title); plt.xlabel("Generation"); plt.ylabel(ylabel); plt.grid(True)
            plt.savefig(os.path.join(out_dir, fname), dpi=300); plt.close()

        # Swarm plot of all fitnesses per generation -----------------------
        df_swarm = pd.DataFrame(
            {"Generation": g_idx + 1, "Fitness": fit}
            for g_idx, fit_list in enumerate(gen_all_fitnesses)
            for fit in fit_list
        )
        plt.figure(figsize=(7, 5))
        sns.swarmplot(data=df_swarm, x="Generation", y="Fitness", size=3, alpha=0.7)
        plt.title(f"All Fitnesses – {label}")
        plt.savefig(os.path.join(out_dir, "all_fitnesses_swarm.png"), dpi=300)
        plt.close()

        # Individual metric plots ------------------------------------------
        save_line_plot(evo.avg_child_fitnesses, f"Avg Child Fitness – {label}", "avg_child_fitness.png")
        save_line_plot(evo.survival_rates, f"Survival Rate – {label}", "survival_rate.png", ylabel="Rate", marker="x")
        save_line_plot(evo.best_fitness_history[1:], f"Best Fitness – {label}", "best_fitness.png")
        save_line_plot(evo.delta_best[1:], f"Δ‑Best – {label}", "delta_best.png", ylabel="Δ‑Best")
        save_line_plot(
            evo.child_parent_success_ratio[1:],
            f"Child‑Parent Success Ratio – {label}",
            "success_ratio.png",
            ylabel="Ratio",
        )

        # Operator gains combined plot ------------------------------------
        plt.figure()
        plt.plot(range(1, gens + 1), evo.mean_crossover_gain[1:], marker="o", label="Mean Crossover Gain")
        plt.plot(range(1, gens + 1), evo.mean_mutation_gain[1:], marker="x", label="Mean Mutation Gain")
        plt.xlabel("Generation"); plt.ylabel("Mean Gain"); plt.title(f"Operator Gains – {label}")
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(out_dir, "operator_gains.png"), dpi=300); plt.close()


        duration = time.time() - start
        print(f"Elapsed time: {duration:.1f} s")
        evo.log_lines.append(f"Elapsed time: {duration:.1f} s")
                             
        # ------------------ CSV dump ------------------
        csv_hdr = [
            "Generation", "BestFitness", "AvgChildFitness", "SurvivalRate", "DeltaBest", "SuccessRatio", "CrossoverGain", "MutationGain"
        ]
        with open(os.path.join(out_dir, "generation_data.csv"), "w", encoding="utf-8") as fh:
            fh.write(",".join(csv_hdr) + "\n")
            for g in range(gens):
                fh.write(
                    f"{g+1},"
                    f"{evo.best_fitness_history[g+1]},"  # skip generation 0 placeholder
                    f"{evo.avg_child_fitnesses[g]},"
                    f"{evo.survival_rates[g]},"
                    f"{evo.delta_best[g+1]},"
                    f"{evo.child_parent_success_ratio[g+1]},"
                    f"{evo.mean_crossover_gain[g+1]},"
                    f"{evo.mean_mutation_gain[g+1]}\n"
                )

        # ------------------ log file ------------------
        with open(os.path.join(out_dir, "run_log.txt"), "w", encoding="utf-8") as fh:
            for line in evo.log_lines:
                fh.write(line + "\n")

        print(f"Completed {gens} generations for {label} → {out_dir}\n\n")


# ------------------------------------------------------------------
# Convenience CLI entry‑point
# ------------------------------------------------------------------

if __name__ == "__main__":
    default_container = Container(140, 200)
    default_path = "/Users/aysegulbarlas/codestuff/GarmentCode/nesting-assets/Configured_design_specification_asym_dress.json"
    extractor = PatternPathExtractor(default_path)
    pieces = extractor.get_all_panel_pieces(samples_per_edge=20)
    param_sweep(pieces, default_container)
