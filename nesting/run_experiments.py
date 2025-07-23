"""
Extended GA parameter sweep
–––––––––––––––––––––––––––
* New hyper‑parameter grid (see lists below)
* Automatic comparison plots for **each** parameter
* Master CSVs for generations and per‑run summaries
"""

from __future__ import annotations
import os, time, itertools, shutil
from functools import partial
from itertools import product
#from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import nesting.config as config

from nesting.path_extractor import PatternPathExtractor
from .evolution import Evolution
from .layout import Container

# -----------------------------------------------------------------------------
# Hyper-parameter grid
# -----------------------------------------------------------------------------
population_sizes    = [20, 50, 100]
elite_sizes         = [3, 5, 10, 15]
mutation_rates      = [0.1]
num_generations_ls  = [150]

# -----------------------------------------------------------------------------
# Helper: save a simple line or box plot
# -----------------------------------------------------------------------------
def _save_plot(fig, out_dir: str, fname: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)
    
def _calc_fitness(child, container):
    """
    Compute fitness for one individual and return it.
    Kept very small so it is picklable.
    """
    child.fitness = child.calculate_fitness(container)  # or whatever you do
    return child

# -----------------------------------------------------------------------------
# Main sweep driver
# -----------------------------------------------------------------------------
def param_sweep(pieces, container, *, log_interval: int = 20, plot_progress: bool = True ) -> None: 
    """
    Runs GA on the full Cartesian product of the parameter grid and
    produces – in addition to the per‑run artefacts – aggregate CSVs and
    diagnostic plots comparing *one* parameter at a time.
    """

    master_rows_gen:  list[dict] = []     # 1 row / generation
    master_rows_run:  list[dict] = []     # 1 row / run (final generation)

    combinations = list(product(population_sizes, elite_sizes,
                                mutation_rates, num_generations_ls))

    print(f"Total combinations to run: {len(combinations)}")

    # ------------------------------------------------------------------ sweep
    for (pop, elite, mut_r, crossover, gens) in combinations:
        label = (
            f"pop{pop}_elite{elite}_mut{mut_r}_"
            f"cross{crossover}_gen{gens}"
        )
                # ------------------------------------------------------------------ sweep (inside the for-each-combination loop)

        out_dir = os.path.join("sweep_results_3", label)
        shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir, exist_ok=True)        # ← create first!

        # ---------- rolling CSV & progress-plot folder ---------------------
        header = ("Generation,BestFitness,AvgChildFitness,DeltaBest\n")

        progress_csv = os.path.join(out_dir, "generation_progress.csv")
        with open(progress_csv, "w") as fh:        # write header once
            fh.write(header)

        prog_dir = os.path.join(out_dir, "progress_plots")
        if plot_progress:
            os.makedirs(prog_dir, exist_ok=True)

        print(f"\n— Running {label} —")
        t0 = time.time()

        # ---------------- run GA ----------------
        evo = Evolution(
            pieces,
            container,
            num_generations        = gens,
            population_size        = pop,
            elite_population_size  = elite,
            mutation_rate          = mut_r,
            crossover_method       = crossover,
            enable_dynamic_stopping= config.ENABLE_DYNAMIC_STOPPING,
            early_stop_window      = config.EARLY_STOP_WINDOW,
            early_stop_tolerance   = config.EARLY_STOP_TOLERANCE,
            enable_extension       = config.ENABLE_EXTENSION,
            extend_window          = config.EXTEND_WINDOW,
            extend_threshold       = config.EXTEND_THRESHOLD,
            max_generations        = config.MAX_GENERATIONS,
        )

        evo.generate_population()
        per_gen_fitness: list[list[float]] = []

        for g_idx in range(gens):
            evo.next_generation()
            per_gen_fitness.append([c.fitness for c in evo.population])

            # ──────────────────────────────────────────────────────────────
            # STEP 3 – PROGRESS LOG & OPTIONAL QUICK PLOTS  (copy as-is)
            # ──────────────────────────────────────────────────────────────
            g = g_idx + 1                         # human-friendly generation number

            if (g % log_interval == 0) or (g == gens):
                with open(progress_csv, "a") as fh:
                    fh.write(
                        f"{g},"
                        f"{evo.best_fitness_history[g]},"
                        f"{evo.avg_child_fitnesses[g_idx]},"
                        f"{evo.delta_best[g]}\n"
                    )

                # ---- 3-B  brief console read-out ------------------------
                print(
                    f"[{label}] Gen {g:3d} | "
                    f"best={evo.best_fitness_history[g]:.4f}  "
                    f"avg_child={evo.avg_child_fitnesses[g_idx]:.4f}  "
                    f"Δ_best={evo.delta_best[g]:+.4f}"
                )

                # ---- 3-C  overwrite quick progress plots ----------------
                if plot_progress:
                    def _quick(series, title, fname,
                                ylabel="Value", marker="o"):
                        fig, ax = plt.subplots()
                        ax.plot(range(1, g + 1), series,
                                marker=marker, ms=4, lw=1)
                        ax.set(xlabel="Generation", ylabel=ylabel, title=title)
                        ax.grid(True)
                        _save_plot(fig, prog_dir, fname)   # overwrites previous file

                    _quick(evo.best_fitness_history[1:g + 1],
                            f"Best Fitness ≤ Gen {g}", "best_fitness.png")
                    _quick(evo.avg_child_fitnesses[:g_idx + 1],
                            f"Avg Child Fitness ≤ Gen {g}", "avg_child_fitness.png")
                    _quick(evo.delta_best[1:g + 1],
                            f"Δ-Best ≤ Gen {g}", "delta_best.png",
                            ylabel="Δ-Best", marker="x")

        # ---------------- small helper for individual plots -----------------
        def line_plot(series, title, fname, ylabel="Value", marker="o"):
            fig, ax = plt.subplots()
            ax.plot(range(1, gens + 1), series, marker=marker)
            ax.set(xlabel="Generation", ylabel=ylabel, title=title)
            ax.grid(True)
            _save_plot(fig, out_dir, fname)

        # -------------------------------------------------------------------
        # 1. Per‑run plots (unchanged part, plus operator gains together)
        # -------------------------------------------------------------------
        df_swarm = pd.DataFrame(
            {"Generation": g + 1, "Fitness": f}
            for g, flist in enumerate(per_gen_fitness) for f in flist
        )
        fig = plt.figure(figsize=(7, 5))
        sns.swarmplot(data=df_swarm, x="Generation", y="Fitness", size=3, alpha=0.7)
        plt.title(f"All Fitnesses – {label}")
        _save_plot(fig, out_dir, "all_fitnesses_swarm.png")

        line_plot(evo.avg_child_fitnesses, f"Avg Child Fitness – {label}",
                  "avg_child_fitness.png")
        line_plot(evo.best_fitness_history[1:], f"Best Fitness – {label}",
                  "best_fitness.png")
        line_plot(evo.delta_best[1:], f"Δ‑Best – {label}",
                  "delta_best.png", ylabel="Δ‑Best")

        # -------------------------------------------------------------------
        # 2. CSVs for this run
        # -------------------------------------------------------------------
        header = ("Generation,BestFitness,AvgChildFitness,DeltaBest\n")
        with open(os.path.join(out_dir, "generation_data.csv"), "w") as fh:
            fh.write(header)
            for g in range(gens):
                fh.write(
                    f"{g+1},"
                    f"{evo.best_fitness_history[g+1]},"
                    f"{evo.avg_child_fitnesses[g]},"
                    f"{evo.delta_best[g+1]}\n"
                )

        with open(os.path.join(out_dir, "run_log.txt"), "w") as fh:
            fh.write("\n".join(evo.log_lines))

        # -------------------------------------------------------------------
        # 3. Append to master DataFrames
        # -------------------------------------------------------------------
        def _param_dict() -> dict:
            return dict(population_size=pop, elite_size=elite, mutation_rate=mut_r,
                        crossover=crossover, generations=gens)

        # per‑generation rows
        for g in range(gens):
            master_rows_gen.append(
                dict(_param_dict(),
                     generation        = g + 1,
                     best_fitness      = evo.best_fitness_history[g+1],
                     avg_child_fitness = evo.avg_child_fitnesses[g],
                     delta_best        = evo.delta_best[g+1])
            )

        # per‑run summary (final generation only)
        master_rows_run.append(
            dict(_param_dict(),
                 best_fitness      = evo.best_fitness_history[-1],
                 avg_child_fitness = evo.avg_child_fitnesses[-1],
                 delta_best        = evo.delta_best[-1],
                 elapsed_sec       = time.time() - t0)
        )

        print(f"✔  finished {label} in {time.time() - t0:.1f}s  →  {out_dir}")

    # -----------------------------------------------------------------------
    # 4. Master CSV dumps
    # -----------------------------------------------------------------------
    df_gen = pd.DataFrame(master_rows_gen)
    df_run = pd.DataFrame(master_rows_run)
    df_gen.to_csv("all_generations.csv", index=False)
    df_run.to_csv("run_summary.csv",   index=False)
    print("\nMaster CSVs written: all_generations.csv, run_summary.csv")

    # -----------------------------------------------------------------------
    # 5. Automatic comparison plots (one parameter at a time)
    # -----------------------------------------------------------------------
    comparison_root = "comparison_plots_2"
    shutil.rmtree(comparison_root, ignore_errors=True)

    # Utility: median best‑fitness per generation for a *fixed* param dict
    def _median_series(df, group_cols):
        return (df
                .groupby(group_cols + ["generation"])["best_fitness"]
                .median()
                .unstack(level=-1))        # → rows: group signature, cols: generation

    param_columns = ["population_size", "elite_size", "mutation_rate",
                     "crossover", "generations"]

    for target_param in param_columns:
        other_params = [c for c in param_columns if c != target_param]

        for combo_vals, sub_df in df_gen.groupby(other_params):
            # combo_vals is a tuple aligned with other_params
            sig = "_".join(f"{k}{v}" for k, v in zip(other_params, combo_vals))
            out_dir = os.path.join(comparison_root, target_param, sig)
            os.makedirs(out_dir, exist_ok=True)

            # 5‑A line plot (median best‑fitness per generation)
            med = _median_series(sub_df, [target_param])
            fig, ax = plt.subplots()
            for val, row in med.iterrows():
                ax.plot(row.index, row.values, marker="o", label=str(val))
            ax.set(xlabel="Generation", ylabel="Median best fitness",
                   title=f"{target_param} – {sig} (medians)")
            ax.legend(title=target_param); ax.grid(True)
            _save_plot(fig, out_dir, "median_best_fitness.png")

            # 5‑B box plot (final best fitness distribution)
            last_gen = sub_df[sub_df["generation"] == sub_df["generations"]]
            fig = plt.figure(figsize=(6, 4))
            sns.boxplot(data=last_gen, x=target_param, y="best_fitness")
            plt.title(f"Final best fitness vs {target_param}\n{sig}")
            _save_plot(fig, out_dir, "final_best_fitness_box.png")

    print(f"All comparison plots written under: {comparison_root}")


# -----------------------------------------------------------------------------
# CLI convenience
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    container = Container(140, 500)
    default_path = "/Users/aysegulbarlas/codestuff/GarmentCode/nesting-assets/Configured_design_specification_asym_dress.json"
    pieces = PatternPathExtractor(default_path).get_all_panel_pieces(samples_per_edge=3)
    param_sweep(pieces, container)
