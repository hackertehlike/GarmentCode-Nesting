from __future__ import annotations
import copy
import csv
import multiprocessing
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .layout import Piece, Container
from .chromosome import Chromosome
import nesting.config as config

# evolution.py
def _eval_fitness(chrom: "Chromosome") -> "Chromosome":
    """Worker helper for ProcessPoolExecutor: recompute fitness in place."""
    chrom.calculate_fitness()            # container is already held by chrom
    return chrom

class Evolution:

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        pieces: dict[str, Piece],
        container: Container,
        num_generations: int = 10,
        population_size: int = 10,
        #elite_population_size: int = 5,
        mutation_rate: float = 0.2,
        crossover_method: str = "pmx",
        #pmx: bool = True,`
        # allow_duplicate_genes: bool = False,
        # max_duplicate_retries: int = 50,
        enable_dynamic_stopping: bool = True,
        early_stop_window: int = 20,
        early_stop_tolerance: float = 1e-4,
        enable_extension: bool = True,
        extend_window: int = 10,
        extend_threshold: float = 0.1,
        max_generations: int = 200,
        design_params: dict = None,
        body_params: object = None,
        design_sampler: object = None,
    ) -> None:
        self.generation = 0
        self.container = container
        self.pieces = pieces
        self.population: list[Chromosome] = []
        
        # Initialize logging first to avoid attribute errors
        self.log_lines = []
        
        # Setup log files and directories
        ts = time.strftime("%Y%m%d_%H%M%S")
        log_dir = config.SAVE_LOGS_PATH
        
        # append timestamp to log path
        log_dir = f"{log_dir}_{ts}"
        Path(log_dir).mkdir(parents=True, exist_ok=True)        # ensure folder exists
        
        self.log_path = Path(log_dir) / f"evolution_log_{ts}.txt"
        self.csv_path = Path(log_dir) / f"evolution_metrics_{ts}.csv"
        
        plots_dir = Path(log_dir) / "plots"
        plots_dir.mkdir(exist_ok=True)

        self.plot_path = plots_dir / f"fitness_curves_{ts}.png"
        self.gain_plot_path = plots_dir / f"mean_gains_{ts}.png"
        self.swarm_plot_path = plots_dir / f"fitness_swarm_{ts}.png"
        self.delta_best_plot_path = plots_dir / f"delta_best_{ts}.png"
        self.mut_plot_path = plots_dir / "mutation_gains.png" 

        self._metrics_buffer = []
        self._all_metrics = []
        self._csv_header_done = False
        self._swarm_df = pd.DataFrame(columns=["generation", "fitness", "origin"])

        # GA parameters
        self.num_generations = num_generations
        self.population_size = population_size
        #self.elite_population_size = elite_population_size
        self.mutation_rate = mutation_rate
        self.crossover_method = crossover_method
        #self.pmx = pmx
        # self.allow_duplicate_genes = allow_duplicate_genes
        # self.max_duplicate_retries = max_duplicate_retries
        
        # Design parameters
        self.design_params = design_params
        self.body_params = body_params
        self.design_sampler = design_sampler

        # ── early stopping ───────────────────────────────────────────────
        self.enable_dynamic_stopping = enable_dynamic_stopping
        self.early_stop_window = early_stop_window
        self.early_stop_tolerance = early_stop_tolerance
        self.enable_extension = enable_extension
        self.extend_window = extend_window
        self.extend_threshold = extend_threshold
        self.max_generations = max_generations

        self.population_weights = config.POPULATION_WEIGHTS
        self.n_elites    = int(self.population_weights['elites'] * population_size)
        self.n_offspring = int(self.population_weights['offspring'] * population_size)
        self.n_mutants   = int(self.population_weights['mutants']   * population_size)
        self.n_randoms   = int(self.population_weights['randoms']   * population_size)
        self.remainder   = population_size - (self.n_elites + self.n_offspring + self.n_mutants + self.n_randoms)
        self.n_mutants  += self.remainder  # put leftovers into mutants


        # run‑state flags
        self.early_stopped: bool = False
        self.extended_generations: int = 0


        # ── metrics ────────────────────────────────────────────
        self.avg_child_fitnesses: list[float] = []
        self.best_fitness_history: list[float] = []
        self.delta_best: list[float] = []
        self.pop_fitness_history: list[list[float]] = []
        self.mean_offspring_gain: list[float] = []
        self.mean_mutant_gain: list[float] = []
        self.mutation_perf: dict[str, list[float]] = {}      # per-generation mean gain
        
        self._log(
            f"Evolution initialized with {self.population_size} chromosomes, "
            f"{self.n_elites} elites, {self.n_offspring} offspring, "
            f"{self.n_mutants} mutants, {self.n_randoms} randoms.",
            divider=True
        )
        
        # Log design parameter status
        if self.design_params:
            self._log(f"Design parameters loaded: {len(self._flatten_params(self.design_params))} parameters found")
        else:
            self._log("WARNING: No design parameters provided to Evolution constructor")
            
        if self.body_params is None:
            self._log("WARNING: No body parameters provided to Evolution constructor")
            
        if self.design_sampler is None:
            self._log("WARNING: No design sampler provided to Evolution constructor")
    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------

    def _log(self, msg: str = "", *, divider: bool = False) -> None:
        if divider:
            line = "-" * 60
            print(line)
            self.log_lines.append(line)
        if msg:
            print(msg)
            self.log_lines.append(msg)

    # ------------------------------------------------------------------
    # Population helpers
    # ------------------------------------------------------------------

    def _generate_random_chromosome(self) -> Chromosome:
        
        #self._log("Generating random chromosome...")
        ids_ = list(self.pieces.keys())
        random.shuffle(ids_)
        chrom = Chromosome(
            [self.pieces[i] for i in ids_], 
            self.container,
            origin="random",
            design_params=self.design_params, 
            body_params=self.body_params,
            design_sampler=self.design_sampler
        )

        # ramdomize rotations
        for piece in chrom.genes:
            rotation = random.choice([0, 90, 180, 270])
            # rotation = 90
            if rotation != 0:
                piece.rotate(rotation)
        
        #chrom.sync_order()
        chrom.calculate_fitness()

        if config.VERBOSE:
            self._log(f"Random chromosome generated with fitness: {chrom.fitness:.4f}")
        return chrom

    def generate_population(self) -> None:
        # Initialize design parameter tracking
        self.design_param_changes = []
        
        # generate initial population in parallel
        self._log("Generating initial population...")

        if config.MULTITHREADING:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(self._generate_random_chromosome) for _ in range(self.population_size)]
                self.population = [f.result() for f in futures]
        else:
            self.population = [self._generate_random_chromosome() for _ in range(self.population_size)]

        self.pop_fitness_history.append([chrom.fitness for chrom in self.population])
        # Record initial generation (gen 0) metrics for plots and CSV
        best0 = max(c.fitness for c in self.population)
        avg_pop = sum(c.fitness for c in self.population) / len(self.population)
        row0 = {
            'generation': 0,
            'avg_child_fitness': avg_pop,
            'avg_elite': avg_pop,
            'avg_off': 0.0,
            'avg_mut': 0.0,
            'avg_rand': avg_pop,
            'best_fit': best0,
            'delta_best': 0.0,
            'mean_offspring_gain': 0.0,
            'mean_mutant_gain': 0.0,
        }
        self._all_metrics.append(row0)
        self._metrics_buffer.append(row0)
        # Seed best and delta history
        self.best_fitness_history.append(best0)
        self.delta_best.append(0.0)
        # Seed swarm plot data for gen 0
        frame0 = pd.DataFrame([
            {'generation': 0, 'fitness': c.fitness, 'origin': c.origin or 'initial'}
            for c in self.population
        ])
        self._swarm_df = pd.concat([self._swarm_df, frame0], ignore_index=True)
        
        self._log(f"Initial population of {self.population_size} layouts generated.", divider=True)
        for i, chrom in enumerate(self.population):
            self._log(f"Layout {i}: {[(p.id, p.rotation) for p in chrom.genes]} | Fitness: {chrom.fitness:.4f}")
        self._log("Population generation completed.", divider=True)


    # ------------------------------------------------------------------
    # GA core operations
    # ------------------------------------------------------------------

    def _get_elite(self) -> list[Chromosome]:

        # Filter out unfit chromosomes
        viable = [c for c in self.population if c.fitness > 0.0]

        if not viable:
            self._log("No viable chromosomes found (all fitness = 0.0)", divider=True)
            return []

        # Sort viable chromosomes by fitness (descending)
        viable.sort(key=lambda c: c.fitness, reverse=True)

        # Check if we have enough elite chromosomes
        elite_count = min(self.n_elites, len(viable))
        elite_list = viable[:elite_count]
        
        # If we don't have enough elites, pad with random chromosomes
        if elite_count < self.n_elites:
            missing_count = self.n_elites - elite_count
            self._log(f"Only {elite_count} viable chromosomes found, padding with {missing_count} random chromosomes", divider=True)
            
            # Generate random chromosomes to fill the gap
            random_chromosomes = []
            for _ in range(missing_count):
                random_chrom = self._generate_random_chromosome()
                random_chrom.origin = "random_elite_padding"
                random_chromosomes.append(random_chrom)
            
            elite_list.extend(random_chromosomes)
        
        self._log(
            f"Returning elite layouts. Best fitness: {elite_list[0].fitness:.4f}",
            divider=True
        )

        # print the elite chromosomes
        for i, chrom in enumerate(elite_list):
            source = " (random padding)" if i >= elite_count else ""
            #printout the signature of the chromosome
            self._log(f"Elite {i+1}: {chrom._signature()} | Fitness: {chrom.fitness:.4f}{source}")

        return elite_list


    
    def _generate_offspring(self, old_pop: list[Chromosome]) -> Chromosome:
        """
        Create one child chromosome by crossover (and optional mutation) and
        return it.  Duplicate-avoidance is honoured unless
        self.allow_duplicate_genes is True.
        """
        start   = time.time()
        #retries = 0

        p1, p2 = random.sample(old_pop, 2)

        # --- crossover ---
        # child = p1.crossover_pmx(p2) if self.pmx else p1.crossover_ox1_k(p2)

        if self.crossover_method == "pmx":
            child = p1.crossover_pmx(p2)
        elif self.crossover_method == "ox1":
            child = p1.crossover_ox1(p2, config.OX_K)
        else:
            raise ValueError(f"Unknown crossover method: {self.crossover_method}")

        # --- mutation (optional) ---
        if random.random() < self.mutation_rate:
            child.mutate()

        # --- fitness evaluation ---
        # use ProcessPoolExecutor to parallelize fitness evaluation
        
        child.calculate_fitness() 

        #self._log(f"Offspring generated in {time.time() - start:.2f} s")
        return child


    
    def next_generation(self) -> None:
        self._log("Next generation…", divider=True)
        start = time.time()

        old_elite = self._get_elite()
        if not old_elite:
            self._log("No elite chromosomes found. Cannot create next generation.", divider=True)
            return

        # 1) Start new population with elites
        new_population = list(old_elite)
        avg_elite = sum(c.fitness for c in old_elite) / self.n_elites

        # Prepare containers for the three groups
        offspring, mutants, randoms = [], [], []
        off_gains, mut_gains = [], []

        # ────────────────
        # Phase A: Offspring
        # ────────────────
        self._log("Phase A: submitting offspring tasks…")
        offspring_jobs = {}
        with ThreadPoolExecutor() as offspring_executor:
            for _ in range(self.n_offspring):
                p1, p2 = random.sample(old_elite, 2)
                parent_f = max(p1.fitness, p2.fitness)

                def do_offspring(p1=p1, p2=p2, parent_f=parent_f):
                    try:
                        if self.crossover_method == 'pmx':
                            child = p1.crossover_pmx(p2)
                        else:
                            child = p1.crossover_ox1(p2)
                        if random.random() < self.mutation_rate:
                            child.mutate()
                        child.calculate_fitness()
                        if config.VERBOSE:
                            self._log(f"Offspring created with fitness: {child.fitness:.4f}")
                        child.origin = "offspring"
                        return child, parent_f, None, None
                    except Exception as e:
                        return None, parent_f, None, e

                fut = offspring_executor.submit(do_offspring)
                offspring_jobs[fut] = None  # marker for offspring phase

            self._log("Phase A: collecting offspring results…")
            for fut in as_completed(offspring_jobs):
                child, parent_f, _, err = fut.result()
                if err is not None:
                    self._log(f"‼ offspring task failed: {err}")
                    continue
                gain = child.fitness - parent_f
                offspring.append(child)
                off_gains.append(gain)
                new_population.append(child)

        # ────────────────
        # Phase B: Mutants
        # ────────────────
        self._log("Phase B: submitting mutant tasks…")
        mutant_jobs = {}
        with ThreadPoolExecutor() as mutant_executor:
            for _ in range(self.n_mutants):
                parent = random.choice(old_elite)
                parent_f = parent.fitness

                def do_mutant(parent=parent, parent_f=parent_f):
                    try:
                        child = copy.deepcopy(parent)
                        child.mutate()
                        child.calculate_fitness()
                        if config.VERBOSE:
                            self._log(f"Mutant created with fitness: {child.fitness:.4f}")
                        child.origin = "mutant"
                        return child, parent_f, child.last_mutation, None
                    except Exception as e:
                        return None, parent_f, None, e

                fut = mutant_executor.submit(do_mutant)
                mutant_jobs[fut] = None  # marker for mutant phase

            self._log("Phase B: collecting mutant results…")
            for fut in as_completed(mutant_jobs):
                child, parent_f, op_used, err = fut.result()
                if err is not None:
                    self._log(f"‼ mutant task failed: {err}")
                    continue
                gain = child.fitness - parent_f
                mutants.append(child)
                mut_gains.append(gain)
                if op_used is not None:
                    self.mutation_perf.setdefault(op_used, []).append(gain)
                new_population.append(child)

        # ───────────────────
        # Phase C: Randoms
        # ───────────────────
        self._log("Phase C: submitting random-chromosome tasks…")
        random_jobs = {}
        with ThreadPoolExecutor() as random_executor:
            for _ in range(self.n_randoms):
                def do_random():
                    try:
                        child = self._generate_random_chromosome()
                        if config.VERBOSE:
                            self._log(f"Random chromosome created with fitness: {child.fitness:.4f}")
                        child.origin = "random"
                        return child, 0.0, None, None
                    except Exception as e:
                        return None, 0.0, None, e

                fut = random_executor.submit(do_random)
                random_jobs[fut] = None  # marker for random phase

            self._log("Phase C: collecting random-chromosome results…")
            for fut in as_completed(random_jobs):
                child, _, _, err = fut.result()
                if err is not None:
                    self._log(f"‼ random task failed: {err}")
                    continue
                randoms.append(child)
                new_population.append(child)

        # Summary of collected results
        self._log(
            f"Collected {len(offspring)} offspring, "
            f"{len(mutants)} mutants, {len(randoms)} randoms."
        )

        # 5) Compute per‐operator statistics
        avg_off  = sum(c.fitness for c in offspring) / len(offspring) if offspring else 0.0
        avg_mut  = sum(c.fitness for c in mutants)   / len(mutants)   if mutants else 0.0
        avg_rand = sum(c.fitness for c in randoms)   / len(randoms)   if randoms else 0.0

        mean_offspring_gain = sum(off_gains) / len(off_gains) if off_gains else 0.0
        mean_mutant_gain    = sum(mut_gains) / len(mut_gains) if mut_gains else 0.0
        self.mean_offspring_gain.append(mean_offspring_gain)
        self.mean_mutant_gain.append(mean_mutant_gain)

        self._log(
            f"Avg offspring gain vs parent: {mean_offspring_gain:+.4f}; "
            f"mutants: {mean_mutant_gain:+.4f}"
        )

        # 6) Finalize population and record history
        self.population = new_population
        self.pop_fitness_history.append([chrom.fitness for chrom in self.population])
        self.generation += 1

        frame = pd.DataFrame([{
            "generation": self.generation,
            "fitness":   c.fitness,
            "origin":    c.origin or "unknown"
        } for c in self.population])
        self._swarm_df = pd.concat([self._swarm_df, frame], ignore_index=True)

        best = max(c.fitness for c in new_population)
        delta = best - (self.best_fitness_history[-1] if self.best_fitness_history else best)
        self.best_fitness_history.append(best)
        self.delta_best.append(delta)
        self._log(f"Generation {self.generation}: best {best:.4f} (Δ {delta:+.4f})", divider=True)

        # 7) Log metrics
        row = {
            'generation': self.generation,
            'avg_child_fitness': sum(c.fitness for c in new_population) / len(new_population),
            'avg_elite': avg_elite,
            'avg_off': avg_off,
            'avg_mut': avg_mut,
            'avg_rand': avg_rand,
            'best_fit': best,
            'delta_best': delta,
            'mean_offspring_gain': mean_offspring_gain,
            'mean_mutant_gain': mean_mutant_gain,
        }
        row_mut = {
            f"mut_gain_{k}": (sum(v) / len(v) if v else 0.0)
            for k, v in self.mutation_perf.items()
        }
        row.update(row_mut)
        self.mutation_perf.clear()  # reset for next generation

        # Track design parameter changes
        self._track_design_param_changes(self.generation, new_population)

        self._metrics_buffer.append(row)
        self._all_metrics.append(row)

        if (len(self._metrics_buffer) >= config.GENERATION_PER_FLUSH
                or self.generation == self.num_generations) and config.SAVE_LOGS:
            self._flush_metrics()
            self.update_plots()

        self._log(f"Generation {self.generation} completed in {time.time() - start:.2f}s.")

    def run(self) -> Chromosome:
        """Evolve the population, applying dynamic‑stopping logic if enabled."""
        start = time.time()
        self._log("Starting evolution…", divider=True)
        # log the config
        self._log(f"Config: {config.__dict__}", divider=True)
        self.generate_population()

        planned_generations = self.num_generations
        
        if planned_generations <= 0:
            raise ValueError("num_generations must be at least 1")
        if self.max_generations is not None and self.max_generations < planned_generations:
            raise ValueError("max_generations cannot be smaller than num_generations")

        while self.generation < planned_generations:
            print (f"Generation {self.generation + 1} of {planned_generations}...")
            self.next_generation()

            # this should never happen in theory because we always have at least one working saved
            # otherwise it immediately fails
            if not self.population:
                self._log("TPK :( Stopping evolution.", divider=True)
                return None

            print(f"Generation {self.generation} completed.")
            print(f"Population size: {len(self.population)}")
            # ------------------ EARLY‑STOPPING -------------------
            if (
                self.enable_dynamic_stopping
                and self.generation >= self.early_stop_window
            ):
                recent_delta = self.delta_best[-self.early_stop_window:]
                if all(abs(d) < self.early_stop_tolerance for d in recent_delta):
                    self._log(
                        "Early‑stopping: fitness plateau for "
                        f"{self.early_stop_window} generations.",
                        divider=True,
                    )
                    self.early_stopped = True
                    break


            print(f"Generation {self.generation} fitness: {self._get_elite()[0].fitness:.4f}")
            # ----------------‑ DYNAMIC EXTENSION ---------------‑
            if (
                self.enable_extension
                and self.generation >= planned_generations - self.extend_window
            ):
                recent_delta = self.delta_best[-self.extend_window:]
                if any(d > self.extend_threshold for d in recent_delta):
                    # still improving → extend
                    extra = self.extend_window
                    if (
                        self.max_generations is not None
                        and planned_generations + extra > self.max_generations
                    ):
                        extra = self.max_generations - planned_generations
                    if extra > 0:
                        planned_generations += extra
                        self.extended_generations += extra
                        self._log(
                            f"Run extended by {extra} generations (new target: "
                            f"{planned_generations}).",
                            divider=True,
                        )

            # max gen clamp
            if self.max_generations is not None and planned_generations > self.max_generations:
                planned_generations = self.max_generations

        self._log("Evolution completed.", divider=True)
        end = time.time()
        self._log(f"Total time: {end - start:.2f} seconds")

        if config.SAVE_LOGS:
            self._flush_log()
            # Save design parameter changes
            self.save_design_param_changes()

        return self._get_elite()[0]

    def _flush_log(self) -> None:
        """Write accumulated log lines to disk (append or create)."""
        with self.log_path.open("w", encoding="utf-8") as f:
            for line in self.log_lines:
                f.write(line + "\n")

    
    def _flush_metrics(self) -> None:
        """Thread-safe flush of buffered metrics to CSV."""
        self._log("Flushing metrics to CSV…")
        if not self._metrics_buffer:
            return

        # 1. Build a superset header
        all_keys: set[str] = set().union(*self._metrics_buffer)
        if self._csv_header_done:
            # include any new keys discovered since the first header was written
            all_keys.update(self._csv_fieldnames)
        self._csv_fieldnames = sorted(all_keys)      # stable column order

        mode = "a" if self._csv_header_done else "w"
        with self.csv_path.open(mode, newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self._csv_fieldnames,
                extrasaction="ignore",   # 2. never raise on unseen keys
            )
            if not self._csv_header_done:
                writer.writeheader()
                self._csv_header_done = True
            writer.writerows(self._metrics_buffer)

        self._metrics_buffer.clear()


    def update_plots(self) -> None:
        """
        Overwrite two PNGs in plots/:
        - fitness_curves: avg_off, avg_mut, avg_rand, best_fit vs generation
        - mean_gains:      mean_offspring_gain, mean_mutant_gain vs generation
        """
        if not self._all_metrics:
            return

        import pandas as pd
        import matplotlib.pyplot as plt

        self._log("Updating plots…")

        df = pd.DataFrame(self._all_metrics)

        self._log(f"Plotting {len(df)} generations of metrics.")
        # ——— Plot 1: fitness curves ———
        plt.figure(figsize=(8, 5))
        for col in ['avg_off', 'avg_mut', 'avg_rand', 'best_fit']:
            plt.plot(df['generation'], df[col], label=col)
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Average Fitness Over Generations")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(self.plot_path)
        plt.close()

        self._log(f"Fitness curves plot saved to {self.plot_path}")

        # ——— Plot 2: mean gains ———
        plt.figure(figsize=(8, 5))
        for col in ['mean_offspring_gain', 'mean_mutant_gain']:
            plt.plot(df['generation'], df[col], marker='o', linestyle='--', label=col)
        plt.xlabel("Generation")
        plt.ylabel("Mean Gain")
        plt.title("Mean Gains Over Generations")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(self.gain_plot_path)
        plt.close()

        self._log(f"Mean gains plot saved to {self.gain_plot_path}")

        # ——— Plot for delta best (improvement in best fitness) ———
        plt.figure(figsize=(8, 5))
        plt.plot(df['generation'], df['delta_best'], marker='o', color='green', linestyle='-', label='Delta Best')
        plt.xlabel("Generation")
        plt.ylabel("Improvement in Best Fitness")
        plt.title("Best Fitness Improvement per Generation")
        plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(self.delta_best_plot_path)
        plt.close()

        self._log(f"Delta best plot saved to {self.delta_best_plot_path}")

        # ——— Plot 3: mutation gains (only if there are any mutants) ———
        # Note: In your case, self.n_mutants == 0, so this block is effectively skipped.
        if self.n_mutants != 0:
            mut_cols = [c for c in df.columns if c.startswith("mut_gain_")]
            if mut_cols:
                plt.figure(figsize=(8, 5))
                for col in mut_cols:
                    plt.plot(df['generation'], df[col], marker='o', linestyle='-', label=col.removeprefix("mut_gain_"))
                plt.xlabel("Generation")
                plt.ylabel("Δ Fitness vs parent")
                plt.title("Average Fitness Gain per Mutation Operator")
                plt.axhline(0, linewidth=1)
                plt.legend(loc="best")
                plt.tight_layout()
                plt.savefig(self.mut_plot_path)
                plt.close()

            self._log(f"Mutation gains plot saved to {self.mut_plot_path}")

        # ——— Swarm plot: fitness points per generation ———
        # Ensure we always create `fig, ax` before calling swarmplot.
        fig, ax = plt.subplots(figsize=(10, 6))

        if self._swarm_df.empty:
            self._log("Swarm plot data is empty; skipping swarm plot.")
            return

        # Prepare the data
        df_swarm = self._swarm_df.copy()
        df_swarm["generation"] = pd.to_numeric(df_swarm["generation"], errors="coerce")
        df_swarm["fitness"]    = pd.to_numeric(df_swarm["fitness"], errors="coerce")
        df_swarm = df_swarm.dropna(subset=["generation", "fitness", "origin"])

        self._log(f"Swarm plot data: {len(df_swarm)} points")

        # Draw the swarmplot on the (now guaranteed) `ax`
        sns.swarmplot(
            data=df_swarm,
            x="generation",
            y="fitness",
            hue="origin",
            ax=ax,
            size=3,
            alpha=0.6,
            dodge=True   # separates the hue levels slightly
        )

        self._log("Swarm plot created.")

        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.set_title("Fitness swarm plot (points jittered), colored by origin")
        ax.legend(title="Origin", loc="upper left", bbox_to_anchor=(1, 1))
        fig.tight_layout()
        fig.savefig(self.swarm_plot_path)
        plt.close(fig)

        self._log(f"Swarm plot saved to {self.swarm_plot_path}")
        self._log("Plots updated successfully.")

    def _track_design_param_changes(self, generation: int, population: list[Chromosome]) -> None:
        """
        Track and log design parameter changes in the population.
        Compares design params of all chromosomes to detect parameter mutations.
        Enhanced to check ALL chromosomes, not just those marked with design_param mutation.
        """
        if not hasattr(self, 'design_param_changes'):
            self.design_param_changes = []
        
        # Skip if design params aren't being used
        if self.design_params is None:
            self._log(f"[Generation {generation}] No original design parameters available for comparison")
            return
            
        # First, check specifically for chromosomes with design_param as their last mutation
        mutated_chroms = [c for c in population if c.last_mutation == "design_param"]
        self._log(f"[Generation {generation}] Found {len(mutated_chroms)} chromosomes with design_param as last mutation")
        
        # Now check ALL chromosomes for any design parameter differences
        mutated_count = 0
        
        for idx, chrom in enumerate(population):
            if chrom.design_params is None:
                if config.VERBOSE:
                    self._log(f"  Chromosome {idx}: No design parameters available")
                continue
                
            # Use our comparison function
            diffs = self._compare_design_params(self.design_params, chrom.design_params)
            
            if diffs:
                mutated_count += 1
                mutation_type = "design_param" if chrom.last_mutation == "design_param" else chrom.last_mutation or "unknown"
                self._log(f"  Chromosome {idx} (origin: {chrom.origin}, last mutation: {mutation_type}): {len(diffs)} parameter differences")
                
                for path, old_val, new_val in diffs:
                    self._log(f"    Parameter '{path}' changed: {old_val} -> {new_val}")
                    self.design_param_changes.append({
                        'generation': generation,
                        'chromosome': idx,
                        'chromosome_origin': chrom.origin,
                        'last_mutation': chrom.last_mutation,
                        'parameter': path,
                        'old_value': old_val,
                        'new_value': new_val,
                        'fitness': chrom.fitness
                    })
        
        if mutated_count > 0:
            self._log(f"[Generation {generation}] Total of {mutated_count} chromosomes with design parameter differences")
    
    def save_design_param_changes(self):
        """Save design parameter changes to a CSV file with enhanced details."""
        if not hasattr(self, 'design_param_changes') or not self.design_param_changes:
            self._log("No design parameter changes to save.")
            return
            
        import pandas as pd
        from datetime import datetime
        
        # Create a DataFrame from the changes
        df = pd.DataFrame(self.design_param_changes)
        
        # Save to CSV with timestamp in the directory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(self.log_path).parent
        csv_path = log_dir / f"design_param_changes.csv"
        df.to_csv(csv_path, index=False)
        
        self._log(f"Saved {len(self.design_param_changes)} design parameter changes to {csv_path}")
        
        # Print comprehensive summary
        param_counts = df['parameter'].value_counts()
        self._log("Parameter change frequency:")
        for param, count in param_counts.items():
            self._log(f"  {param}: {count} times")
        
        gen_counts = df['generation'].value_counts().sort_index()
        self._log("Changes per generation:")
        for gen, count in gen_counts.items():
            self._log(f"  Generation {gen}: {count} changes")
            
        if 'chromosome_origin' in df.columns:
            origin_counts = df['chromosome_origin'].value_counts()
            self._log("Changes by chromosome origin:")
            for origin, count in origin_counts.items():
                self._log(f"  {origin}: {count} changes")
                
        if 'last_mutation' in df.columns:
            mutation_counts = df['last_mutation'].value_counts()
            self._log("Changes by last mutation type:")
            for mutation, count in mutation_counts.items():
                self._log(f"  {mutation}: {count} changes")
                
        # Create parameter change summary plot
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plots_dir = log_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            # Plot changes per generation
            plt.figure(figsize=(10, 6))
            ax = sns.countplot(x='generation', data=df)
            plt.title('Design Parameter Changes per Generation')
            plt.xlabel('Generation')
            plt.ylabel('Number of Changes')
            plt.tight_layout()
            plt.savefig(plots_dir / "design_param_changes_by_gen.png")
            plt.close()
            
            # Plot changes by parameter
            if len(param_counts) > 0:
                plt.figure(figsize=(12, 8))
                param_df = pd.DataFrame({'parameter': param_counts.index, 'count': param_counts.values})
                param_df = param_df.sort_values('count', ascending=False)
                ax = sns.barplot(x='count', y='parameter', data=param_df)
                plt.title('Design Parameter Change Frequency')
                plt.xlabel('Number of Changes')
                plt.ylabel('Parameter')
                plt.tight_layout()
                plt.savefig(plots_dir / "design_param_changes_by_param.png")
                plt.close()
                
            self._log(f"Design parameter change plots saved to {plots_dir}")
        except Exception as e:
            self._log(f"Error creating parameter change plots: {e}")

    def output_best_result(self):
        """
        Output the best result from the evolution process.
        Compares the best chromosome's design parameters to the original parameters
        and outputs a summary of all changed parameters.
        """
        self._log("Generating report of best result...", divider=True)
        
        # Find the best chromosome
        best_chrom = max(self.population, key=lambda c: c.fitness)
        self._log(f"Best fitness: {best_chrom.fitness:.6f}")
        
        # Only proceed if we have design parameters
        if not self.design_params or not best_chrom.design_params:
            self._log("No design parameters to compare")
            return
        
        # Compare design parameters
        changed_params = []
        
        def compare_params(orig_params, new_params, path=[]):
            """Compare original and new parameters recursively"""
            if isinstance(orig_params, dict) and isinstance(new_params, dict):
                for key in set(orig_params.keys()) | set(new_params.keys()):
                    if key in orig_params and key in new_params:
                        # Both have the key, compare values
                        if key == 'v':  # This is a value field
                            orig_v = orig_params[key]
                            new_v = new_params[key]
                            if orig_v != new_v:
                                full_path = '.'.join(path)
                                changed_params.append({
                                    'parameter': full_path,
                                    'original_value': orig_v,
                                    'best_value': new_v
                                })
                        elif key != 'range':  # Skip comparing range fields
                            compare_params(orig_params[key], new_params[key], path + [key])
        
        # Start comparison at the root
        compare_params(self.design_params, best_chrom.design_params)
        
        # Output changed parameters
        if changed_params:
            self._log(f"Found {len(changed_params)} changed parameters in the best result:")
            
            # Write to CSV
            report_path = Path(self.log_path).parent / "best_result_changes.csv"
            with report_path.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['parameter', 'original_value', 'best_value'])
                writer.writeheader()
                writer.writerows(changed_params)
            
            # Log the changes
            for change in changed_params:
                self._log(f"  {change['parameter']}: {change['original_value']} -> {change['best_value']}")
            
            self._log(f"Full report saved to {report_path}")
        else:
            self._log("No parameter changes in the best result")

    def _flatten_params(self, params, prefix=None):
        """Recursively yield parameter paths as tuples of keys."""
        if prefix is None:
            prefix = []

        paths = []

        if isinstance(params, dict):
            for key, value in params.items():
                new_prefix = prefix + [str(key)]
                if isinstance(value, (dict, list)):
                    paths.extend(self._flatten_params(value, new_prefix))
                else:
                    paths.append(tuple(new_prefix))

        elif isinstance(params, list):
            for idx, value in enumerate(params):
                new_prefix = prefix + [str(idx)]
                if isinstance(value, (dict, list)):
                    paths.extend(self._flatten_params(value, new_prefix))
                else:
                    paths.append(tuple(new_prefix))

        else:
            if prefix:
                paths.append(tuple(prefix))

        return paths

    def _get_param_value(self, params, path):
        """Return value from nested dict/list using a tuple or dotted path."""
        if isinstance(path, str):
            path = path.split(".")

        cur = params
        for key in path:
            if isinstance(cur, dict):
                cur = cur.get(key)
            elif isinstance(cur, list):
                try:
                    cur = cur[int(key)]
                except (ValueError, IndexError):
                    return None
            else:
                return None
        return cur

    def _compare_design_params(self, params1, params2):
        """
        Compare two design parameter dictionaries and return a list of differences.
        Returns a list of tuples (param_path, value1, value2) for parameters that differ.
        """
        if params1 is None or params2 is None:
            return []
            
        differences = []
        
        # Get all param paths and values from both dictionaries
        flattened1 = dict(self._flatten_params(params1))
        flattened2 = dict(self._flatten_params(params2))
        
        # Get all unique paths
        all_paths = set(flattened1.keys()).union(set(flattened2.keys()))
        
        for path in all_paths:
            val1 = flattened1.get(path)
            val2 = flattened2.get(path)
            
            # Compare values (handling missing keys)
            if val1 != val2:
                differences.append((path, val1, val2))
                
        return differences
        
    def _print_param_comparison(self, gen_num, pop_idx, chrom, orig_params):
        """Print a comparison of chromosome design params vs original params"""
        if chrom.design_params is None:
            self._log(f"[Generation {gen_num}] Chromosome {pop_idx}: No design parameters")
            return
            
        diffs = self._compare_design_params(orig_params, chrom.design_params)
        
        if not diffs:
            if config.VERBOSE:
                self._log(f"[Generation {gen_num}] Chromosome {pop_idx}: Design parameters identical to original")
            return
            
        self._log(f"[Generation {gen_num}] Chromosome {pop_idx}: Found {len(diffs)} parameter differences")
        for path, orig_val, new_val in diffs:
            self._log(f"  Parameter '{path}': {orig_val} -> {new_val}")
    
    def _flatten_params(self, params_dict, prefix=''):
        """
        Flatten a nested dictionary of parameters into a list of (key, value) tuples.
        This is used for counting and tracking design parameters.
        
        Args:
            params_dict: Dictionary of parameters, potentially nested
            prefix: Prefix to add to keys from upper levels of nesting
            
        Returns:
            List of (key, value) tuples
        """
        result = []
        if isinstance(params_dict, dict):
            for key, value in params_dict.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, (dict, list)):
                    result.extend(self._flatten_params(value, full_key))
                else:
                    result.append((full_key, value))
        elif isinstance(params_dict, list):
            for i, item in enumerate(params_dict):
                full_key = f"{prefix}.{i}"
                if isinstance(item, (dict, list)):
                    result.extend(self._flatten_params(item, full_key))
                else:
                    result.append((full_key, item))
        return result

