from __future__ import annotations
import copy
import csv
import multiprocessing
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import seaborn as sns
import threading

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .layout import Piece, Container, LayoutView
from .placement_engine import DECODER_REGISTRY
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
        # crossover_method: str = "oxk",
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
        pattern_name: str = None,
    ) -> None:
        self.generation = 0
        self.container = container
        self.pieces = pieces
        self.population: list[Chromosome] = []
        self.pattern_name = pattern_name or ""
        
        # Initialize logging first to avoid attribute errors
        self.log_lines = []
        
        # Setup log files and directories
        ts = time.strftime("%Y%m%d_%H%M%S")
        log_dir = config.SAVE_LOGS_PATH
        
        # Use pattern name in the log directory path
        if self.pattern_name:
            log_dir = Path(log_dir) / self.pattern_name
        
        # append timestamp to log path
        log_dir = f"{log_dir}_{ts}"
        if config.SAVE_LOGS:
            Path(log_dir).mkdir(parents=True, exist_ok=True)        # ensure folder exists
        
        self.log_path = Path(log_dir) / f"evolution_log_{ts}.txt"
        self.csv_path = Path(log_dir) / f"evolution_metrics_{ts}.csv"
        
        plots_dir = Path(log_dir) / "plots"
        if config.SAVE_LOGS:
            plots_dir.mkdir(exist_ok=True)

        self.plot_path = plots_dir / f"fitness_curves_{ts}.png"
        self.gain_plot_path = plots_dir / f"mean_gains_{ts}.png"
        self.swarm_plot_path = plots_dir / f"fitness_swarm_{ts}.png"
        self.delta_best_plot_path = plots_dir / f"delta_best_{ts}.png"
        # Removed mut_plot_path as we no longer create the average gain per mutation plot

        # Folder to store per-generation SVGs of the best layout
        self.svg_dir = Path(log_dir) / "svgs"
        if config.SAVE_GENERATION_SVGS and config.SAVE_LOGS:
            self.svg_dir.mkdir(exist_ok=True)

        # Create a new property to store all mutation gains for swarm plotting
        self._mutation_swarm_data = pd.DataFrame(columns=["mutation_type", "fitness_gain", "generation"])
        
        # Create a new path for the mutation swarm plot
        self.mut_swarm_plot_path = plots_dir / "mutation_gains_swarm.png"

        # Create a violin plot path as well
        self.mut_violin_plot_path = plots_dir / "mutation_gains_violin.png"

        # Metrics/CSV state and swarm-plot buffer
        self._metrics_buffer = []
        self._all_metrics = []
        self._csv_header_done = False
        self._swarm_df = pd.DataFrame(columns=["generation", "fitness", "origin"])

        # Thread-safe logging
        self._log_lock = threading.Lock()

        # GA parameters
        self.num_generations = num_generations
        self.population_size = population_size
        #self.elite_population_size = elite_population_size
        self.mutation_rate = mutation_rate
        # self.crossover_method = crossover_method
        #self.pmx = pmx
        # self.allow_duplicate_genes = allow_duplicate_genes
        # self.max_duplicate_retries = max_duplicate_retries
        
        # Design parameters
        self.design_params = design_params
        self.body_params = body_params
        # self.design_sampler = design_sampler

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
        self.improvement_from_initial: list[float] = []  # New: track improvement from generation 0
        self.pop_fitness_history: list[list[float]] = []
        self.mean_offspring_gain: list[float] = []
        self.mean_mutant_gain: list[float] = []
        # Removed mutation_perf as we no longer track per-mutation averages
        
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
            
        # if self.design_sampler is None:
        #     self._log("WARNING: No design sampler provided to Evolution constructor")
    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------

    def _log(self, msg: str = "", *, divider: bool = False) -> None:
        lines_to_write = []
        if divider:
            line = "-" * 60
            print(line)
            self.log_lines.append(line)
            lines_to_write.append(line)
        if msg:
            print(msg)
            self.log_lines.append(msg)
            lines_to_write.append(msg)

        # Continuous flush to disk for each log call
        if config.SAVE_LOGS and lines_to_write:
            with self._log_lock:
                with self.log_path.open("a", encoding="utf-8") as f:
                    for ln in lines_to_write:
                        f.write(ln + "\n")

    # ------------------------------------------------------------------
    # Population helpers
    # ------------------------------------------------------------------

    def _generate_random_chromosome(self) -> Chromosome:
        
        #self._log("Generating random chromosome...")
        ids_ = list(self.pieces.keys())
        random.shuffle(ids_)
        
        # Ensure we always pass a deep copy of design_params
        design_params_copy = copy.deepcopy(self.design_params) if self.design_params else None
        
        chrom = Chromosome(
            [self.pieces[i] for i in ids_], 
            self.container,
            origin="random",
            design_params=design_params_copy, 
            body_params=self.body_params
        )

        # ramdomize rotations
        for piece in chrom.genes:
            rotation = random.choice(config.ALLOWED_ROTATIONS)
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

        # Print debug information about chromosomes
        # if config.VERBOSE:
        #     self._log("DEBUG: Checking design_params uniqueness in population:")
        #     dp_ids = [id(c.design_params) for c in self.population]
        #     unique_ids = set(dp_ids)
        #     self._log(f"Population size: {len(self.population)}, Unique design_params objects: {len(unique_ids)}")
        #     if len(unique_ids) != len(self.population):
        #         self._log("WARNING: Some chromosomes share the same design_params object!")
        #         # Find which chromosomes share design_params
        #         id_counts = {}
        #         for i, dp_id in enumerate(dp_ids):
        #             id_counts.setdefault(dp_id, []).append(i)
        #         for dp_id, indices in id_counts.items():
        #             if len(indices) > 1:
        #                 self._log(f"design_params object {dp_id} is shared by chromosomes at indices: {indices}")
        
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
            'improvement_from_initial': 0.0,  # Always 0 for generation 0
            'mean_offspring_gain': 0.0,
            'mean_mutant_gain': 0.0,
        }
        self._all_metrics.append(row0)
        self._metrics_buffer.append(row0)
        # Seed best and delta history
        self.best_fitness_history.append(best0)
        self.delta_best.append(0.0)
        self.improvement_from_initial.append(0.0)  # Always 0 for generation 0
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


    
    # def _generate_offspring(self, old_pop: list[Chromosome]) -> Chromosome:
    #     """
    #     Create one child chromosome by crossover (and optional mutation) and
    #     return it.  Duplicate-avoidance is honoured unless
    #     self.allow_duplicate_genes is True.
    #     """
    #     start   = time.time()
    #     #retries = 0

    #     p1, p2 = random.sample(old_pop, 2)
        
    #     # Ensure parents have their own deep copies of design_params
    #     if p1.design_params is not None:
    #         p1.design_params = copy.deepcopy(p1.design_params)
    #     if p2.design_params is not None:
    #         p2.design_params = copy.deepcopy(p2.design_params)

    #     # --- crossover ---
    #     # child = p1.crossover_pmx(p2) if self.pmx else p1.crossover_ox1_k(p2)

    #     if self.crossover_method == "pmx":
    #         child = p1.crossover_pmx(p2)
    #     elif self.crossover_method == "ox1":
    #         child = p1.crossover_ox1(p2, config.OX_K)
    #     else:
    #         raise ValueError(f"Unknown crossover method: {self.crossover_method}")

    #     # --- mutation (optional) ---
    #     if random.random() < self.mutation_rate:
    #         child.mutate()

    #     # --- fitness evaluation ---
    #     # use ProcessPoolExecutor to parallelize fitness evaluation
        
    #     child.calculate_fitness()
        
    #     # Debug - verify the child has unique design_params
    #     if config.VERBOSE and p1.design_params is not None and child.design_params is not None:
    #         p1_id = id(p1.design_params)
    #         p2_id = id(p2.design_params) if p2.design_params is not None else None
    #         child_id = id(child.design_params)
            
    #         if child_id == p1_id or child_id == p2_id:
    #             print(f"WARNING: Child chromosome has the same design_params object as a parent!")
    #             print(f"  Parent 1 design_params ID: {p1_id}")
    #             print(f"  Parent 2 design_params ID: {p2_id}")
    #             print(f"  Child design_params ID: {child_id}")
                
    #     #self._log(f"Offspring generated in {time.time() - start:.2f} s")
    #     return child


    
    def next_generation(self) -> None:
        self._log("Next generation…", divider=True)
        start = time.time()

        old_elite = self._get_elite()
        if not old_elite:
            self._log("No elite chromosomes found. Cannot create next generation.", divider=True)
            return
            
        # Check if we have enough elite chromosomes for meaningful evolution
        if len(old_elite) <= 1:
            self._log("Only 1 elite chromosome found. Not enough for crossover operations.", divider=True)
            self._log("Keeping current population and adding random chromosomes.")
            # Add some random chromosomes to avoid stagnation
            for _ in range(self.n_offspring + self.n_mutants):
                random_chrom = self._generate_random_chromosome()
                self.population.append(random_chrom)
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
        self._log("Phase A: creating offspring…")

        def do_offspring(p1, p2, parent_f):
            """Produce one mating -> 1 or 2 children depending on config flag."""
            try:
                # Use cross-stitch crossover with DP-closure and split-root handling
                if (config.SELECTED_CROSSOVER == "cross_stitch_oxk"):
                    res = p1.cross_stitch_oxk(p2, k=config.OX_K, mode=config.CROSS_STITCH_MODE)
                elif (config.SELECTED_CROSSOVER == "oxk"):
                    res = p1.crossover_oxk(p2, k=config.OX_K)
                children = []
                if isinstance(res, tuple) or isinstance(res, list):
                    children = [c for c in res if c is not None]
                else:
                    children = [res]

                out_children = []
                for child in children:
                    if random.random() < self.mutation_rate:
                        child.mutate()
                    child.calculate_fitness()
                    if config.VERBOSE:
                        self._log(f"Offspring created with fitness: {child.fitness:.4f}")
                    child.origin = "offspring"
                    out_children.append(child)
                return out_children, parent_f, None
            except Exception as e:
                return [], parent_f, e

        offspring, off_gains = [], []
        
        # Filter out parents with 0 fitness
        viable_parents = [p for p in old_elite if p.fitness > 0]
        if len(viable_parents) < 2:
            # Not enough viable parents, use all elite chromosomes
            self._log("Warning: Not enough parents with non-zero fitness. Using all elite chromosomes.")
            viable_parents = old_elite
            
        # Select two different parents
        p1, p2 = random.sample(viable_parents, 2)
        parent_f = max(p1.fitness, p2.fitness)

        # Create offspring using cross-stitch crossover
        target_offspring = self.n_offspring
        if config.MULTITHREADING:
            offspring_jobs = {}
            with ThreadPoolExecutor() as ex:
                # Submit one mating per job; each job may return up to 2 kids
                for _ in range(max(1, (target_offspring + 1) // 2)):
                    fut = ex.submit(do_offspring, p1, p2, parent_f)
                    offspring_jobs[fut] = None
            for fut in as_completed(offspring_jobs):
                kids, parent_f, err = fut.result()
                if err:
                    self._log(f"offspring task failed: {err}")
                    continue
                for child in kids:
                    if len(offspring) >= target_offspring:
                        break
                    gain = child.fitness - parent_f
                    offspring.append(child)
                    off_gains.append(gain)
                    new_population.append(child)
            # If still short (e.g., single-child mode), top up with extra matings
            while len(offspring) < target_offspring:
                kids, parent_f, err = do_offspring(p1, p2, parent_f)
                if err:
                    self._log(f"offspring task failed: {err}")
                    break
                for child in kids:
                    if len(offspring) >= target_offspring:
                        break
                    gain = child.fitness - parent_f
                    offspring.append(child)
                    off_gains.append(gain)
                    new_population.append(child)
        else:  # single-threaded fallback
            while len(offspring) < target_offspring:
                kids, parent_f, err = do_offspring(p1, p2, parent_f)
                if err:
                    self._log(f"offspring task failed: {err}")
                    continue
                for child in kids:
                    if len(offspring) >= target_offspring:
                        break
                    gain = child.fitness - parent_f
                    offspring.append(child)
                    off_gains.append(gain)
                    new_population.append(child)

        # ────────────────
        # Phase B: Mutants
        # ────────────────
        self._log("Phase B: creating mutants…")

        mutants, mut_gains = [], []

        def do_mutant(parent):
            try:
                # Create a deep copy of the parent and ensure design_params is deeply copied
                parent_f = parent.fitness
                child = copy.deepcopy(parent)
                if child.design_params is not None:
                    child.design_params = copy.deepcopy(parent.design_params)
                # mutate the child until child.mutate() returns True
                mutated = child.mutate()
                while not mutated:
                    mutated = child.mutate()
                child.calculate_fitness()
                if config.VERBOSE:
                    self._log(f"Mutant created with fitness: {child.fitness:.4f}")
                child.origin = "mutant"
                return child, parent_f, child.last_mutation, None
            except Exception as e:
                return None, parent_f, None, e
                    

        mutants, mut_gains = [], []
        # Collect rows for mutation swarm data (to avoid concurrent writes)
        mut_swarm_rows: list[dict] = []
        if config.MULTITHREADING:
            mutant_jobs = {}
            with ThreadPoolExecutor() as ex:
                for _ in range(self.n_mutants):
                    par = random.choice(viable_parents)
                    fut = ex.submit(do_mutant, par)
                    mutant_jobs[fut] = None
            for fut in as_completed(mutant_jobs):
                mutant, parent_f, mut_type, err = fut.result()
                if err:
                    self._log(f"‼ mutant task failed: {err}")
                    continue
                mutants.append(mutant)
                gain = mutant.fitness - parent_f
                mut_gains.append(gain)
                new_population.append(mutant)
                # Stage row for mutation swarm data
                if mut_type is not None:
                    mut_swarm_rows.append({
                        "mutation_type": mut_type,
                        "fitness_gain": gain,
                        "generation": self.generation + 1  # next gen index
                    })
        else:
            for _ in range(self.n_mutants):
                par = random.choice(viable_parents)
                mutant, parent_f, mut_type, err = do_mutant(par)
                if err:
                    self._log(f"‼ mutant task failed: {err}")
                    continue
                mutants.append(mutant)
                gain = mutant.fitness - parent_f
                mut_gains.append(gain)
                new_population.append(mutant)
                if mut_type is not None:
                    mut_swarm_rows.append({
                        "mutation_type": mut_type,
                        "fitness_gain": gain,
                        "generation": self.generation + 1
                    })

        # ───────────────────
        # Phase C: Randoms
        # ───────────────────
        self._log("Phase C: creating random chromosomes…")

        randoms = []

        def do_random():
                    try:
                        child = self._generate_random_chromosome()
                        if config.VERBOSE:
                            self._log(f"Random chromosome created with fitness: {child.fitness:.4f}")
                        child.origin = "random"
                        return child, 0.0, None, None
                    except Exception as e:
                        return None, 0.0, None, e

        if config.MULTITHREADING:
            random_jobs = {}
            with ThreadPoolExecutor() as ex:
                for _ in range(self.n_randoms):
                    fut = ex.submit(do_random)                  # unchanged helper
                    random_jobs[fut] = None
            for fut in as_completed(random_jobs):
                child, _, _, err = fut.result()
                if err:
                    self._log(f"‼ random task failed: {err}")
                    continue
                randoms.append(child)
                new_population.append(child)
        else:  # ───────────── single-threaded fallback ─────────────
            for _ in range(self.n_randoms):
                child, _, _, err = do_random()
                if err:
                    self._log(f"‼ random task failed: {err}")
                    continue
                randoms.append(child)
                new_population.append(child)

        # Summary of collected results
        self._log(
            f"Collected {len(offspring)} offspring, "
            f"{len(mutants)} mutants, {len(randoms)} randoms."
        )

        # Compute per‐operator statistics
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

        # Append any collected mutation swarm rows
        if mut_swarm_rows:
            try:
                self._mutation_swarm_data = pd.concat(
                    [self._mutation_swarm_data, pd.DataFrame(mut_swarm_rows)],
                    ignore_index=True
                )
            except Exception as e:
                self._log(f"Failed appending mutation swarm rows: {e}")

        frame = pd.DataFrame([{
            "generation": self.generation,
            "fitness":   c.fitness,
            "origin":    c.origin or "unknown"
        } for c in self.population])
        self._swarm_df = pd.concat([self._swarm_df, frame], ignore_index=True)

        best = max(c.fitness for c in new_population)
        delta = best - (self.best_fitness_history[-1] if self.best_fitness_history else best)
        
        # Calculate improvement from initial fitness (generation 0)
        initial_fitness = self.best_fitness_history[0] if self.best_fitness_history else best
        improvement_from_initial = best - initial_fitness
        
        self.best_fitness_history.append(best)
        self.delta_best.append(delta)
        self.improvement_from_initial.append(improvement_from_initial)
        self._log(f"Generation {self.generation}: best {best:.4f} (Δ {delta:+.4f}) (from initial: {improvement_from_initial:+.4f})", divider=True)

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
            'improvement_from_initial': improvement_from_initial,
            'mean_offspring_gain': mean_offspring_gain,
            'mean_mutant_gain': mean_mutant_gain,
        }
        
        # No need to collect per-mutation-operator average gains anymore
        # Since we're only using the individual data points for swarm/violin plots
        
        # Track design parameter changes
        self._track_design_param_changes(self.generation, new_population)

        self._metrics_buffer.append(row)
        self._all_metrics.append(row)

        # Save SVG of best layout for this generation
        best_chrom = max(new_population, key=lambda c: c.fitness)
        self._save_generation_svg(best_chrom, self.generation)

        flush_metrics = (
            len(self._metrics_buffer) >= config.GENERATION_PER_FLUSH
            or self.generation == self.num_generations
        )
        if flush_metrics and config.SAVE_LOGS:
            self._flush_metrics()  # also flushes logs
            self.update_plots()
        elif config.SAVE_LOGS and self.generation % config.LOG_FLUSH_INTERVAL == 0:
            self._flush_log()

        self._log(f"Generation {self.generation} completed in {time.time() - start:.2f}s.")

    def run(self) -> Chromosome:
        """Evolve the population, applying dynamic‑stopping logic if enabled."""
        start = time.time()
        self._log("Starting evolution…", divider=True)
        # log the config
        self._log(f"Config: {config.__dict__}", divider=True)
        self.generate_population()

        # Check if we have enough viable elites to run the GA
        old_elite = self._get_elite()
        if len(old_elite) <= 1:
            self._log("Insufficient viable chromosomes to run GA (need at least 2 elites).", divider=True)
            self._log("Returning best chromosome from initial population without running evolution.")
            return max(self.population, key=lambda c: c.fitness) if self.population else None

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
        elapsed_time = end - start
        self._log(f"Total time: {elapsed_time:.2f} seconds")

        # Save plots and CSV one last time
        self.update_plots()
        if config.SAVE_LOGS:
            # Output best result with all changed parameters
            self.output_best_result()
            # Flush the log to include the best result output
            self._flush_log()
            # Save design parameter changes
            self.save_design_param_changes()
        
        # Return the best chromosome from the population
        if self.population:
            best = max(self.population, key=lambda c: c.fitness)
            return best
        return None

    def _flush_log(self) -> None:
        """Write accumulated log lines to disk (append or create)."""
        # Ensure the log directory exists before writing
        if not self.log_path.parent.exists():
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            self.log_path.touch()

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

        # Ensure logs are persisted whenever metrics are flushed
        self._flush_log()


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

        # ——— Plot 3: delta best ———
        plt.figure(figsize=(8, 5))
        if 'delta_best' in df.columns:
            plt.plot(df['generation'], df['delta_best'], marker='o', color='green', linestyle='-', label='Delta Best')
            plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
            plt.xlabel("Generation")
            plt.ylabel("Improvement in Best Fitness")
            plt.title("Best Fitness Improvement per Generation")
            plt.legend(loc="best")
            plt.tight_layout()
            plt.savefig(self.delta_best_plot_path)
            plt.close()
            
            self._log(f"Delta best plot saved to {self.delta_best_plot_path}")
        else:
            self._log("Delta best data not available, skipping delta best plot")

        # ——— Plot 4: mutation gains (only if there are any mutants) ———
        # Note: In your case, self.n_mutants == 0, so this block is effectively skipped.
        if self.n_mutants != 0:
            # Create the swarm and violin plots for mutation gains (skipping the average line plot)
            if not self._mutation_swarm_data.empty:
                plt.figure(figsize=(10, 6))
                # Create box plot to show distribution statistics
                ax = sns.boxplot(
                    data=self._mutation_swarm_data, 
                    x="mutation_type", 
                    y="fitness_gain",
                    color="lightgrey",
                    fliersize=0  # Don't show outlier points in the box plot
                )
                
                # Add swarm plot on top of the box plot
                sns.swarmplot(
                    data=self._mutation_swarm_data, 
                    x="mutation_type", 
                    y="fitness_gain", 
                    hue="generation",
                    palette="viridis",
                    ax=ax
                )
                
                # Add a horizontal line at y=0
                plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                
                # Improve the plot
                plt.title("Fitness Gain Distribution by Mutation Type")
                plt.xlabel("Mutation Type")
                plt.ylabel("Fitness Gain")
                plt.xticks(rotation=45)
                plt.legend(title="Generation", bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                
                # Save the plot
                plt.savefig(self.mut_swarm_plot_path)
                plt.close()
                self._log(f"Mutation gains swarm plot saved to {self.mut_swarm_plot_path}")
                
                # Create a violin plot as well for a different perspective on the distribution
                plt.figure(figsize=(10, 6))
                
                # Get the y-axis limits from the previous plot to ensure consistency
                y_min, y_max = ax.get_ylim()
                
                # Create violin plot
                ax_violin = sns.violinplot(
                    data=self._mutation_swarm_data, 
                    x="mutation_type", 
                    y="fitness_gain",
                    palette="Set3",
                    inner="stick",  # Show individual data points inside
                    cut=0  # Don't extend the violin past the observed data points
                )
                
                # Set the same y-limits as the swarm plot for consistency
                ax_violin.set_ylim(y_min, y_max)
                
                # Add a horizontal line at y=0
                plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                
                # Improve the plot
                plt.title("Fitness Gain Distribution by Mutation Type (Violin Plot)")
                plt.xlabel("Mutation Type")
                plt.ylabel("Fitness Gain")
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Save the plot
                plt.savefig(self.mut_violin_plot_path)
                plt.close()
                self._log(f"Mutation gains violin plot saved to {self.mut_violin_plot_path}")
        else:
            self._log("No mutation data available, skipping mutation gain plots")

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

        # ——— Mutation Swarm plot: fitness gains per mutation type ———
        if self.n_mutants > 0 and not self._mutation_swarm_data.empty:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Prepare the data
            df_mutation_swarm = self._mutation_swarm_data.copy()
            df_mutation_swarm["generation"] = pd.to_numeric(df_mutation_swarm["generation"], errors="coerce")
            df_mutation_swarm["fitness_gain"]    = pd.to_numeric(df_mutation_swarm["fitness_gain"], errors="coerce")
            df_mutation_swarm = df_mutation_swarm.dropna(subset=["generation", "fitness_gain", "mutation_type"])

            self._log(f"Mutation swarm plot data: {len(df_mutation_swarm)} points")

            # Draw the swarmplot on the (now guaranteed) `ax`
            sns.swarmplot(
                data=df_mutation_swarm,
                x="generation",
                y="fitness_gain",
                hue="mutation_type",
                ax=ax,
                size=3,
                alpha=0.6,
                dodge=True   # separates the hue levels slightly
            )

            self._log("Mutation swarm plot created.")

            ax.set_xlabel("Generation")
            ax.set_ylabel("Fitness Gain")
            ax.set_title("Fitness Gain Swarm Plot by Mutation Type")
            ax.legend(title="Mutation Type", loc="upper left", bbox_to_anchor=(1, 1))
            fig.tight_layout()
            fig.savefig(self.mut_swarm_plot_path)
            plt.close(fig)

            self._log(f"Mutation swarm plot saved to {self.mut_swarm_plot_path}")

    def _track_design_param_changes(self, generation: int, population: list[Chromosome]) -> None:
        """
        Track and log design parameter changes in the population.
        Only reports changes for chromosomes that have been modified in the current generation
        by using the param_changes_this_gen attribute set during mutation.
        """
        if not hasattr(self, 'design_param_changes'):
            self.design_param_changes = []
        
        # Skip if design params aren't being used
        if self.design_params is None:
            self._log(f"[Generation {generation}] No original design parameters available for comparison")
            return
            
        # Count chromosomes with recorded parameter changes in this generation
        mutated_count = 0
        
        for idx, chrom in enumerate(population):
            # Skip chromosomes with no design parameters
            if chrom.design_params is None:
                if config.VERBOSE:
                    self._log(f"  Chromosome {idx}: No design parameters available")
                continue
                
            # Skip chromosomes without recorded parameter changes from this generation
            if not hasattr(chrom, 'param_changes_this_gen') or not chrom.param_changes_this_gen:
                continue
                
            # Process recorded parameter changes for this chromosome
            mutated_count += 1
            mutation_type = chrom.last_mutation
            self._log(f"  Chromosome {idx} (origin: {chrom.origin}, last mutation: {mutation_type}): {len(chrom.param_changes_this_gen)} parameter differences")
            
            for change in chrom.param_changes_this_gen:
                path = change['param_path']
                old_val = change['old_value']
                new_val = change['new_value']
                
                self._log(f"    Parameter '{path}' changed: {old_val} -> {new_val}")
                self.design_param_changes.append({
                    'generation': generation,
                    'chromosome': idx,
                    'chromosome_origin': chrom.origin,
                    'last_mutation': mutation_type,
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
        
        # Calculate fitness improvement
        initial_fitness = self.best_fitness_history[0] if self.best_fitness_history else 0
        final_fitness = best_chrom.fitness
        improvement = final_fitness - initial_fitness
        improvement_percent = (improvement / initial_fitness * 100) if initial_fitness > 0 else 0
        
        self._log(f"Initial fitness: {initial_fitness:.6f}")
        self._log(f"Final fitness:   {final_fitness:.6f}")
        self._log(f"Improvement:     {improvement:.6f} ({improvement_percent:.2f}%)")
        
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
            
            # Import functions to identify affected panels
            from nesting.panel_mapping import affected_panels, select_genes
            
            # Get the original pieces for comparison
            from assets.garment_programs.meta_garment import MetaGarment
            from nesting.path_extractor import PatternPathExtractor
            import tempfile
            import json
            from pathlib import Path
            
            # Generate the original pattern with the initial parameters
            original_mg = MetaGarment("original", self.body_params, self.design_params)
            original_pattern = original_mg.assembly()
            original_pieces = {}
            
            with tempfile.TemporaryDirectory() as td:
                out_dir = Path(td)
                original_pattern.serialize(out_dir, to_subfolder=False, with_3d=False, with_text=False, view_ids=False)
                spec_path = out_dir / f"{original_pattern.name}_specification.json"
                extractor = PatternPathExtractor(spec_path)
                original_pieces = extractor.get_all_panel_pieces(samples_per_edge=config.SAMPLES_PER_EDGE)
            
            # Generate the best pattern with the final parameters
            best_mg = MetaGarment("best", self.body_params, best_chrom.design_params)
            best_pattern = best_mg.assembly()
            best_pieces = {}
            
            with tempfile.TemporaryDirectory() as td:
                out_dir = Path(td)
                best_pattern.serialize(out_dir, to_subfolder=False, with_3d=False, with_text=False, view_ids=False)
                spec_path = out_dir / f"{best_pattern.name}_specification.json"
                extractor = PatternPathExtractor(spec_path)
                best_pieces = extractor.get_all_panel_pieces(samples_per_edge=config.SAMPLES_PER_EDGE)
            
            # Enhance the changed parameters with affected pieces and their paths
            for change in changed_params:
                param_name = change['parameter']
                
                # Get the panel patterns affected by this parameter
                affected_patterns = affected_panels([param_name])
                
                # Get the affected piece IDs
                affected_ids = select_genes(best_pieces.keys(), affected_patterns)
                
                # Store affected pieces and their paths
                affected_pieces = []
                for piece_id in affected_ids:
                    if piece_id in original_pieces and piece_id in best_pieces:
                        affected_pieces.append({
                            'piece_id': piece_id,
                            'old_path': original_pieces[piece_id].outer_path,
                            'new_path': best_pieces[piece_id].outer_path
                        })
                
                # Add to the change record
                change['affected_pieces'] = affected_pieces
                change['affected_count'] = len(affected_pieces)
            
            # Write enhanced report to CSV
            report_path = Path(self.log_path).parent / "best_result_changes.csv"
            
            # First write the main CSV with parameter changes
            with report_path.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['parameter', 'original_value', 'best_value', 'affected_count'])
                writer.writeheader()
                for change in changed_params:
                    # Create a copy without the affected_pieces list for the main CSV
                    row = {k: v for k, v in change.items() if k != 'affected_pieces'}
                    writer.writerow(row)
            
            # Then write a detailed CSV with all affected pieces
            detailed_report_path = Path(self.log_path).parent / "best_result_paths.csv"
            with detailed_report_path.open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["parameter", "piece_id", "old_path", "new_path"])
                for change in changed_params:
                    param_name = change['parameter']
                    for piece in change.get('affected_pieces', []):
                        writer.writerow([
                            param_name,
                            piece['piece_id'],
                            json.dumps(piece['old_path']),
                            json.dumps(piece['new_path'])
                        ])
            
            # Log the changes
            for change in changed_params:
                self._log(f"  {change['parameter']}: {change['original_value']} -> {change['best_value']} (affects {change['affected_count']} pieces)")
            
            self._log(f"Parameter change report saved to {report_path}")
            self._log(f"Detailed path changes saved to {detailed_report_path}")
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

    def _record_design_param_change(self, param_name, old_value, new_value, mutation_type):
        """Record a design parameter change only if it affects pieces."""
        # Use the panel_mapping module to check if this parameter affects any pieces
        from nesting.panel_mapping import affected_panels, select_genes
        
        # Get the panel IDs from the current pieces
        panel_ids = {p.id for p in self.pieces.values()}
        
        # Check if the parameter affects any of the current panels
        affected_panel_patterns = affected_panels([param_name])
        affected_pieces = select_genes(panel_ids, affected_panel_patterns)
        
        # Only record if the parameter affects at least one piece
        if affected_pieces:
            self.design_param_changes.append({
                "param": param_name,
                "old_value": old_value,
                "new_value": new_value,
                "last_mutation": mutation_type,
                "affected_pieces": len(affected_pieces)
            })
            if config.VERBOSE:
                print(f"[Evolution] Recorded change to '{param_name}' affecting {len(affected_pieces)} pieces")
        elif config.VERBOSE:
            print(f"[Evolution] Skipped recording change to '{param_name}' (affects 0 pieces)")

    def _save_generation_svg(self, chrom: Chromosome, generation: int) -> None:
        """Render and save the best layout of the given generation as an SVG."""
        if not (config.SAVE_LOGS and config.SAVE_GENERATION_SVGS):
            return

        import copy
        import svgwrite

        view = LayoutView([copy.deepcopy(p) for p in chrom.genes])
        decoder = DECODER_REGISTRY[config.SELECTED_DECODER](view, self.container, step=config.GRAVITATE_STEP)
        decoder.decode()

        dwg = svgwrite.Drawing(
            filename=str(self.svg_dir / f"gen_{generation:04d}.svg"),
            size=(f"{self.container.width}cm", f"{self.container.height}cm"),
            viewBox=f"0 0 {self.container.width} {self.container.height}"
        )

        for piece in decoder.placed:
            pts = [(x + piece.translation[0], y + piece.translation[1]) for x, y in piece.get_outer_path()]
            dwg.add(dwg.polygon(points=pts, fill="none", stroke="black", stroke_width=0.5))

        dwg.save()

