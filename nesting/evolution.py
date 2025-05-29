from __future__ import annotations
import csv
import multiprocessing
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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
        elite_population_size: int = 5,
        mutation_rate: float = 0.2,
        crossover_method: str = config.SELECTED_CROSSOVER,
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
    ) -> None:
        self.generation = 0
        self.container = container
        self.pieces = pieces
        self.population: list[Chromosome] = []

        # GA parameters
        self.num_generations = num_generations
        self.population_size = population_size
        self.elite_population_size = elite_population_size
        self.mutation_rate = mutation_rate
        self.crossover_method = crossover_method
        #self.pmx = pmx
        # self.allow_duplicate_genes = allow_duplicate_genes
        # self.max_duplicate_retries = max_duplicate_retries

        # ── early stopping ───────────────────────────────────────────────
        self.enable_dynamic_stopping = enable_dynamic_stopping
        self.early_stop_window = early_stop_window
        self.early_stop_tolerance = early_stop_tolerance
        self.enable_extension = enable_extension
        self.extend_window = extend_window
        self.extend_threshold = extend_threshold
        self.max_generations = max_generations

        # run‑state flags
        self.early_stopped: bool = False
        self.extended_generations: int = 0


        # ── metrics ────────────────────────────────────────────
        self.survival_rates: list[float] = []
        self.avg_child_fitnesses: list[float] = []
        self.best_fitness_history: list[float] = []
        self.delta_best: list[float] = []
        self.child_parent_success_ratio: list[float] = []
        self.mean_crossover_gain: list[float] = []
        self.mean_mutation_gain: list[float] = []

        log_dir = "evo_runs"
        Path(log_dir).mkdir(parents=True, exist_ok=True)        # ensure folder exists
        ts            = time.strftime("%Y%m%d_%H%M%S")
        self.log_path = Path(log_dir) / f"evolution_log_{ts}.txt"
        self.csv_path   = Path(log_dir) / f"evolution_metrics_{ts}.csv"

        self.log_lines       : list[str]          = []
        self._metrics_buffer : list[dict[str, float]] = []
        self._csv_header_done: bool               = False

        self._log(
            f"Evolution instance created.",
            divider=True,
        )
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
        ids_ = list(self.pieces.keys())
        random.shuffle(ids_)
        chrom = Chromosome([self.pieces[i] for i in ids_], self.container)

        # ramdomize rotations
        for piece in chrom.genes:
            rotation = random.choice([0, 90, 180, 270])
            # rotation = 90
            if rotation != 0:
                piece.rotate(rotation)
        
        chrom.sync_order()
        chrom.calculate_fitness()
        return chrom

    def generate_population(self) -> None:
        # generate initial population in parallel
        self._log("Generating initial population...")
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._generate_random_chromosome) for _ in range(self.population_size)]
            self.population = [f.result() for f in futures]

        self._log(f"Initial population of {self.population_size} layouts generated.", divider=True)
        for i, chrom in enumerate(self.population):
            self._log(f"Layout {i}: {[(p.id, p.rotation) for p in chrom.genes]} | Fitness: {chrom.fitness:.4f}")
        self._log("Population generation completed.", divider=True)

        best = max(c.fitness for c in self.population)
        # self.best_fitness_history.append(best)
        # self.delta_best.append(0.0)
        # self.child_parent_success_ratio.append(0.0)
        # self.mean_crossover_gain.append(0.0)
        # self.mean_mutation_gain.append(0.0)

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

        self._log(
            f"Returning elite layouts. Best fitness: {viable[0].fitness:.4f}",
            divider=True
        )

        return viable[: self.elite_population_size]


    
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
            child = p1.crossover_ox1(p2)
        elif self.crossover_method == "ox1k":
            child = p1.crossover_ox1_k(p2)
        else:
            raise ValueError(f"Unknown crossover method: {self.crossover_method}")

        # --- mutation (optional) ---
        if random.random() < self.mutation_rate:
            child.mutate()

        self._log(f"Offspring generated in {time.time() - start:.2f} s")
        return child



    def next_generation(self) -> None:
        self._log("Next generation…", divider=True)

        start = time.time()

        old_elite = self._get_elite()
        new_population = list(old_elite)

        # generate offspring until we reach the target population size
        remaining = self.population_size - len(new_population)
        # parallel offspring generation
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._generate_offspring, self.population)
                       for _ in range(remaining)]
            for fut in as_completed(futures):
                child = fut.result()
                new_population.append(child)

        with ThreadPoolExecutor() as executor:
            new_population = list(executor.map(_eval_fitness, new_population))

        self.population = new_population
        self.generation += 1

        # ── average child fitness and full population fitness vector ─────────
        children      = new_population[len(old_elite):]
        avg_child_fit = (sum(c.fitness for c in children) / len(children)) if children else 0.0
        row = {"generation": self.generation,
            "avg_child_fit": avg_child_fit}
        row.update({f"fit_{i}": c.fitness for i, c in enumerate(new_population)})
        self._metrics_buffer.append(row)

        # flush every 5 generations to limit I/O
        if len(self._metrics_buffer) >= 5:
            self._flush_metrics()

            # ---- logging ----
        for i, chrom in enumerate(self.population):
            self._log(f"Layout {i}: {[(p.id, p.rotation) for p in chrom.genes]} | Fitness: {chrom.fitness:.4f}")
        
        end = time.time()
        self._log(f"Generation {self.generation} took {end - start:.2f} seconds.", divider=True)


    def run(self) -> Chromosome:
        """Evolve the population, applying dynamic‑stopping logic if enabled."""
        start = time.time()
        self._log("Starting evolution…", divider=True)
        self.generate_population()

        planned_generations = self.num_generations
        
        if planned_generations <= 0:
            raise ValueError("num_generations must be at least 1")
        if self.max_generations is not None and self.max_generations < planned_generations:
            raise ValueError("max_generations cannot be smaller than num_generations")

        while self.generation < planned_generations:
            self.next_generation()

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

        self._flush_log()  

        return self._get_elite()[0]

    def _flush_log(self) -> None:
        """Write accumulated log lines to disk (append or create)."""
        with self.log_path.open("w", encoding="utf-8") as f:
            for line in self.log_lines:
                f.write(line + "\n")

    
    def _flush_metrics(self) -> None:
        if not self._metrics_buffer:
            return
        mode = "a" if self._csv_header_done else "w"
        with self.csv_path.open(mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._metrics_buffer[0].keys())
            if not self._csv_header_done:
                writer.writeheader()
                self._csv_header_done = True
            writer.writerows(self._metrics_buffer)
        self._metrics_buffer.clear()



# CLI entry point
if __name__ == "__main__":
    from .path_extractor import PatternPathExtractor
    # create a simple test case
    layout_file = "/Users/aysegulbarlas/codestuff/GarmentCode/nesting-assets/Configured_design_specification_asym_dress.json"
    pieces = PatternPathExtractor(layout_file).get_all_panel_pieces(samples_per_edge=3)
    container = Container(140, 500)
    # create an instance of the Evolution class
    evolution = Evolution(pieces, container, early_stop_window=5, early_stop_tolerance=0.01)
    evolution.run()