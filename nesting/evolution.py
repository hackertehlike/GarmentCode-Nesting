from __future__ import annotations
import random
import time
#from concurrent.futures import ProcessPoolExecutor

from .layout import Piece, Container
from .chromosome import Chromosome

# ── evolution.py  (top-level, above class Evolution) ────────────────────────
def _eval_fitness(chrom: "Chromosome") -> "Chromosome":
    """Worker helper for ProcessPoolExecutor: recompute fitness in place."""
    chrom.calculate_fitness()            # container is already held by chrom
    return chrom


class Evolution:
    """Genetic‑Algorithm driver that evolves a population of layouts.

    New metrics added in this revision
    ----------------------------------
    * **mean_crossover_gain** – average \(child_fitness − mean_parent_fitness) for
      offspring produced *before* mutation.
    * **mean_mutation_gain** – average \(post_mutation_fitness − pre_mutation_fitness)
      for offspring that actually underwent mutation.

    The two series are stored in `self.mean_crossover_gain` and
    `self.mean_mutation_gain`, one float per generation.  Zero is recorded if
    the denominator is zero (e.g. no mutation happened in that generation).
    """

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
        pmx: bool = True,
        allow_duplicate_genes: bool = False,
        max_duplicate_retries: int = 50,
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
        self.pmx = pmx
        self.allow_duplicate_genes = allow_duplicate_genes
        self.max_duplicate_retries = max_duplicate_retries

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

        # ── in‑memory run log ───────────────────────────────────────
        self.log_lines: list[str] = []
        self._log(
            f"Evolution instance created. allow_duplicates={self.allow_duplicate_genes}",
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
            piece.rotate(rotation)
        
        chrom.sync_order()
        chrom.calculate_fitness()
        return chrom

    def generate_population(self) -> None:
        while len(self.population) < self.population_size:
            self.population.append(self._generate_random_chromosome())
        self._log(f"Initial population of {self.population_size} layouts generated.", divider=True)
        for i, chrom in enumerate(self.population):
            self._log(f"Layout {i}: {[(p.id, p.rotation) for p in chrom.genes]} | Fitness: {chrom.fitness:.4f}")
        self._log("Population generation completed.", divider=True)

        best = max(c.fitness for c in self.population)
        self.best_fitness_history.append(best)
        self.delta_best.append(0.0)
        self.child_parent_success_ratio.append(0.0)
        self.mean_crossover_gain.append(0.0)
        self.mean_mutation_gain.append(0.0)

    # ------------------------------------------------------------------
    # GA core operations
    # ------------------------------------------------------------------

    def _get_elite(self) -> list[Chromosome]:
        self.population.sort(key=lambda c: c.fitness, reverse=True)
        self._log(f"Returning elite layouts. Best fitness: {self.population[0].fitness:.4f}", divider=True)
        return self.population[: self.elite_population_size]

    def _generate_offspring(self, old_pop: list[Chromosome]) -> tuple[Chromosome, bool, float, float]:
        """Return (child, success_flag, crossover_gain, mutation_gain)."""
        retries = 0
        while True:
            p1, p2 = random.sample(old_pop, 2)
            avg_parent_fit = (p1.fitness + p2.fitness) / 2

            # --- crossover ---
            child = p1.crossover_pmx(p2) if self.pmx else p1.crossover_ox1_k(p2)
            child.calculate_fitness()  # fitness after crossover
            crossover_gain = child.fitness - avg_parent_fit

            # --- mutation (optional) ---
            mutation_gain = 0.0
            if random.random() < self.mutation_rate:
                pre_mut_fit = child.fitness
                child.mutate()
                child.calculate_fitness()
                mutation_gain = child.fitness - pre_mut_fit

            success = child.fitness > p1.fitness and child.fitness > p2.fitness

            if self.allow_duplicate_genes or child not in old_pop:
                return child, success, crossover_gain, mutation_gain
            retries += 1
            if retries >= self.max_duplicate_retries:
                self._log("Max duplicate retries reached; accepting duplicate.")
                return child, success, crossover_gain, mutation_gain

    def next_generation(self) -> None:
        self._log("Next generation…", divider=True)
        old_elite = self._get_elite()
        old_elite_ids = {id(c) for c in old_elite}

        new_population = list(old_elite)
        success_count = child_count = 0
        crossover_gain_sum = mutation_gain_sum = 0.0
        crossover_count = mutation_count = 0

        while len(new_population) < self.population_size:
            child, success, c_gain, m_gain = self._generate_offspring(self.population)
            if self.allow_duplicate_genes or child not in new_population:
                new_population.append(child)
                child_count += 1
                if success:
                    success_count += 1
                crossover_gain_sum += c_gain
                crossover_count += 1
                if m_gain != 0.0:
                    mutation_gain_sum += m_gain
                    mutation_count += 1

        # Recalculate fitness in parallel
        # def calc_fitness(chromosome):
        #     chromosome.calculate_fitness()
        #     return chromosome

        # with ProcessPoolExecutor() as executor:
        #     new_population = list(executor.map(_eval_fitness, new_population))

        # serially evaluate fitness
        for chrom in new_population:
            chrom.calculate_fitness()

        self.population = new_population
        self.generation += 1

        # ---- metrics ----
        new_elite = self._get_elite()
        new_elite_ids = {id(c) for c in new_elite}
        survival_rate = len(old_elite_ids & new_elite_ids) / len(old_elite_ids)
        self.survival_rates.append(survival_rate)

        children = new_population[len(old_elite):]
        avg_child_fit = sum(c.fitness for c in children) / len(children) if children else 0.0
        self.avg_child_fitnesses.append(avg_child_fit)

        cur_best = new_elite[0].fitness
        prev_best = self.best_fitness_history[-1]
        self.best_fitness_history.append(cur_best)
        self.delta_best.append(cur_best - prev_best)

        ratio = success_count / child_count if child_count else 0.0
        self.child_parent_success_ratio.append(ratio)

        mean_c_gain = crossover_gain_sum / crossover_count if crossover_count else 0.0
        mean_m_gain = mutation_gain_sum / mutation_count if mutation_count else 0.0
        self.mean_crossover_gain.append(mean_c_gain)
        self.mean_mutation_gain.append(mean_m_gain)

        # ---- logging ----
        for i, chrom in enumerate(self.population):
            self._log(f"Layout {i}: {[(p.id, p.rotation) for p in chrom.genes]} | Fitness: {chrom.fitness:.4f}")
        self._log(f"Elite survival rate: {survival_rate:.2%}")
        self._log(f"Average child fitness: {avg_child_fit:.4f}")
        self._log(f"Δ‑Best: {self.delta_best[-1]:+.4f}")
        self._log(f"Child‑parent success ratio: {ratio:.2%}")
        self._log(f"Mean crossover gain: {mean_c_gain:+.4f}")
        self._log(f"Mean mutation gain: {mean_m_gain:+.4f}")
        self._log(f"Generation {self.generation} completed.", divider=True)


    # ------------------------------------------------------------------
    # Run convenience
    # ------------------------------------------------------------------

    def run(self) -> Chromosome:
        """Evolve the population, applying dynamic‑stopping logic if enabled."""
        start = time.time()
        self._log("Starting evolution…", divider=True)
        self.generate_population()

        planned_generations = self.num_generations
        # guard against pathological user input
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

            # ----------------‑ ADAPTIVE EXTENSION ---------------‑
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

            # ----------------‑ MAX‑GEN CLAMP ---------------------‑
            if self.max_generations is not None and planned_generations > self.max_generations:
                planned_generations = self.max_generations

        self._log("Evolution completed.", divider=True)
        end = time.time()
        self._log(f"Total time: {end - start:.2f} seconds")
        return self._get_elite()[0]

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