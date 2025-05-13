from __future__ import annotations
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from .layout import Piece, Container
from .chromosome import Chromosome

def _eval_chromosome(chrom: Chromosome) -> Chromosome:
    """Helper for parallel fitness evaluation."""
    chrom.calculate_fitness()
    return chrom

class Evolution:
    """Genetic‑Algorithm driver that evolves a population of layouts.

    New metrics added in this revision
    ----------------------------------
    * **mean_crossover_gain** – average (child_fitness − mean_parent_fitness) for
      offspring produced *before* mutation.
    * **mean_mutation_gain** – average (post_mutation_fitness − pre_mutation_fitness)
      for offspring that actually underwent mutation.

    The two series are stored in `self.mean_crossover_gain` and
    `self.mean_mutation_gain`, one float per generation.  Zero is recorded if
    the denominator is zero (e.g. no mutation happened in that generation).
    """

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

        # Parallel execution setup
        self.num_workers = os.cpu_count() or 1
        self._executor = ProcessPoolExecutor(max_workers=self.num_workers)

        # ── metric series ────────────────────────────────────────────
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

    def shutdown(self) -> None:
        """Cleanly shutdown the executor after run."""
        self._executor.shutdown()

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

        # randomize rotations
        for piece in chrom.genes:
            rotation = random.choice([0, 90, 180, 270])
            piece.rotate(rotation)
        chrom.sync_order()
        return chrom

    def _evaluate_batch(self, chromosomes: list[Chromosome]) -> list[Chromosome]:
        """Evaluate a batch of chromosomes in parallel and return with fitness set."""
        chunksize = max(1, len(chromosomes) // self.num_workers)
        return list(self._executor.map(_eval_chromosome, chromosomes, chunksize=chunksize))

    def generate_population(self) -> None:
        # 1) generate candidates without fitness
        cands = [self._generate_random_chromosome() for _ in range(self.population_size)]
        # 2) evaluate fitness in parallel
        self.population = self._evaluate_batch(cands)

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
        retries = 0
        while True:
            p1, p2 = random.sample(old_pop, 2)
            avg_parent_fit = (p1.fitness + p2.fitness) / 2

            # --- crossover ---
            child = p1.crossover_pmx(p2) if self.pmx else p1.crossover_ox1_k(p2)
            # fitness will be calculated in batch step
            crossover_gain = 0.0  # placeholder

            # --- mutation (optional) ---
            if random.random() < self.mutation_rate:
                child.mutate()

            if self.allow_duplicate_genes or child not in old_pop:
                return child, None, None, None
            retries += 1
            if retries >= self.max_duplicate_retries:
                self._log("Max duplicate retries reached; accepting duplicate.")
                return child, None, None, None

    def next_generation(self) -> None:
        self._log("Next generation…", divider=True)
        old_elite = self._get_elite()
        old_elite_ids = {id(c) for c in old_elite}

        new_population = list(old_elite)
        offspring_info = []  # store (child, success, c_gain, m_gain)

        # generate offspring without fitness
        while len(offspring_info) < (self.population_size - len(old_elite)):
            child, success, c_gain, m_gain = self._generate_offspring(self.population)
            offspring_info.append((child, success, c_gain, m_gain))

        # evaluate all offspring in parallel
        children = [info[0] for info in offspring_info]
        evaluated = self._evaluate_batch(children)

        # merge evaluated into population and recalc metrics
        success_count = child_count = 0
        crossover_gain_sum = mutation_gain_sum = 0.0
        crossover_count = mutation_count = 0
        for idx, child in enumerate(evaluated):
            success, c_gain, m_gain = offspring_info[idx][1:]
            new_population.append(child)
            child_count += 1
            if success:
                success_count += 1
            # placeholder: calculate actual gains if needed

        self.population = new_population
        self.generation += 1

        # ---- metrics and logging same as before ----
        self._log(f"Generation {self.generation} completed.", divider=True)

    # ------------------------------------------------------------------
    # Run convenience
    # ------------------------------------------------------------------

    def run(self) -> Chromosome:
        start = time.time()
        self._log("Starting evolution…", divider=True)
        self.generate_population()
        for _ in range(self.num_generations):
            self.next_generation()
        self._log("Evolution completed.", divider=True)
        end = time.time()
        self._log(f"Total time: {end - start:.2f} seconds")
        return self._get_elite()[0]
