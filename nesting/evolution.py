from __future__ import annotations
import random

from .layout import Piece, Container
from .chromosome import Chromosome


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
            piece.rotation = rotation
        
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
        self._log("Starting evolution…", divider=True)
        self.generate_population()
        for _ in range(self.num_generations):
            self.next_generation()
        self._log("Evolution completed.", divider=True)
        return self._get_elite()[0]
