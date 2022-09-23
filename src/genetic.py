import string
import random


def from_genetic_code_to_string(genetic_code: list[bool]):
    return ''.join([Problem.first_sequence[i] for i in range(len(genetic_code)) if genetic_code[i]])


def is_subsequence(needle: str, haystack: str):
    # if len(needle) > len(haystack):
    #     return False
    #
    # i = j = 0
    # while i < len(needle) and j < len(haystack):
    #     while haystack[j] != needle[i]:
    #         j += 1
    #         if j >= len(haystack):
    #             return False
    #     i += 1
    #     j += 1
    #
    # if i >= len(needle):
    #     return True
    # return False
    it = iter(haystack)
    return all(x in it for x in needle)


class GAParams:
    def __init__(self, generation_size: int = 100, chromosome_size: int = 50, tournament_size: int = 10,
                 reproduction_size: int = 10, mutation_rate: float = 0.1, crossover_rate: float = 0.7,
                 selection_function: str = "tournament_selection", brute_force_len: int = -1):
        self.generation_size = generation_size
        self.chromosome_size = chromosome_size
        self.tournament_size = tournament_size
        self.reproduction_size = reproduction_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.selection_function = selection_function
        self.brute_force_len = brute_force_len


class Chromosome:
    def __init__(self, genetic_code: list[bool], fitness: int):
        self.genetic_code = genetic_code.copy()
        self.fitness = fitness

    def __lt__(self, other: 'Chromosome'):
        return self.fitness < other.fitness

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"{from_genetic_code_to_string(self.genetic_code)} = {self.fitness}, " \
               f"length: {len(from_genetic_code_to_string(self.genetic_code))} "


class Problem:
    first_sequence: str = None

    def __init__(self, sequences: list[str]):
        self.possible_gene_values = [True, False]
        self.sequences = sorted(sequences, key=lambda seq: len(seq))
        Problem.first_sequence = self.sequences[0]
        self.n = len(Problem.first_sequence)
        self.m = len(self.sequences)
        self.brute_force_len = -1

    def calc_k_s(self, str_genetic_code: str):
        k_s = int(0)
        for sequence in self.sequences:
            if is_subsequence(str_genetic_code, sequence):
                k_s += 1
        return k_s

    def calculate_fitness(self, genetic_code: list[bool]):
        fitness = int(0)
        c_s = from_genetic_code_to_string(genetic_code)
        n_c_s = len(c_s)
        if n_c_s == 0:
            return 100
        k_s = self.calc_k_s(c_s)

        if n_c_s == self.n and k_s == self.m:
            fitness = 3000 * (n_c_s + 30 * k_s + 50)
        elif n_c_s < self.n and k_s == self.m:
            fitness = 3000 * (n_c_s + 30 * k_s)
        elif n_c_s == self.n and k_s < self.m:
            fitness = -1000 * (n_c_s + 50) * (self.m - k_s)
        elif n_c_s < self.n and k_s < self.m:
            fitness = -1000 * n_c_s * (self.m - k_s)
        #
        # if k_s == self.m:
        #     fitness = n_c_s * k_s * 30
        # else:
        #     fitness = -(n_c_s * 30)
        return fitness

    def best_fit(self, chromosome: Chromosome):
        if self.brute_force_len != -1 and chromosome.fitness > 0:
            return len(from_genetic_code_to_string(chromosome.genetic_code)) == self.brute_force_len
        return False


class GeneticAlgorithm:
    def __init__(self, problem: Problem, ga_params: GAParams):
        self.problem = problem
        self.problem.brute_force_len = ga_params.brute_force_len

        self.generation_size = ga_params.generation_size
        self.chromosome_size = ga_params.chromosome_size
        self.tournament_size = ga_params.tournament_size
        self.reproduction_size = ga_params.reproduction_size
        self.mutation_rate = ga_params.mutation_rate
        self.percent_mutated = 75
        self.max_iterations = 1000
        self.elitism_size = self.generation_size // 5
        self.crossover_rate = ga_params.crossover_rate
        self.selection_function = self.tournament_selection if ga_params.selection_function == "tournament_selection" else self.roulette_selection

    def initial_population(self):
        result: list[Chromosome] = []
        for _ in range(self.generation_size):
            genetic_code = [random.choice(self.problem.possible_gene_values) for _ in range(self.chromosome_size)]
            fitness = self.problem.calculate_fitness(genetic_code)
            chromosome = Chromosome(genetic_code, fitness)
            result.append(chromosome)
        return result

    def roulette_selection(self, population: list[Chromosome]) -> Chromosome:
        min_fitness = min(population).fitness - 1
        result = random.choices(population, weights=[x.fitness - min_fitness for x in population], k=1)
        return result[0]

    def tournament_selection(self, population: list[Chromosome]) -> Chromosome:
        selected = random.sample(population, self.tournament_size)
        result = max(selected)
        return result

    def selection(self, population: list[Chromosome]) -> list[Chromosome]:
        result = [self.selection_function(population) for _ in range(self.generation_size)]
        return result

    def crossover(self, parent1: Chromosome, parent2: Chromosome):
        # break_point = random.randint(1, self.chromosome_size)
        # child1 = parent1.genetic_code[:break_point] + parent2.genetic_code[break_point:]
        # child2 = parent2.genetic_code[:break_point] + parent1.genetic_code[break_point:]
        p1gc = parent1.genetic_code
        p2gc = parent2.genetic_code
        child1 = [p1gc[i] if random.random() > 0.5 else p2gc[i] for i in range(len(p1gc))]
        child2 = [p2gc[i] if random.random() > 0.5 else p1gc[i] for i in range(len(p1gc))]
        return child1, child2

    def mutate(self, genetic_code: list[bool]):
        random_value = random.random()
        amount_of_mutations = random.randint(1, self.percent_mutated) * len(genetic_code) // 100

        if random_value < self.mutation_rate:
            random_indices = random.sample(range(len(genetic_code)), max(amount_of_mutations, 1))
            for random_index in random_indices:
                genetic_code[random_index] = not genetic_code[random_index]

        return genetic_code

    def create_generation(self, selected: list[Chromosome]):
        result: list[Chromosome] = []

        for _ in range((self.generation_size - self.elitism_size) // 2):
            parents = random.sample(selected, 2)

            child1_code, child2_code = self.crossover(parents[0], parents[1])

            child1_code = self.mutate(child1_code)
            child2_code = self.mutate(child2_code)

            child1 = Chromosome(child1_code, self.problem.calculate_fitness(child1_code))
            child2 = Chromosome(child2_code, self.problem.calculate_fitness(child2_code))

            result.append(child1)
            result.append(child2)

        return result

    def optimize(self):
        population = self.initial_population()

        result = global_best = max(population, key=lambda x: x.fitness)
        global_best_iteration_found = 0

        for i in range(self.max_iterations):
            selected = self.selection(population)

            old_generation = (sorted(population, reverse=True))[0:self.elitism_size]
            population = old_generation + self.create_generation(selected)

            current_best = max(population, key=lambda x: x.fitness)
            # print(current_best)

            if global_best.fitness < current_best.fitness:
                global_best = current_best
                global_best_iteration_found = i

            if self.problem.best_fit(current_best):
                result = current_best
                break

            if i - global_best_iteration_found >= self.problem.n * self.problem.m:
                print(f"No better chromosome in {self.problem.n * self.problem.m} iterations")
                result = global_best
                break
            result = global_best
        return result
