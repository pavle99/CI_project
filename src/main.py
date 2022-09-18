import random
import string

from src.brute_force import find_lcs
from src.genetic import Problem, GeneticAlgorithm, ChromosomeParams, GAParams


def get_random_string(n: int, alphabet: string):
    return ''.join([random.choice(list(alphabet)) for _ in range(n)])


if __name__ == "__main__":
    ms = [2, 5, 10, 20, 50, 100]
    ns = [10, 20, 50, 100, 200, 500]
    alphabets = ['01', 'ACTG', string.ascii_lowercase]
    chromosome_params_arr = [ChromosomeParams(m, n, alphabet) for m in ms for n in ns for alphabet in alphabets]

    generation_sizes = [20, 50, 100, 200, 500, 1000]
    chromosome_sizes = [10, 20, 50, 100, 200, 500]  # = ns
    tournament_sizes = [2, 5, 10, 20, 50, 100]
    reproduction_sizes = [2, 5, 10, 20, 50, 100]
    mutation_rates = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25]
    crossover_rates = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
    selection_functions = ["tournament_selection", "roulette_selection"]
    ga_params_arr = [
        GAParams(generation_size, chromosome_size, tournament_size, reproduction_size, mutation_rate, crossover_rate,
                 selection_function) for generation_size in generation_sizes for chromosome_size in chromosome_sizes
        for tournament_size in tournament_sizes for reproduction_size in reproduction_sizes for mutation_rate in
        mutation_rates for crossover_rate in crossover_rates for selection_function in selection_functions
    ]

    results = []
    for chromosome_params in chromosome_params_arr:
        for ga_params in ga_params_arr:
            strs = []
            for i in range(chromosome_params.m):
                strs.append(get_random_string(chromosome_params.n, chromosome_params.alphabet))
            ga_params.brute_force_len = len(find_lcs(strs)) if chromosome_params.m < 10 and chromosome_params.n < 100 else None

            problem = Problem(strs)
            genetic_algorithm = GeneticAlgorithm(problem, ga_params)
            result = genetic_algorithm.optimize()
            results.append({
                "chromosome_params": chromosome_params.__dict__,
                "ga_params": ga_params.__dict__,
                "result": result
            })

    for result in results:
        print(result)



