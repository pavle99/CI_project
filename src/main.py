import json
import random
import string
import time

from brute_force import find_lcs
from genetic import Problem, GeneticAlgorithm, GAParams


def get_random_string(n: int, alphabet: string):
    return ''.join([random.choice(list(alphabet)) for _ in range(n)])


class ChromosomeParams:
    def __init__(self, m: int = 5, n: int = 50, alphabet: str = string.ascii_lowercase):
        self.m = m
        self.n = n
        self.alphabet = alphabet


if __name__ == "__main__":
    ms = [2, 3, 5, 10, 20]
    ns = [10, 20, 50, 100, 200]
    alphabets = ['01', 'ACTG', string.ascii_lowercase]
    chromosome_params_arr = [ChromosomeParams(m, n, alphabet) for m in ms for n in ns for alphabet in alphabets]
    results = []
    for chromosome_params in chromosome_params_arr:
        n = chromosome_params.n
        m = chromosome_params.m
        ga_params_arr = [
            GAParams(n * m, n, n // 5, n, i * 0.02, 0.5 + i * 0.05, selection_function) for i in
            range(1, 7) for selection_function in ["tournament_selection", "roulette_selection"]
        ]
        for ga_params in ga_params_arr:
            strs = []
            for i in range(chromosome_params.m):
                strs.append(get_random_string(chromosome_params.n, chromosome_params.alphabet))

            start_time = time.time()
            ga_params.brute_force_len = len(
                find_lcs(strs)) if chromosome_params.m < 5 and chromosome_params.n < 100 else None
            brute_force_time = time.time() - start_time

            problem = Problem(strs)
            genetic_algorithm = GeneticAlgorithm(problem, ga_params)
            start_time = time.time()
            result = genetic_algorithm.optimize()
            genetic_algorithm_time = time.time() - start_time
            results.append({
                "chromosome_params": chromosome_params.__dict__,
                "ga_params": ga_params.__dict__,
                "result": str(result),
                "brute_force_time": brute_force_time if ga_params.brute_force_len is not None else None,
                "genetic_algorithm_time": genetic_algorithm_time
            })
            print(results[-1])

    json.dump(results, open("results.json", "w"))
