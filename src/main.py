import json
import random
import string
import time

from brute_force import find_lcs
from genetic import Problem, GeneticAlgorithm, GAParams, is_subsequence
from beam_search import Beam


def get_random_string(n: int, alphabet: string):
    return ''.join([random.choice(list(alphabet)) for _ in range(n)])


class ChromosomeParams:
    def __init__(self, m: int = 5, n: int = 50, alphabet: str = string.ascii_lowercase):
        self.m = m
        self.n = n
        self.alphabet = alphabet


def test_GA(chromosome_params_arr: list[ChromosomeParams]):
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
    return results


def test_BS(chromosome_params_arr: list[ChromosomeParams]):
    results = []
    algorithms = ['POW', 'H']
    betas = [5, 20, 60]
    for chromosome_params in chromosome_params_arr:
        strs = []
        for i in range(chromosome_params.m):
            strs.append(get_random_string(chromosome_params.n, chromosome_params.alphabet))

        for algorithm in algorithms:
            for beta in betas:
                beam = Beam(strs, chromosome_params.alphabet, algorithm)
                start_time = time.time()
                result = beam.search(50 if algorithm == 'H' else 100, beta)
                beam_search_time = time.time() - start_time
                results.append({
                    "chromosome_params": chromosome_params.__dict__,
                    "beam_search_time": beam_search_time,
                    "result": str(result),
                    "algorithm": algorithm,
                    "beta": beta*10
                })
                print(results[-1])
    return results


if __name__ == "__main__":
    ms = [10, 15, 20, 25, 40, 60, 80, 100, 150, 200]
    ns = [20, 50, 100, 200, 600]
    alphabets = ['ACTG', string.ascii_lowercase[:20]]
    chromosome_params_arr = [ChromosomeParams(m, n, alphabet) for m in ms for n in ns for alphabet in alphabets]

    results_GA = test_GA(chromosome_params_arr)
    results_BS = test_BS(chromosome_params_arr)

    json.dump(results_GA, open("GA_results.json", "w"))
    json.dump(results_BS, open("BS_results.json", "w"))
