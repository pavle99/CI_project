import numpy as np
from numpy import ndarray
from itertools import product


def create_dp(seqs: list[str], lengths: list[int]) -> ndarray:
    dp = np.zeros(tuple(length + 1 for length in lengths))

    for indices, elements in zip(product(*(range(len(seq)) for seq in seqs)), product(*seqs)):
        if len(set(elements)) == 1:
            dp[tuple(index + 1 for index in indices)] = dp[(tuple(indices))] + 1
        else:
            dp[tuple(index + 1 for index in indices)] = max(
                [
                    dp[tuple(index + 1 if k != i else index for k, index in enumerate(indices))] for i in
                    range(len(indices))
                ]
            )

    return dp


def find_lcs(seqs: list[str]) -> str:
    lengths = [len(seq) for seq in seqs]
    dp = create_dp(seqs, lengths)

    lcs = ""
    while all(length > 0 for length in lengths):
        step = dp[tuple(length for length in lengths)]
        for i in range(len(lengths)):
            if step == dp[tuple(length - 1 if k == i else length for k, length in enumerate(lengths))]:
                lengths[i] -= 1
                break
        else:
            lcs += seqs[0][lengths[0] - 1]
            for i in range(len(lengths)):
                lengths[i] -= 1

    return lcs[::-1]
