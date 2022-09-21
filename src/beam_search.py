from brute_force import find_lcs
import math
import numpy as np

class Node:
    def __init__(self, left_vector : list[int], l : int, sequence : str):
        self.left_vector = left_vector
        self.l = l
        self.sequence = sequence

    def __str__(self):
        return "left vector: " + str(self.left_vector) + "\nl: " + \
            str(self.l) + "\nsequence: " + self.sequence

class Beam:
    algorithms = ['POW', 'H']

    def __init__(self, S, alphabet, algorithm):
        self.S = S
        self.m = len(self.S)
        self.alphabet = alphabet

        if algorithm not in Beam.algorithms:
            print("Input a valid algorithm")
            return
        self.algorithm = algorithm
        self.best = ""
        if algorithm == 'H':
            self.probability_matrix = self.calculate_probability_matrix()
        self.a = 1
        self.b = 1
        self.c = 0
    

    # Mousavi and Tabataba
    def calculate_probability_matrix(self):
        n = max([len(s) for s in self.S])

        alpha = 1/len(self.alphabet)
        beta = (len(self.alphabet) - 1)/len(self.alphabet)

        matrix = [[-1 for j in range(n)] for i in range(n)]

        for p in range(n):
            for q in range(n):
                if p == 0:
                    matrix[p][q] = 1
                elif p > q:
                    matrix[p][q] = 0
                else:
                    if matrix[p-1][q-1] == -1 or matrix[p][q-1] == -1:
                        print("ERROR!!!")
                        return
                    
                    matrix[p][q] = alpha*matrix[p-1][q-1] + beta*matrix[p][q-1]

        return matrix

    def dominates(self, good_node : Node, node : Node):
        for i, index in enumerate(good_node.left_vector):
            if index < node.left_vector[i]:
                return False
        return True

    def fraser_upper_bound(self, v : Node):
        return min([len(self.S[i]) - v.left_vector[i] for i in range(self.m)])

    def blum_upper_bound(self, v : Node):
        counts = {}

        for i, s in enumerate(self.S):
            new_counts = {}
            for letter in self.alphabet:
                new_counts[letter] = 0

            for j in range(v.left_vector[i], len(s)):
                new_counts[s[j]] += 1

            for letter, count in new_counts.items():
                if counts.get(letter) is None or counts[letter] > count:
                    counts[letter] = count

        return sum(counts.values())

    def wang_upper_bound(self, v : Node):
        min = float('inf')
        for i in range(self.m - 1):
            s1 = self.S[i]
            s2 = self.S[i+1]

            v1 = v.left_vector[i]
            v2 = v.left_vector[i+1]
            lcs = find_lcs([s1[v1:], s2[v2:]])
            if len(lcs) < min:
                min = len(lcs)

        return min

    def blum_wang_upper_bound(self, v : Node):
        return min(self.blum_upper_bound(v), self.wang_upper_bound(v))


    def get_k(self, extended_nodes : list[Node]):
        k = max(1, math.floor(1/len(self.alphabet) * min([len(self.S[i]) - v.left_vector[i] for v in extended_nodes for i in range(self.m)])))
        return k
    
    def probability_heuristic(self, v : Node, k):
        result = np.prod([self.probability_matrix[k][len(self.S[i]) - v.left_vector[i]] for i in range(self.m)])

        return result

    def power_heuristic(self, v : Node):
        q = self.a * math.exp(-self.b * self.m) + self.c
        
        return (np.prod([len(self.S[i]) - v.left_vector[i] for i in range(self.m)]))**q * self.fraser_upper_bound(v)


    def extend_node(self, v : Node):
        list_of_successors = []
        counts = {}

        for i, s in enumerate(self.S):
            new_counts = {}
            for letter in self.alphabet:
                new_counts[letter] = 0

            for j in range(v.left_vector[i], len(s)):
                new_counts[s[j]] += 1

            for letter, count in new_counts.items():
                if counts.get(letter) is None or counts[letter] > count:
                    counts[letter] = count
                

        for key, value in counts.items():
            if value > 0:
                new_left_vector = [(self.S[i][v.left_vector[i]:]).index(key) + 1 + v.left_vector[i] for i in range(self.m)]
                list_of_successors.append(Node(new_left_vector, v.l + 1, v.sequence + key))

        return list_of_successors

    def extend_and_evaluate(self, B : list[Node], h) -> list[Node]:
        extended_nodes = []
        complete_nodes = []
        for v in B:
            extended_node = self.extend_node(v)
            if len(extended_node) > 0:
                extended_nodes += extended_node
            else:
                complete_nodes.append(v)
        
        if len(extended_nodes) > 0:
            if h == self.probability_heuristic:
                k = self.get_k(extended_nodes)
                extended_nodes = sorted(extended_nodes, key=lambda x: h(x, k), reverse=True)
            else:
                extended_nodes = sorted(extended_nodes, key=lambda x: h(x), reverse=True)

        return extended_nodes, complete_nodes

    def prune(self, extended_nodes : list[Node], upper_bound, W):
        pruned = [node for node in extended_nodes if upper_bound(node) + node.l > W]
        return pruned

    def filter_k(self, extended_nodes : list[Node], k_best):
        best_nodes = extended_nodes[:k_best]

        rest = extended_nodes[k_best:]

        result = best_nodes.copy()

        for node in rest:
            for good_node in best_nodes:
                if not self.dominates(good_node, node):
                    result.append(node)
        

        return result

    def search(self, k_best, beta):

        B = [Node([0 for _ in self.S], 0, "")]

        W = 5

        if self.algorithm == 'POW':
            h = self.power_heuristic
        elif self.algorithm == 'HBLUM':
            h = self.probability_heuristic
        elif self.algorithm == 'H':
            h = self.probability_heuristic

        while len(B) > 0:
            extended_nodes, complete_nodes = self.extend_and_evaluate(B, h)

            # update LCS if a complete node with a new largest value reached

            for v in complete_nodes:
                if v.l > len(self.best):
                    self.best = v.sequence

            if len(extended_nodes) > 0:
                extended_nodes = self.filter_k(extended_nodes, max(k_best, 1))


            B = extended_nodes[:beta]  # reduce

        return self.best


