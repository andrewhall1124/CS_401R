import numpy as np
from rich import print

graph = {
    'A': ['B', 'C'],
    'B': ['B', 'D'],
    'C': ['B', 'D'],
    'D': ['A', 'D']
}

cost_table = {
    'A': [1, 2, 3, 4, 0],
    'B': [4, 3, 2, 1, 5],
    'C': [2, 1, 2, 1, 5],
    'D': [0, 5, 0, 5, 0]
}

N = len(graph.keys())
K = len(cost_table['A'])

optimal_path = {
    n: [None] * K for n in 'ABCD'
}

value_table = {
    n: [None] * K for n in 'ABCD'
}

# Set terminal cost
for n in value_table.keys():
    value_table[n][K-1] = cost_table[n][K-1]

# Set terminal path
for n in optimal_path.keys():
    optimal_path[n][K-1] = n


# Iterate over K backwards
for k in range(K - 1)[::-1]:

    # Iterate over each state
    for n in graph.keys():

        # Find the path that has the minimum cost
        # Cost = current_cost + min_next_cost
        min_cost = float('inf')
        best_path = None

        for path in graph[n]:
            cost = cost_table[n][k] + value_table[path][k + 1]

            if cost < min_cost:
                min_cost = cost
                best_path = path

        # Update tables
        optimal_path[n][k] = best_path
        value_table[n][k] = min_cost

print("Value Table:")
print(value_table)

print("Optimal Path Table:")
print(optimal_path)

