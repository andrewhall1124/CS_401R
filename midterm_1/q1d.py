import numpy as np
from rich import print

def iteration(graph: dict[str, list[str]], value_table_old: dict[str, list], cost_table: dict[str, int], gamma: float):
    value_table = value_table_old.copy()

    # Iterate over each state
    for n in graph.keys():

        # Find the path that has the minimum cost
        # Cost = current_cost + min_next_cost
        min_cost = float('inf')

        for path in graph[n]:
            
            cost = cost_table[n] + gamma * value_table_old[path]

            if path == n:
                cost += 1

            if cost < min_cost:
                min_cost = cost

        # Update tables
        value_table[n] = min_cost
    
    return value_table

def value_iteration(graph: dict[str, list[str]], cost_table: dict[str, int], gamma: float, epsilon: float, max_iterations: int = 100):
    value_table_old = {
        n: 0 for n in graph.keys()
    }

    for i in range(max_iterations):

        value_table = iteration(graph, value_table_old, cost_table, gamma)

        max_diff = 0
        for n in graph.keys():
            diff = abs(value_table[n] - value_table_old[n])
            max_diff = max(max_diff, diff)
        
        if max_diff < epsilon:
            print(f"Converged after {i + 1} iterations!")
            break

        value_table_old = value_table

    return value_table

def get_optimal_path(graph: dict[str, list[str]], cost_table: dict[str, int], value_table: dict[str, int], gamma: float):
    optimal_path = {n: '' for n in graph.keys()}

    for n in optimal_path.keys():
        min_value = float('inf')

        for path in graph[n]:
            value = cost_table[n] + gamma * value_table[path]
            
            if path == n:
                value += 1

            if value < min_value:
                min_value = value
                optimal_path[n] = path
    
    return optimal_path

# Parameters
graph = {
    'A': ['B', 'C'],
    'B': ['B', 'D'],
    'C': ['B', 'D'],
    'D': ['A', 'D']
}

cost_table = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 0
}

gamma = .6
epsilon = 1e-6

# Value Table
value_table = value_iteration(graph, cost_table, gamma, epsilon)
print(value_table)

# Optimal Path
optimal_path = get_optimal_path(graph, cost_table, value_table, gamma)
print(optimal_path)
