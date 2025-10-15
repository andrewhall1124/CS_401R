import numpy as np
from rich import print

def iteration(value_table_old: dict[str, list], policy: dict[str, str], graph: dict[str, list[str]], cost_table: dict[str, int], gamma: float):
    value_table = value_table_old.copy()

    # Iterate over each state
    for n in graph.keys():

        # Follow the policy path
        path = policy[n]
        
        # Compute cost
        cost = cost_table[n] + gamma * value_table_old[path]

        if path == n:
            cost += 1

        # Update tables
        value_table[n] = cost
    
    return value_table

def policy_valuation(policy: dict[str, str], graph: dict[str, list[str]], cost_table: dict[str, int], gamma: float, epsilon: float, max_iterations: int = 100):
    value_table_old = {n: 0.0 for n in graph.keys()}        
    for i in range(max_iterations):
        value_table = iteration(
            value_table_old=value_table_old, 
            policy=policy, 
            graph=graph, 
            cost_table=cost_table, 
            gamma=gamma
        )

        max_diff = 0
        for n in graph.keys():
            diff = abs(value_table[n] - value_table_old[n])
            max_diff = max(max_diff, diff)
        
        if max_diff < epsilon:
            print(f"Policy valuation converged after {i + 1} iterations!")
            break
            
        value_table_old = value_table

    return value_table

def policy_improvement(policy: dict[str, str], value_table: dict[str, int], graph: dict[str, list[str]], cost_table: dict[str, int], gamma: float):
    new_policy = policy.copy()

    for n in graph.keys():
        min_cost = float('inf')
        best_path = None

        for path in graph[n]:

            cost = cost_table[n] + gamma * value_table[path]

            if path == n:
                cost += 1
            
            if cost < min_cost:
                min_cost = cost
                best_path = path
        new_policy[n] = best_path
    
    return new_policy



def policy_iteration(graph: dict[str, list[str]], cost_table: dict[str, int], gamma: float, epsilon: float, max_iterations: int = 100):
    policy = {n: graph[n][0] for n in graph.keys()}
    value_table = {n: 0.0 for n in graph.keys()}

    for i in range(max_iterations):
        value_table = policy_valuation(
            policy=policy, 
            graph=graph, 
            cost_table=cost_table, 
            gamma=gamma, 
            epsilon=epsilon, 
            max_iterations=max_iterations
        )

        new_policy = policy_improvement(
            policy=policy, 
            value_table=value_table,
            graph=graph, 
            cost_table=cost_table, 
            gamma=gamma
        )

        if new_policy == policy:
            print(f"Policy iteration converged after {i + 1} iterations.")
            break

        policy = new_policy


    return policy, value_table

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
optimal_policy, value_table = policy_iteration(
    graph=graph, 
    cost_table=cost_table, 
    gamma=gamma, 
    epsilon=epsilon
)

print("Optimal Policy")
print(optimal_policy)

print("Value Table")
print(value_table)
