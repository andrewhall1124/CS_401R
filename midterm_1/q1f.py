from rich import print

def calculate_sensor_noise(y: str, x: str, graph: dict[str, list[str]]) -> float:
    # p(y = x) = .5
    if y == x:
        return .5
    else:
        # p(outgoing neighbors) = .25
        if y in graph[x]:
            return .25
        else:
            return 0

def calculate_transition_noise(x: str, policy: dict[str, str], x_prev: str, graph: dict[str, list[str]]) -> float:
    if x in graph[x_prev]:
        # p(optimal policy) = .75
        if x == policy[x_prev]:
            return .75
        # p(other) = .25
        else:
            return .25
    else:
        return 0

def calculate_belief(x: str, y: str, graph: dict[str, list[str]], policy: dict[str, str], beliefs: dict[str, float]) -> float:
    print(f"\nCalculating belief for: {x}")
    total_belief = 0
    for x_prev in graph.keys():
        sensor_noise = calculate_sensor_noise(
            y=y, 
            x=x, 
            graph=graph
        )

        transition_noise = calculate_transition_noise(
            x=x,
            policy=policy,
            x_prev=x_prev,
            graph=graph
        )

        print(f"y={y}   x_prev={x_prev}    s={sensor_noise}    t={transition_noise}")

        numerator = sensor_noise * transition_noise * beliefs[x_prev]
        
        total_belief += numerator
        print(f"numerator = {numerator}")
    return total_belief

# Parameters
graph = {
    'A': ['B', 'C'],
    'B': ['B', 'D'],
    'C': ['B', 'D'],
    'D': ['A', 'D']
}

policy = {
    'A': 'B',
    'B': 'B',
    'C': 'B',
    'D': 'D'
}

beliefs = {
    'A': 0.4,
    'B': 0.1,
    'C': 0.5,
    'D': 0.0
}

observation = 'C'

new_beliefs = {}
for x in beliefs.keys():
    new_beliefs[x] = calculate_belief(x, y=observation, graph=graph, policy=policy, beliefs=beliefs)

    for key, belief in new_beliefs.items():
        belief_sum = sum(new_beliefs.values())
        if belief_sum == 0:
            new_beliefs[key] = 0
        else:
            new_beliefs[key] /= sum(new_beliefs.values())

    new_beliefs[x] = round(new_beliefs[x], 2)
    print(f"x={x}   belief={new_beliefs[x]}")

print("\nNew Beliefs")
print(new_beliefs)
