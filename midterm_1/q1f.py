from rich import print

def calculate_sensor_noise(y: str, x: str, graph: dict[str, list[str]]) -> float:
    # p(y = x) = .5
    if y == x:
        return .5
    else:
        # p(outgoing neighbors) = .25
        if x in graph[y]:
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

        numerator = sensor_noise * transition_noise

        print(f"numerator = {numerator}")

        denominator = 0
        for x_tilde in graph.keys():
            sensor_noise_tilde = calculate_sensor_noise(
                y=y, 
                x=x_tilde, 
                graph=graph
            )

            transition_noise_tilde = calculate_transition_noise(
                x=x_tilde,
                policy=policy,
                x_prev=x_prev,
                graph=graph
            )

            print(f"    x_tilde={x_tilde}   s={sensor_noise_tilde:.4f}  t={transition_noise_tilde:.4f}")

            denominator += sensor_noise_tilde * transition_noise_tilde

        print(f"denominator = {denominator}")

        # Normalize
        normalized = numerator / denominator
        
        contribution = normalized * beliefs[x_prev]
        
        total_belief += contribution
    
    print(f"Belief ={total_belief:.4f}")
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
    new_beliefs[x] = round(new_beliefs[x], 2)

print("\nNew Beliefs")
print(new_beliefs)
