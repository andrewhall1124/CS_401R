import copy
from enum import Enum
from rich import print

class Direction(Enum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3

    @property
    def counterclockwise(self):
        return Direction((self.value - 1) % 4)

    @property
    def clockwise(self):
        return Direction((self.value + 1) % 4)

def get_next_position(i, j, action, m, n):
    if action == Direction.LEFT:
        return i, max(0, j - 1)
    elif action == Direction.UP:
        return max(0, i - 1), j
    elif action == Direction.RIGHT:
        return i, min(n - 1, j + 1)
    elif action == Direction.DOWN:
        return min(m - 1, i + 1), j
    return i, j

def print_grid(grid: list[list[float]]):
    for row in grid:
        formatted_row = []
        for cell in row:
            if cell is None:
                formatted_row.append("Wall")
            else:
                formatted_row.append(round(cell, 2))
        print(formatted_row)

def print_policy(grid: list[list[float]], rewards: list[list[float]], action_probs: list[float], gamma: float):
    m = len(grid)
    n = len(grid[0])

    # Direction arrows for visualization
    direction_arrows = {
        Direction.LEFT: "←",
        Direction.UP: "↑",
        Direction.RIGHT: "→",
        Direction.DOWN: "↓"
    }

    policy_grid = []

    for i in range(m):
        policy_row = []
        for j in range(n):

            if grid[i][j] is None:
                policy_row.append("Wall")
                continue

            best_action = None
            min_expected_value = float('inf')

            # Consider all 4 possible intended actions to find the best one
            for intended_action in Direction:
                expected_value = 0

                actions = [
                    intended_action.counterclockwise,
                    intended_action,
                    intended_action.clockwise
                ]

                for k, action in enumerate(actions):
                    prob = action_probs[k]
                    next_i, next_j = get_next_position(i, j, action, m, n)

                    next_value = 0 if grid[next_i][next_j] is None else grid[next_i][next_j]
                    expected_value += prob * next_value

                if expected_value < min_expected_value:
                    min_expected_value = expected_value
                    best_action = intended_action

            policy_row.append(direction_arrows[best_action])

        policy_grid.append(policy_row)

    # Print the policy grid
    for row in policy_grid:
        print(row)

def iteration(current_grid: list[list[float]], rewards: list[list[float]], action_probs: list[float], gamma: float):
    m = len(current_grid)
    n = len(current_grid[0])
    new_grid = copy.deepcopy(current_grid)
    
    for i in range(m):
        for j in range(n):

            # Skip walls/obstacles
            if rewards[i][j] is None:
                continue

            min_expected_value = float('inf')

            # Consider all 4 possible intended actions
            for intended_action in Direction:
                expected_value = 0

                actions = [
                    intended_action.counterclockwise,
                    intended_action,
                    intended_action.clockwise
                ]

                for k, action in enumerate(actions):
                    prob = action_probs[k]
                    next_i, next_j = get_next_position(i, j, action, m, n)

                    next_value = 0 if current_grid[next_i][next_j] is None else current_grid[next_i][next_j]
                    expected_value += prob * next_value

                min_expected_value = min(min_expected_value, expected_value)

            new_grid[i][j] = rewards[i][j] + gamma * min_expected_value
    
    return new_grid

def value_iteration_algorithm(rewards: list[list[float]], action_probs: list[float], epsilon: float = 1e-6, gamma: float = 0.9, max_iterations: int = 100):
    m = len(rewards)
    n = len(rewards[0])
    
    current_grid = copy.deepcopy(rewards)
    
    iteration_count = 0
    while iteration_count < max_iterations:
        print(f'\nW_{iteration_count} (Iteration {iteration_count}):')
        print_grid(current_grid)
        
        new_grid = iteration(current_grid, rewards, action_probs, gamma)

        # Calculate infinity norm of the difference
        max_diff = 0
        for i in range(m):
            for j in range(n):
                if current_grid[i][j] is not None and new_grid[i][j] is not None:
                    diff = abs(new_grid[i][j] - current_grid[i][j])
                    max_diff = max(max_diff, diff)
        
        iteration_count += 1
        
        if max_diff < epsilon:
            print(f"\nConverged after {iteration_count} iterations (max_diff: {max_diff:.6f})")
            break
            
        current_grid = new_grid

    print(f'\nFinal W_{iteration_count}:')
    print_grid(current_grid)
    
    return current_grid

rewards = [
    [0, 0, 0, -1],
    [0, None, 0, 100],
    [0, 0, 0, 0]
]
    
action_probs = [0.1, 0.8, 0.1]  # [counterclockwise, intended, clockwise]
gamma = 0.9
epsilon = 1e-7

print("Rewards grid (g(x)):")
print_grid(rewards)

print(f"\nAction probabilities: {action_probs}")
print(f"Gamma (discount factor): {gamma}")
print(f"Epsilon: {epsilon}")

optimal_grid = value_iteration_algorithm(rewards, action_probs, epsilon=epsilon, gamma=gamma, max_iterations=100)

print("Optimal Policy")
print_policy(optimal_grid, rewards, action_probs, gamma)