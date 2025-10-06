import copy
from enum import Enum
from rich import print
import numpy as np

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

def policy_iteration_algorithm(rewards: list[list[float]], action_probs: list[float], gamma: float = 0.9):
    m = len(rewards)
    n = len(rewards[0])

    g = []
    P = [[0] * (m * n) for _ in range(m * n)]
    for i in range(m):
        for j in range(n):
            state_index = i * n + j
            reward = rewards[i][j] or 0
            g.append(reward)

            remaining_prob = 1
            for direction, prob in zip([Direction.RIGHT.counterclockwise, Direction.RIGHT, Direction.RIGHT.clockwise], action_probs):
                i_1, j_1 = get_next_position(i, j, direction, m, n)
                next_state_index = i_1 * n + j_1
                if (i, j) != (i_1, j_1):
                    P[state_index][next_state_index] += prob
                    remaining_prob -= prob
            P[state_index][state_index] += remaining_prob

    P = np.array(P)
    g = np.array(g)

    I = np.eye(len(P))
    policy = np.linalg.inv(I - gamma * P) @ g
    policy = policy.reshape(m, n)
    
    policy_list = []
    for i in range(m):
        row = []
        for j in range(n):
            if rewards[i][j] is not None:
                row.append(policy[i][j].item())
            else:
                row.append(None)
        policy_list.append(row)
    
    return policy_list


if __name__ == '__main__':
    rewards = [
        [0, 0, 0, -1],
        [0, None, 0, 100],
        [0, 0, 0, 0]
    ]
        
    action_probs = [0.1, 0.8, 0.1]  # [counterclockwise, intended, clockwise]
    gamma = 0.9

    print("Rewards grid (g(x)):")
    print_grid(rewards)

    print(f"\nAction probabilities: {action_probs}")
    print(f"Gamma (discount factor): {gamma}")

    optimal_grid = policy_iteration_algorithm(rewards, action_probs, gamma=gamma)
    print(optimal_grid)

    print("Optimal Policy")
    print_policy(optimal_grid, rewards, action_probs, gamma)