
grid = [
    [0, 0, 0, 0, -5],
    [25, None, None, None, 100],
    [0, 50, 0, None, 0],
    [0, -1, 0, 0, 0]
]

belief_grid = [
    [0, 0, 0, 0, 0],
    [0.1, None, None, None, 0],
    [0.1, 0.1, 0.2, None, 0],
    [0.1, 0.2, 0.1, 0.1, 0]
]

action_probs = [.25, .5, .25]

y_k = (2, 2)
action = "left"

m = len(grid)
n = len(grid[0])

def is_valid(cell: tuple) -> bool:
    row, col = cell
    return 0 <= row < m and 0 <= col < n and grid[row][col] is not None

def get_neighbors(cell: tuple) -> list[tuple]:
    row, col = cell
    directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]  # up, down, right, left

    neighbors = []
    for dr, dc in directions:
        neighbor = (row + dr, col + dc)
        if is_valid(neighbor):
            neighbors.append(neighbor)

    return neighbors

def observation_likelihood(y_obs: tuple, x_true: tuple) -> float:
    """
    Calculate P(y_k | x): probability of observing y given true state x.

    If x == y: P = 0.6
    If x is a neighbor of y: P = 0.1
    Otherwise: P = 0

    If y has fewer than 4 neighbors, the missing probability is added to P(y|y).
    """
    if x_true == y_obs:
        neighbors = get_neighbors(y_obs)
        # Base probability is 0.2, plus 0.2 for each missing neighbor
        return 0.2 + 0.2 * (4 - len(neighbors))

    neighbors_of_y = get_neighbors(y_obs)
    if x_true in neighbors_of_y:
        return 0.2

    return 0.0


def transition_likelihood(x_curr: tuple, action: str, x_prev: tuple) -> float:
    """
    Calculate P(x | u_{k-1}, x'): probability of being in state x given
    previous action and previous state x'.

    The agent moves in the intended direction with probability 0.8,
    and to the left or right with probability 0.1 each.
    """
    if not is_valid(x_curr) or not is_valid(x_prev):
        return 0.0

    # Define action directions
    action_dirs = {
        'up': (-1, 0),
        'down': (1, 0),
        'right': (0, 1),
        'left': (0, -1)
    }

    # Define left and right relative to each action
    left_right = {
        'up': ['left', 'right'],
        'down': ['right', 'left'],
        'right': ['up', 'down'],
        'left': ['down', 'up']
    }

    if action not in action_dirs:
        return 0.0

    # Calculate the target position for the intended action
    dr, dc = action_dirs[action]
    intended_pos = (x_prev[0] + dr, x_prev[1] + dc)

    # If intended position is invalid (wall/boundary), agent stays in place
    if not is_valid(intended_pos):
        intended_pos = x_prev

    # Calculate left and right positions
    left_action, right_action = left_right[action]

    dr_left, dc_left = action_dirs[left_action]
    left_pos = (x_prev[0] + dr_left, x_prev[1] + dc_left)
    if not is_valid(left_pos):
        left_pos = x_prev

    dr_right, dc_right = action_dirs[right_action]
    right_pos = (x_prev[0] + dr_right, x_prev[1] + dc_right)
    if not is_valid(right_pos):
        right_pos = x_prev

    # Calculate probability by checking which movements lead to x_curr
    prob = 0.0

    if x_curr == intended_pos:
        prob += 0.5
    if x_curr == left_pos:
        prob += 0.25
    if x_curr == right_pos:
        prob += 0.25

    return prob


def calculate_new_belief(y_obs: tuple, action: str, prev_belief: list[list[float]]) -> list[list[float]]:
    """
    Calculate the new belief p_k(x) for a specific state given observation y_k.

    p_k(x) = sum over x' of [P(y_k|x)P(x|u_{k-1}, x') / sum_x_tilde P(y_k|x_tilde)P(x_tilde|u_{k-1}, x')] * p_{k-1}(x')
    """
    new_belief = [[0.0 for _ in range(n)] for _ in range(m)]

    for row in range(m):
        for col in range(n):
            if grid[row][col] is None:
                continue
            x_curr = (row, col)
            belief_sum = 0.0

            # Sum over all previous states x'
            for prev_row in range(m):
                for prev_col in range(n):
                    if grid[prev_row][prev_col] is None or prev_belief[prev_row][prev_col] == 0:
                        continue

                    x_prev = (prev_row, prev_col)

                    # Calculate numerator: P(y_k|x) * P(x|u_{k-1}, x')
                    numerator = observation_likelihood(y_obs, x_curr) * transition_likelihood(x_curr, action, x_prev)
                    # Calculate denominator: sum over all x_tilde
                    denominator = 0.0
                    for tilde_row in range(m):
                        for tilde_col in range(n):
                            if grid[tilde_row][tilde_col] is None:
                                continue

                            x_tilde = (tilde_row, tilde_col)
                            denominator += observation_likelihood(y_obs, x_tilde) * transition_likelihood(x_tilde, action, x_prev)

                    if denominator > 0:
                        belief_sum += (numerator / denominator) * prev_belief[prev_row][prev_col]

            new_belief[row][col] = belief_sum

    return new_belief


new_belief = calculate_new_belief(y_k, action, belief_grid)
print("New belief grid:")
for row in new_belief:
    print([f"{x:.3f}" if x > 0 else "0" for x in row])