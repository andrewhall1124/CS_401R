from typing import List, Tuple, Optional

grid: List[List[Optional[int]]] = [
    [0, 0, 0, -1],
    [0, None, 0, 100],
    [0, 0, 0, 0]
]

belief_grid: List[List[Optional[float]]] = [
    [0, 0, .2, .3],
    [0, None, .1, .4],
    [0, 0, 0, 0]
]

# Update belief when observing y_k = c (position [0, 2])
observed_position: Tuple[int, int] = (0, 2)  # row 0, col 2 corresponds to 'c'

# Get neighbors of position c
def get_neighbors(row: int, col: int) -> List[Tuple[int, int]]:
    """Returns list of valid neighbor positions (up, down, left, right)"""
    neighbors: List[Tuple[int, int]] = []
    directions: List[Tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    for dr, dc in directions:
        r, c = row + dr, col + dc
        if 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] is not None:
            neighbors.append((r, c))
    return neighbors

# Calculate observation likelihood for each state
# P(y_k = c | x_k = state)
def observation_likelihood(state: Tuple[int, int], observed: Tuple[int, int]) -> float:
    """
    Probability of observing 'observed' position given true state is 'state'
    - If state == observed: probability = 0.6 + 0.1 * (number of wall neighbors)
    - If state is neighbor of observed: probability = 0.1
    - Otherwise: probability = 0
    """
    if state == observed:
        # Count how many directions from observed hit walls or boundaries
        all_directions: List[Tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        wall_count: int = 0
        for dr, dc in all_directions:
            r, c = observed[0] + dr, observed[1] + dc
            # Check if it's out of bounds or a wall (None)
            if not (0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] is not None):
                wall_count += 1
        return 0.6 + (0.1 * wall_count)
    elif state in get_neighbors(observed[0], observed[1]):
        return 0.1
    else:
        return 0.0

# Apply Bayes filter update: belief_new ∝ P(y_k | x_k) * belief_old
updated_belief: List[List[Optional[float]]] = []
for row in range(len(belief_grid)):
    updated_row: List[Optional[float]] = []
    for col in range(len(belief_grid[row])):
        if belief_grid[row][col] is None:
            updated_row.append(None)
        else:
            likelihood: float = observation_likelihood((row, col), observed_position)
            updated_row.append(belief_grid[row][col] * likelihood)
    updated_belief.append(updated_row)

# Normalize so beliefs sum to 1
total: float = sum(val for row in updated_belief for val in row if val is not None)
normalized_belief: List[List[Optional[float]]] = []
for row in updated_belief:
    normalized_row: List[Optional[float]] = []
    for val in row:
        if val is None:
            normalized_row.append(None)
        else:
            normalized_row.append(val / total if total > 0 else 0)
    normalized_belief.append(normalized_row)

print("Original belief after policy action (p_{k-1}):")
for row in belief_grid:
    print(row)

print("\nObserved position: c (row 0, col 2)")
print(f"Neighbors of c: {get_neighbors(observed_position[0], observed_position[1])}")

print("\nUpdated belief after observing y_k = c:")
for row in normalized_belief:
    formatted_row = []
    for val in row:
        if val is None:
            formatted_row.append("None")
        else:
            formatted_row.append(f"{val:.4f}")
    print(formatted_row)