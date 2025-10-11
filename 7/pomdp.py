
grid = [
    [0, 0, 0, -1],
    [0, None, 0, 100],
    [0, 0, 0, 0]
]

belief_grid = [
    [0, 0, .2, .3],
    [0, None, .1, .4],
    [0, 0, 0, 0]
]

action_probs = [.1, .8, .1]

# Update belief when observing y_k = c (position [0, 2])
y_k = (0, 2)  # row 0, col 2 corresponds to 'c'
action = "up"  # The action taken at time k-1

m = len(grid)
n = len(grid[0])

def is_valid(cell: tuple) -> bool:
    row, col = cell
    return 0 <= row <= m and 0 <= col <= n

def get_neighbors(cell: tuple) -> list[tuple]:
    x, y = cell
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)] # up, down, right, left

    neighbors = []
    for dy, dx in directions:
        neighbor = (x + dx, y + dy)
        if is_valid(neighbor) and grid[x + dx][y + dy] is not None:
            neighbors.append(neighbor)
    
    return neighbors

def likelihood(y: tuple, x: tuple) -> float:
    neighbors = get_neighbors(y)
    probability = 1 - (.1 * len(neighbors))
    print(probability)


likelihood(y_k, y_k)