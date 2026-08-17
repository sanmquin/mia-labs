import random
from collections import deque

GRID_SIZE = 10
WALKABLE_TOKENS = {0, 1, 2}

def solve_bfs(grid, start, end):
    """
    Standard BFS solver to find the shortest path from start to end.
    Returns list of (row, col) coordinates or None.
    """
    if not start or not end:
        return None
    if grid[start[0]][start[1]] not in WALKABLE_TOKENS:
        return None

    queue = deque([start])
    visited = {start}
    parent = {}

    while queue:
        cell = queue.popleft()
        if cell == end:
            path = [cell]
            while cell in parent:
                cell = parent[cell]
                path.append(cell)
            path.reverse()
            return path

        r, c = cell
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
                if (nr, nc) not in visited and grid[nr][nc] in WALKABLE_TOKENS:
                    visited.add((nr, nc))
                    parent[(nr, nc)] = cell
                    queue.append((nr, nc))
    return None

def generate_labyrinth(width, height, start, end, num_dead_ends=None, num_loops=0):
    """
    Generates a 10x10 maze from start to end with specified loops.
    """
    grid = [[3] * width for _ in range(height)]

    # Simple randomized DFS to carve corridors
    visited = set()
    def carve(r, c):
        visited.add((r, c))
        grid[r][c] = 0

        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        random.shuffle(dirs)

        for dr, dc in dirs:
            nr, nc = r + 2*dr, c + 2*dc
            if 0 <= nr < height and 0 <= nc < width:
                if (nr, nc) not in visited:
                    grid[r+dr][c+dc] = 0
                    carve(nr, nc)

    # Start carving from a cell near start
    carve(start[0], start[1])

    # Ensure start and end are walkable
    grid[start[0]][start[1]] = 1
    grid[end[0]][end[1]] = 2

    # Make adjacent cells walkable to prevent entrapment
    for r, c in [start, end]:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                grid[nr][nc] = 0

    # Loop injection: randomly turn some walls into walkable paths
    walls = [(r, c) for r in range(1, height-1) for c in range(1, width-1) if grid[r][c] == 3]
    random.shuffle(walls)
    for i in range(min(num_loops, len(walls))):
        r, c = walls[i]
        grid[r][c] = 0

    grid[start[0]][start[1]] = 1
    grid[end[0]][end[1]] = 2

    return grid
