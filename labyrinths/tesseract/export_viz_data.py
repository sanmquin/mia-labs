import os
import sys
import json
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add the directory containing test_adversarial.py to the path
sys.path.append(os.path.dirname(__file__))

from test_adversarial import (
    LabyrinthTransformer,
    solve_autoregressive,
    bfs_shortest_path,
    build_maze_1_detour,
    build_maze_2_spiral,
    build_maze_3_false_shortcut,
    build_maze_4_zigzag,
    build_maze_5_bottleneck,
    build_maze_6_uturn
)

GRID_SIZE = 10
NUM_CELLS = 100
WALKABLE_TOKENS = {0, 1, 2}
DEVICE = "cpu"

# ── Procedural Maze Generator ──
# To create realistic 10x10 mazes without needing the external Intro-NN library.
# We implement a classic randomized DFS generator with loop injection and dead end pruning/tuning.

def generate_procedural_maze(seed, loops=5):
    """
    Generates a 10x10 maze with a start corner and an end corner.
    Injects loops by removing some walls.
    """
    random.seed(seed)
    # Start/End corners
    corners = [
        ((0, 0), (9, 9)),
        ((9, 9), (0, 0)),
        ((0, 9), (9, 0)),
        ((9, 0), (0, 9))
    ]
    start, end = random.choice(corners)

    grid = [[3] * GRID_SIZE for _ in range(GRID_SIZE)]  # 3 = visible wall

    # DFS to carve paths
    visited = set()
    def carve(r, c):
        visited.add((r, c))
        grid[r][c] = 0

        # Shuffle directions
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        random.shuffle(dirs)

        for dr, dc in dirs:
            nr, nc = r + 2*dr, c + 2*dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                if (nr, nc) not in visited:
                    # Carve through intermediate cell as well
                    grid[r+dr][c+dc] = 0
                    carve(nr, nc)

    # Start carving from (1, 1) or a random cell
    carve(1, 1)

    # Ensure start and end are walkable
    grid[start[0]][start[1]] = 1
    grid[end[0]][end[1]] = 2

    # Make surrounding cells adjacent to start and end walkable to prevent blockage
    for r, c in [start, end]:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                grid[nr][nc] = 0

    # Loop injection: randomly turn some walls into walkable paths
    walls = [(r, c) for r in range(1, GRID_SIZE-1) for c in range(1, GRID_SIZE-1) if grid[r][c] == 3]
    random.shuffle(walls)
    for i in range(min(loops, len(walls))):
        r, c = walls[i]
        grid[r][c] = 0

    grid[start[0]][start[1]] = 1
    grid[end[0]][end[1]] = 2

    return grid, start, end

# ── Load Models ──
def load_checkpoints():
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints")

    # Check if files exist or if we need to search for alternative names
    tess_path = os.path.join(ckpt_dir, "tesseract_v5.pt")
    if not os.path.exists(tess_path):
        tess_path = os.path.join(ckpt_dir, "maze_tesseract_v5.pt")

    ce_path = os.path.join(ckpt_dir, "baseline_ce_v3.pt")
    if not os.path.exists(ce_path):
        ce_path = os.path.join(ckpt_dir, "maze_baseline_ce_v3.pt")

    tess_model = LabyrinthTransformer()
    tess_model.load_state_dict(torch.load(tess_path, map_location=DEVICE))
    tess_model.eval()

    ce_model = LabyrinthTransformer()
    ce_model.load_state_dict(torch.load(ce_path, map_location=DEVICE))
    ce_model.eval()

    return tess_model, ce_model

def run_evaluation_on_maze(grid, start, end, tess_model, ce_model):
    bfs_path = bfs_shortest_path(grid, start, end)

    tess_path, tess_limit = solve_autoregressive(tess_model, grid, start, end, DEVICE)
    tess_reached = tess_path[-1] == end

    ce_path, ce_limit = solve_autoregressive(ce_model, grid, start, end, DEVICE)
    ce_reached = ce_path[-1] == end

    return {
        "grid": grid,
        "start": start,
        "end": end,
        "bfs_path": bfs_path,
        "tess_path": tess_path,
        "tess_reached": tess_reached,
        "tess_limit": tess_limit,
        "ce_path": ce_path,
        "ce_reached": ce_reached,
        "ce_limit": ce_limit
    }

def main():
    print("Loading models...")
    tess_model, ce_model = load_checkpoints()

    # 1. Export Adversarial Mazes
    print("Evaluating 6 Adversarial Mazes...")
    adversarial_builders = [
        ("The Detour", build_maze_1_detour, (0, 0), (9, 9)),
        ("The Spiral", build_maze_2_spiral, (0, 0), (5, 5)),
        ("The False Shortcut", build_maze_3_false_shortcut, (0, 0), (9, 9)),
        ("The Zigzag", build_maze_4_zigzag, (0, 0), (9, 0)),
        ("The Bottleneck", build_maze_5_bottleneck, (0, 0), (9, 9)),
        ("The U-Turn", build_maze_6_uturn, (0, 0), (0, 9)),
    ]

    adv_data = []
    for name, builder, start, end in adversarial_builders:
        grid = builder()
        res = run_evaluation_on_maze(grid, start, end, tess_model, ce_model)
        res["name"] = name
        adv_data.append(res)

    out_dir = os.path.join(os.path.dirname(__file__), "exported_data")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "adversarial_results.json"), "w") as f:
        json.dump(adv_data, f, indent=2)
    print("Saved adversarial_results.json")

    # 2. Export Test Dataset
    print("Generating and evaluating 10 Test Mazes...")
    test_data = []
    seed_base = 54321
    for i in range(10):
        # generate a solvable maze
        grid = None
        bfs_path = None
        start = end = None
        attempts = 0
        while bfs_path is None and attempts < 100:
            grid, start, end = generate_procedural_maze(seed_base + i * 100 + attempts, loops=random.randint(4, 8))
            bfs_path = bfs_shortest_path(grid, start, end)
            attempts += 1

        res = run_evaluation_on_maze(grid, start, end, tess_model, ce_model)
        res["name"] = f"Test Maze {i + 1}"
        test_data.append(res)

    with open(os.path.join(out_dir, "test_dataset_results.json"), "w") as f:
        json.dump(test_data, f, indent=2)
    print("Saved test_dataset_results.json")

    # 3. Export Training History
    print("Exporting Training History...")
    history = {
        "runs": [
            {
                "version": "v1",
                "name": "Plain CE Baseline",
                "epochs": 100,
                "train_mazes": 120,
                "test_mazes": 30,
                "success_rate": 60.0,
                "optimal_rate": 50.0,
                "efficiency": 0.0,
                "loss_type": "Plain Cross-Entropy",
                "notes": "Learns basic corridor-following but fails at critical fork decisions due to equal weight dilution."
            },
            {
                "version": "v2",
                "name": "Tesseract Loss (Sparse)",
                "epochs": 100,
                "train_mazes": 120,
                "test_mazes": 30,
                "success_rate": 66.67,
                "optimal_rate": 40.0,
                "efficiency": 0.0,
                "loss_type": "Tesseract Structured Loss",
                "notes": "Loss structure alone improves success rate by 6.7% but exhibits a tradeoff with path optimality at low data scale."
            },
            {
                "version": "v3",
                "name": "All-Position BFS (Dense)",
                "epochs": 80,
                "train_mazes": 300,
                "test_mazes": 30,
                "success_rate": 83.33,
                "optimal_rate": 33.33,
                "efficiency": 0.0,
                "loss_type": "Tesseract Structured Loss",
                "notes": "Denser multi-direction training. Plain CE baseline collapses to 36.67% on this distribution, highlighting Tesseract's robustness."
            },
            {
                "version": "v4",
                "name": "Extended Training",
                "epochs": 240,
                "train_mazes": 300,
                "test_mazes": 30,
                "success_rate": 100.0,
                "optimal_rate": 53.0,
                "efficiency": 88.0,
                "loss_type": "Tesseract Structured Loss",
                "notes": "Extended training and Cosine Annealing allows the model to digest structural signal and achieve perfect test success."
            },
            {
                "version": "v5",
                "name": "Second Seed (Final)",
                "epochs": 240,
                "train_mazes": 300,
                "test_mazes": 30,
                "success_rate": 100.0,
                "optimal_rate": 60.0,
                "efficiency": 90.0,
                "loss_type": "Tesseract Structured Loss",
                "notes": "Continued training on a second seed resolves the optimality tradeoff, raising both optimal rate and efficiency."
            }
        ],
        "stress_test": {
            "tesseract_success": 99.0,
            "ce_baseline_success": 21.8,
            "seeds": [999, 2024, 7777, 31415, 54321],
            "num_mazes": 500
        },
        "per_cell_val_accuracy_v4": {
            "early-corridor": 50.4,
            "early-fork": 58.4,
            "mid-corridor": 62.6,
            "mid-fork": 73.3,
            "endgame-corridor": 75.5,
            "endgame-fork": 69.8
        }
    }

    with open(os.path.join(out_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print("Saved training_history.json")
    print("Export complete!")

if __name__ == "__main__":
    main()
