To design a Sudoku puzzle where the board consists of 81 squares, some of which are initially filled with digits from 1 to 9. 
The puzzle is to fill in all the remaining squares such that no digit appears twice in any row, column, or 3 × 3 box. A row, column, or box is called a unit.

#a). To represent the Sudoku problem as a Constraint Satisfaction Problem (CSP), we need to define its variables, domains, and constraints.

Variables:

Each of the 81 cells in the 9x9 Sudoku grid is a variable. We can denote them as
𝑉𝑖,𝑗  where 𝑖 and 𝑗 range from 0 to 8, representing row and column indices.

Domains:

For each variable
𝑉𝑖,𝑗 :

If the cell is pre-filled with a digit
𝑘 (where 𝑘 is from 1 to 9), then the domain is 𝐷𝑖,𝑗 ={𝑘}

If the cell is empty, the domain is
𝐷𝑖,𝑗={1,2,3,4,5,6,7,8,9}

Constraints: The Sudoku rules impose the following:

All variables in the same row must have distinct values:
𝑉𝑖,𝑗≠𝑉𝑖,𝑘 for all
𝑗≠𝑘.

All variables in the same column must have distinct values:
𝑉𝑖,𝑗≠𝑉𝑚,𝑗
  for all 𝑖≠𝑚.


All variables in the same 3x3 box must have distinct values. For a box starting at
(𝑟,𝑐)
 where
𝑟 and 𝑐 are multiples of 3,
𝑉𝑖,𝑗≠𝑉𝑚,𝑛

  for all
𝑖,𝑗,𝑚,𝑛 in the box where
(𝑖,𝑗)≠(𝑚,𝑛).

This CSP formulation ensures that the solution satisfies the standard Sudoku constraints.

#b). Implement the Problem Using Backtracking Search

We’ll implement a basic backtracking search to solve a Sudoku puzzle and measure the average time over 10 runs.

Assumption : Same Initial Grid for the entire problem to maintain consistency while analysing the run times.

Here’s the Sudoku puzzle to use:


5 3 0  0 7 0  0 0 0

6 0 0  1 9 5  0 0 0

0 9 8  0 0 0  0 6 0

8 0 0  0 6 0  0 0 3

4 0 0  8 0 3  0 0 1

7 0 0  0 2 0  0 0 6

0 6 0  0 0 0  2 8 0

0 0 0  4 1 9  0 0 5

0 0 0  0 8 0  0 7 9


The code will:

Define a validity check function.

Implement backtracking to solve the puzzle.

Measure the time over 10 runs.
"""

import time
import copy

# Sudoku grid (0 represents empty cells)
initial_grid = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

def is_valid(grid, row, col, val):
    # Check row
    if val in grid[row]:
        return False
    # Check column
    if val in [grid[i][col] for i in range(9)]:
        return False
    # Check 3x3 box
    box_row, box_col = (row // 3) * 3, (col // 3) * 3
    for i in range(box_row, box_row + 3):
        for j in range(box_col, box_col + 3):
            if grid[i][j] == val:
                return False
    return True

def solve_sudoku(grid):
    for row in range(9):
        for col in range(9):
            if grid[row][col] == 0:
                for val in range(1, 10):
                    if is_valid(grid, row, col, val):
                        grid[row][col] = val
                        if solve_sudoku(grid):
                            return True
                        grid[row][col] = 0
                return False
    return True

# Measure average time over 10 runs
times = []
for _ in range(10):
    grid_copy = [row[:] for row in initial_grid]
    start_time = time.time()
    solve_sudoku(grid_copy)
    end_time = time.time()
    times.append(end_time - start_time)
    print(f"Run: {len(times)}, Time Taken: {end_time - start_time} seconds")

avg_time = sum(times) / len(times)
print(f"Standard Backtracking - Average time over 10 runs: {avg_time:.6f} seconds")

# Print the solved grid (from the last run)
print("Solved Sudoku:")
for row in grid_copy:
    print(row)

"""#c).	Analyse how different fault finding algorithms such as Forward Checking, Arc consistency improve the computational time of backtracking search?

We’ll implement backtracking with Forward Checking (FC) and Arc Consistency (AC-3), then compare their performance to standard backtracking.

Forward Checking:

After assigning a value, remove it from the domains of related unassigned cells. Backtrack if any domain becomes empty.

Arc Consistency:

Use AC-3 to ensure that for every value in a variable’s domain, there’s a consistent value in the domains of constrained variables.
"""

import time
import copy
from collections import deque

initial_grid = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

def get_related_cells(row, col):
    related = set()
    # Row
    for c in range(9):
        if c != col:
            related.add((row, c))
    # Column
    for r in range(9):
        if r != row:
            related.add((r, col))
    # Box
    box_row, box_col = (row // 3) * 3, (col // 3) * 3
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if (r, c) != (row, col):
                related.add((r, c))
    return related

# Forward Checking
def initialize_domains_fc(grid):
    domains = [[set(range(1, 10)) if grid[r][c] == 0 else {grid[r][c]} for c in range(9)] for r in range(9)]
    for r in range(9):
        for c in range(9):
            if grid[r][c] != 0:
                val = grid[r][c]
                for rr, cc in get_related_cells(r, c):
                    domains[rr][cc].discard(val)
    return domains

def solve_with_fc(grid, domains):
    for row in range(9):
        for col in range(9):
            if grid[row][col] == 0:
                for val in list(domains[row][col]):
                    grid[row][col] = val
                    removed = []
                    for r, c in get_related_cells(row, col):
                        if grid[r][c] == 0 and val in domains[r][c]:
                            domains[r][c].remove(val)
                            removed.append((r, c, val))
                    if any(len(domains[r][c]) == 0 for r, c in get_related_cells(row, col) if grid[r][c] == 0):
                        for r, c, v in removed:
                            domains[r][c].add(v)
                        grid[row][col] = 0
                        continue
                    if solve_with_fc(grid, domains):
                        return True
                    for r, c, v in removed:
                        domains[r][c].add(v)
                    grid[row][col] = 0
                return False
    return True

# Arc Consistency (AC-3)
def revise(domains, xi, xj):
    revised = False
    xi_r, xi_c = xi
    xj_r, xj_c = xj
    if len(domains[xj_r][xj_c]) == 1:
        val = next(iter(domains[xj_r][xj_c]))
        if val in domains[xi_r][xi_c]:
            domains[xi_r][xi_c].remove(val)
            revised = True
    return revised

def ac3(domains):
    queue = deque()
    for r in range(9):
        for c1 in range(9):
            for c2 in range(c1 + 1, 9):
                queue.append(((r, c1), (r, c2)))
                queue.append(((r, c2), (r, c1)))
        for c in range(9):
            for r1 in range(9):
                for r2 in range(r1 + 1, 9):
                    queue.append(((r1, c), (r2, c)))
                    queue.append(((r2, c), (r1, c)))
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            cells = [(br + i, bc + j) for i in range(3) for j in range(3)]
            for i in range(9):
                for j in range(i + 1, 9):
                    queue.append((cells[i], cells[j]))
                    queue.append((cells[j], cells[i]))
    while queue:
        (xi_r, xi_c), (xj_r, xj_c) = queue.popleft()
        if revise(domains, (xi_r, xi_c), (xj_r, xj_c)):
            if len(domains[xi_r][xi_c]) == 0:
                return False
            for r, c in get_related_cells(xi_r, xi_c):
                if (r, c) != (xj_r, xj_c):
                    queue.append(((r, c), (xi_r, xi_c)))
    return True

def solve_with_ac(grid, domains):
    for row in range(9):
        for col in range(9):
            if grid[row][col] == 0:
                for val in list(domains[row][col]):
                    grid[row][col] = val
                    domains_copy = [[set(d) for d in row] for row in domains]
                    domains_copy[row][col] = {val}
                    for r, c in get_related_cells(row, col):
                        if grid[r][c] == 0:
                            domains_copy[r][c].discard(val)
                    if ac3(domains_copy):
                        if solve_with_ac(grid, domains_copy):
                            return True
                    grid[row][col] = 0
                return False
    return True

# Timing measurements
methods = {
    "Forward Checking": lambda g: solve_with_fc(g, initialize_domains_fc(g)),
    "Arc Consistency": lambda g: solve_with_ac(g, initialize_domains_fc(g))  # Using FC domains initially
}

for name, solver in methods.items():
    print(name)
    times = []
    for _ in range(10):
        grid_copy = [row[:] for row in initial_grid]
        start_time = time.time()
        solver(grid_copy)
        end_time = time.time()
        times.append(end_time - start_time)
        print(f"Run: {len(times)}, Time Taken: {end_time - start_time} seconds")
    avg_time = sum(times) / len(times)
    print(f"{name} - Average time over 10 runs: {avg_time:.6f} seconds")
    # Optionally print the solved grid (from the last run)
    print("Solved Sudoku:")
    for row in grid_copy:
      print(row)

"""#d). Analyse how different Heuristics MRV (Minimum Remaining Values), Degree heuristic, Least Constraining Value affect the  computational time of backtracking search?

We’ll implement MRV, Degree Heuristic, and LCV on top of forward checking.

MRV:

Choose the variable with the smallest domain.

Degree Heuristic:

Break MRV ties by selecting the variable with the most unassigned neighbors.

LCV:

Order values by least impact on neighbors’ domains.
"""

import time
import copy
from collections import defaultdict

initial_grid = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

def get_related_cells(row, col):
    related = set()
    for c in range(9):
        if c != col:
            related.add((row, c))
    for r in range(9):
        if r != row:
            related.add((r, col))
    box_row, box_col = (row // 3) * 3, (col // 3) * 3
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if (r, c) != (row, col):
                related.add((r, c))
    return related

def initialize_domains(grid):
    domains = [[set(range(1, 10)) if grid[r][c] == 0 else {grid[r][c]} for c in range(9)] for r in range(9)]
    for r in range(9):
        for c in range(9):
            if grid[r][c] != 0:
                val = grid[r][c]
                for rr, cc in get_related_cells(r, c):
                    domains[rr][cc].discard(val)
    return domains

def count_unassigned_neighbors(grid, row, col):
    return sum(1 for r, c in get_related_cells(row, col) if grid[r][c] == 0)

def get_mrv_variable(grid, domains):
    min_size = 10
    candidates = []
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                size = len(domains[r][c])
                if size < min_size:
                    min_size = size
                    candidates = [(r, c)]
                elif size == min_size:
                    candidates.append((r, c))
    return min(candidates, key=lambda x: count_unassigned_neighbors(grid, *x)) if candidates else None

def get_lcv_values(grid, domains, row, col):
    value_counts = []
    for val in domains[row][col]:
        count = sum(1 for r, c in get_related_cells(row, col) if grid[r][c] == 0 and val in domains[r][c])
        value_counts.append((count, val))
    return [val for _, val in sorted(value_counts)]

def solve_with_heuristics(grid, domains, use_mrv=True, use_degree=True, use_lcv=False):
    empty = get_mrv_variable(grid, domains) if use_mrv else next(((r, c) for r in range(9) for c in range(9) if grid[r][c] == 0), None)
    if not empty:
        return True
    row, col = empty
    values = get_lcv_values(grid, domains, row, col) if use_lcv else domains[row][col]
    for val in values:
        grid[row][col] = val
        removed = []
        for r, c in get_related_cells(row, col):
            if grid[r][c] == 0 and val in domains[r][c]:
                domains[r][c].remove(val)
                removed.append((r, c, val))
        if solve_with_heuristics(grid, domains, use_mrv, use_degree, use_lcv):
            return True
        for r, c, v in removed:
            domains[r][c].add(v)
        grid[row][col] = 0
    return False

# Timing measurements
configs = {
    "MRV": (True, False, False),
    "MRV + Degree": (True, True, False),
    "MRV + LCV": (True, False, True)
}

for name, (mrv, degree, lcv) in configs.items():
    times = []
    print(name)
    for _ in range(10):
        grid_copy = [row[:] for row in initial_grid]
        domains = initialize_domains(grid_copy)
        start_time = time.time()
        solve_with_heuristics(grid_copy, domains, mrv, degree, lcv)
        end_time = time.time()
        times.append(end_time - start_time)
        print(f"Run: {len(times)}, Time Taken: {end_time - start_time} seconds")
    avg_time = sum(times) / len(times)
    print(f"{name} - Average time over 10 runs: {avg_time:.6f} seconds")

        # Optionally print the solved grid (from the last run)
    print("Solved Sudoku:")
    for row in grid_copy:
      print(row)
