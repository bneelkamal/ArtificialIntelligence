Consider a simple N-Queens problem. Take a suitable value of N and write a Python code for simulated annealing to solve the problem. 
As we know, if we take T=0 in SA, it falls back to simple hill climbing search. Modify the SA algorithm in order to make it an HC solution. 
Theoretical justification in support of modification (e.g., calculation of the acceptance probability) and the behavioral difference between the two approaches.

We’ll use
𝑁 = 8 and implement Simulated Annealing (SA), then modify it to Hill Climbing (HC).
"""

import random
import math
import time

N = 8  # Board size (8x8)

def conflicts(state):
    """Calculate the number of queen conflicts in the current state."""
    conf = 0
    for i in range(N):
        for j in range(i + 1, N):
            if state[i] == state[j] or abs(state[i] - state[j]) == abs(i - j):
                conf += 1
    return conf

def get_neighbor(state):
    """Generate a neighboring state by moving one queen randomly."""
    new_state = state[:]
    i = random.randint(0, N - 1)  # Choose a column
    new_row = random.randint(0, N - 1)  # Choose a new row
    while new_row == state[i]:  # Ensure it's a different position
        new_row = random.randint(0, N - 1)
    new_state[i] = new_row
    return new_state

def print_board(state, title="Board"):
    """Display the N-Queens board with queens as 'Q' and empty cells as '.'."""
    print(f"\n{title}:")
    for row in range(N):
        line = ['.' for _ in range(N)]
        for col in range(N):
            if state[col] == row:
                line[col] = 'Q'
        print(' '.join(line))
    print()

def simulated_annealing(max_iter=10000, T_init=100, alpha=0.99, is_hc=False):
    """Solve N-Queens using Simulated Annealing or Hill Climbing based on is_hc flag."""
    state = [random.randint(0, N - 1) for _ in range(N)]
    title = "Initial State for Hill Climbing" if is_hc else "Initial State for Simulated Annealing"
    print_board(state, title)

    T = T_init if not is_hc else 0  # Set T=0 for Hill Climbing
    for _ in range(max_iter):
        if (not is_hc and T < 1e-5) or conflicts(state) == 0:
            break
        new_state = get_neighbor(state)
        delta = conflicts(new_state) - conflicts(state)
        if is_hc:
            # Hill Climbing: only accept if delta < 0 (strictly better)
            if delta < 0:
                state = new_state
        else:
            # Simulated Annealing: accept with probability min(1, e^(-delta/T))
            acceptance_prob = math.exp(-delta / T) if T > 0 and delta > 0 else 1
            if delta <= 0 or random.random() < acceptance_prob:
                state = new_state
            T *= alpha  # Cool down only for SA

    title = "Final State for Hill Climbing" if is_hc else "Final State for Simulated Annealing"
    print_board(state, title)
    return state, conflicts(state)

# Run and evaluate both algorithms
start = time.time()
sa_solution, sa_conflicts = simulated_annealing(is_hc=False)
sa_time = time.time() - start
print(f"Simulated Annealing Time: {sa_time:.6f}s, Conflicts: {sa_conflicts}")

start = time.time()
hc_solution, hc_conflicts = simulated_annealing(is_hc=True)
hc_time = time.time() - start
print(f"Hill Climbing Time: {hc_time:.6f}s, Conflicts: {hc_conflicts}")
