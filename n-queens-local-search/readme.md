# N-Queens Solver using Simulated Annealing and Hill Climbing

This project solves the **N-Queens problem** for N=8 using local search algorithms. It implements and contrasts **Simulated Annealing (SA)** and **Hill Climbing (HC)** to demonstrate their different approaches to optimization and their effectiveness in finding a conflict-free solution on an 8x8 chessboard.

## 1. Problem Introduction

> The N-Queens problem is a classic combinatorial optimization problem where the goal is to place N queens on an N×N chessboard such that no two queens threaten each other. This means no two queens can share the same row, column, or diagonal. For this assignment, we chose N=8, a common and challenging instance of the problem[2][3].

## 2. Algorithms and Results

### a. Simulated Annealing (SA)

Simulated Annealing is a probabilistic algorithm that explores the search space by accepting both improving moves and, occasionally, worsening moves. This ability allows it to escape local optima.

**Algorithm Details:**
*   **Initialization:** Start with a random placement of queens, one per column.
*   **Neighbor Generation:** Generate a new state by randomly moving one queen to a different row in its column.
*   **Acceptance Probability:** A move that results in a change of conflicts, Δ, is accepted with probability \( P = e^{-\Delta/T} \). Worsening moves (Δ > 0) are less likely to be accepted as the temperature, T, cools.
*   **Cooling Schedule:** The temperature `T` is gradually reduced using a cooling factor `alpha = 0.99`.

**Results:**
*   **Analysis:** SA successfully found a solution with zero conflicts. Its ability to accept worse moves early on allowed it to navigate out of local minima and explore the search space effectively before converging to a global optimum.

### b. Hill Climbing (HC)

Hill Climbing is a greedy local search algorithm that only ever accepts moves that improve the current state. In this project, it is implemented by modifying the SA algorithm.

**Modification:**
To transform Simulated Annealing into Hill Climbing, the temperature is set to **T = 0**.

**Results:**
*   **Analysis:** The Hill Climbing algorithm failed to find a conflict-free solution. It quickly improved the board state to a point where no single queen move could further reduce the number of conflicts, trapping it in a suboptimal state.

## 3. Theoretical Justification and Behavioral Differences

### Why does T=0 create a Hill Climber?

The acceptance probability in Simulated Annealing is \( P = e^{-\Delta/T} \). When we set the temperature `T` to 0, this formula changes drastically.

> When \( T \to 0 \), the acceptance probability for a worsening move (where Δ > 0) becomes:
> \[ P(\text{accept}) = e^{-\Delta/0} \to e^{-\infty} \to 0 \]
> This means that the probability of accepting a move that increases conflicts becomes zero. Only moves that are better (Δ < 0) or equal (Δ = 0) will be accepted. Our implementation only accepts strictly better moves (Δ < 0) for HC. This is the exact behavior of Hill Climbing.

### Behavioral Differences

*   **Simulated Annealing:** Balances *exploration* (accepting bad moves to discover new regions of the search space) with *exploitation* (accepting good moves to converge on a solution). This makes it robust and capable of finding global optima.
*   **Hill Climbing:** Is purely *exploitative*. It always moves towards the best immediate neighbor. This makes it fast but highly susceptible to getting trapped in local optima, as seen in the results.

## 4. How to Run

1.  Ensure you have Python installed. The script uses the `random` and `math` libraries, which are part of the standard library.
2.  Run the script from your terminal:
3.  The N-Queens solver will print the initial and final board states for both SA and HC, along with their final conflict counts and execution times.

## 5. Conclusion

This experiment clearly demonstrates that **Simulated Annealing is more effective than Hill Climbing** for solving the N-Queens problem. By intelligently managing the trade-off between exploration and exploitation, SA successfully found a perfect solution. In contrast, the purely greedy nature of HC led it directly into a local minimum, from which it could not escape.
