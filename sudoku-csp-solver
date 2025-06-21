# Sudoku Solver using Constraint Satisfaction Problem Techniques

This project explores solving a 9x9 Sudoku puzzle by formulating it as a Constraint Satisfaction Problem (CSP). It implements and compares several algorithms and heuristics to find the most efficient solution, starting from a baseline backtracking search and progressively adding optimizations.

## 1. Sudoku as a Constraint Satisfaction Problem (CSP)

Sudoku puzzle is formally defined as a CSP with the following components:

> *   **Variables:** Each of the 81 cells in a 9x9 grid, denoted \( V_{i,j} \), where \( i \) and \( j \) range from 0 to 8 (representing row and column indices).
> *   **Domains:**
>     *   Pre-filled cells have a domain of a single value, e.g., \( D_{i,j} = \{k\} \) where \( k \) is the given digit.
>     *   Empty cells have a domain \( D_{i,j} = \{1, 2, 3, 4, 5, 6, 7, 8, 9\} \).
> *   **Constraints:**
>     *   **Row:** \( V_{i,j} ≠ V_{i,k} \) for all \( j ≠ k \).
>     *   **Column:** \( V_{i,j} ≠ V_{m,j} \) for all \( i ≠ m \).
>     *   **Box:** All variables in each 3x3 subgrid must be distinct.

## 2. Algorithms and Heuristics Implemented

The project analyzes the performance of different CSP solving techniques on a consistent initial Sudoku grid.

### a. Standard Backtracking Search

This is the baseline implementation. It uses a simple recursive, depth-first approach to explore the search space.

*   **Analysis:** While it correctly solves the puzzle, this method is inefficient because it only checks constraints *after* making an assignment, often exploring deep into invalid search branches before backtracking.

### b. Fault-Finding Algorithms

These algorithms prune the search space by detecting conflicts earlier.

*   **Forward Checking (FC):** After assigning a value to a cell, FC removes that value from the domains of all unassigned neighboring cells (in the same row, column, and box). This helps detect failure much earlier if a neighbor's domain becomes empty.
*   **Arc Consistency (AC-3):** A more powerful technique that enforces consistency across all variables. It ensures that for every value in a variable's domain, there is a consistent value in the domains of its constrained neighbors.

**Performance Comparison:**

| Method |
| :--- |
| Standard Backtracking |
| **Forward Checking** |
| Arc Consistency |

> **Key Insight:** Forward Checking provides a significant performance boost by pruning the search space locally. In contrast, the overhead of enforcing global Arc Consistency for this puzzle is too high, making it slower than even the standard backtracking approach.

### c. Search Heuristics (with Forward Checking)

These heuristics intelligently guide the search by selecting which variable to assign next and which value to try first.

*   **Minimum Remaining Values (MRV):** Chooses the variable (cell) with the smallest domain. This "fail-first" strategy minimizes the branching factor.
*   **Degree Heuristic:** Used as a tie-breaker for MRV. It selects the variable involved in the largest number of constraints on other unassigned variables.
*   **Least Constraining Value (LCV):** After selecting a variable, this heuristic reorders the values in its domain, trying first the value that rules out the fewest choices for its neighbors.

**Performance Comparison (All with Forward Checking):**

| Method |
| :--- |
| **MRV** |
| MRV + Degree |
| MRV + LCV |

> **Key Insight:** The MRV heuristic, combined with Forward Checking, offers the best performance, providing a ~90.6% improvement over the baseline. The Degree and LCV heuristics add computational overhead without providing a proportional benefit for this specific puzzle, making them slightly slower than MRV alone.

## 3. How to Run

1.  Ensure you have Python installed. No external libraries are needed beyond the standard library (`time`, `copy`, `collections`).
2.  Run the script from your terminal:    
3.  The script will execute the Sudoku solver first, printing the average run times for each implemented technique over 10 runs.

## 4. Conclusion

For this moderately difficult Sudoku puzzle, lightweight optimizations provide the best results. Standard backtracking is a viable but slow baseline. **Forward Checking** offers a major speedup by preventing exploration of invalid paths. The most efficient approach combines **Forward Checking with the MRV heuristic**, which intelligently guides the search toward the most constrained parts of the problem, leading to the fastest solution time.
