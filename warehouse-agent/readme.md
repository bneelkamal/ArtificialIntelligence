# Project A: Dynamic Goal-Based Agent for Warehouse Logistics Optimization

This project implements a goal-based agent that controls a robot in a warehouse logistics scenario. The robot's mission is to efficiently pick up three packages and deliver them to their designated drop-off points while navigating a grid with obstacles[1]. The core of the agent's pathfinding logic is the **Uniform Cost Search (UCS)** algorithm, which ensures the robot finds the least-cost paths.

## 1. Problem Definition
The agent operates under the following conditions :

*   **Grid Environment**: A **4x4** warehouse grid.
*   **Mission**: Pick up **3 packages** and deliver them to **3 unique drop-off points**.
*   **Obstacles**: The grid contains **5 randomly placed obstacles**.
*   **Start Position**: The robot begins at `(0,0)`.
*   **Movement Costs**:
    *   Moving to an empty cell: **1 cost unit**.
    *   Moving through an obstacle: **6 cost units** (a 5-unit penalty + 1-unit move cost).
*   **Reward System**: A reward of **+10 points** for each successful delivery.
*   **Final Score**: Calculated as `Total Reward - Total Movement Cost`.
*   **Agent Logic**: The agent identifies and moves to the nearest package, picks it up, and then delivers it before proceeding to the next nearest package. It can only carry one package at a time.

## 2. Methodology: Using Uniform-Cost Search (UCS)

The pathfinding is driven by the UCS algorithm to ensure the robot always selects the most cost-efficient route. My implementation of UCS is explained below:

> It’s like a smart explorer that keeps track of all possible paths and picks the one with the lowest total cost. I used Python’s `heapq` module to make a priority queue, where each entry is a tuple of the cost so far and the position (like `(0, (0,0))` at the start). The queue always pops out the spot with the smallest cost to check next. From there, it looks at the four directions—up, down, left, right—and figures out the cost of moving to each neighbour. If it’s an empty cell, it adds 1 to the cost; if it’s an obstacle, it adds 6. It keeps a dictionary (`came_from`) to remember where it came from and another (`cost_so_far`) to track the cheapest cost to each spot.
>
> The cool thing about UCS is that it’s guaranteed to find the least-cost path if one exists... In my warehouse, this meant the robot could decide whether it’s worth going around an obstacle or taking the hit and moving through it if that’s cheaper overall.

## 3. How to Run the Simulation

The project is contained in a single Python script.

**Dependencies:**

pip install numpy matplotlib

**Execution:**

*Note: The script contains both Part A and Part B. Part A will run first.*

The script will print step-by-step console logs of the agent's actions and generate a `Matplotlib` visualization of the warehouse grid at each step of the delivery process. For reproducibility, the random seed is set to `42`.

## 4. Findings and Results

With the pre-set random seed, the agent successfully delivered all packages by navigating the warehouse, including strategically crossing two obstacles where it was more cost-effective than taking a longer path.
