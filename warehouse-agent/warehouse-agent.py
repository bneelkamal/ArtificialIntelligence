import numpy as np
import random
from heapq import heappush, heappop
import textwrap
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import re

# Constants for cell types
EMPTY = "EMPTY"
PACKAGE = "PACKAGE"
DROPOFF = "DROPOFF"
OBSTACLE = "OBSTACLE"
AGENT = "AGENT"
PICKED = "PICKED"
DROPPED = "DROPPED"

class Warehouse:
    def __init__(self, n, m, num_packages, num_obstacles, start_pos):
        self.n = n
        self.m = m
        self.num_packages = num_packages
        self.num_obstacles = num_obstacles
        self.start_pos = start_pos

        # Initialize grid with EMPTY cells
        self.grid = np.full((n, m), EMPTY, dtype='<U16')

        # Place agent at starting position
        self.agent_pos = start_pos
        self.grid[start_pos] = AGENT

        # Maintain mappings between packages and drop-offs
        self.package_locations = []
        self.dropoff_locations = []

        # Place obstacles randomly
        self.obstacle_locations = []

        # Setup warehouse grid
        self._setup_warehouse()

    def _setup_warehouse(self):
        positions = [(i, j) for i in range(self.n) for j in range(self.m) if (i, j) != self.start_pos]
        random.shuffle(positions)

        # Place packages and drop-offs ensuring no overlap
        for i in range(1, self.num_packages + 1):
            package_pos = positions.pop()
            dropoff_pos = positions.pop()

            self.package_locations.append(package_pos)
            self.dropoff_locations.append(dropoff_pos)

            self.grid[package_pos] = PACKAGE + str(i)
            self.grid[dropoff_pos] = DROPOFF + str(i)

        # Place obstacles ensuring no overlap with packages/drop-offs
        for _ in range(self.num_obstacles):
            obstacle_pos = positions.pop()
            self.obstacle_locations.append(obstacle_pos)
            self.grid[obstacle_pos] = OBSTACLE

    def display(self):
        print("\nWarehouse Grid:")
        print(self.grid)
        visualize_grid(self.grid)

def visualize_grid(grid):
    color_map = {
        "EMPTY": "white",
        "PACKAGE": "green",
        "DROPOFF": "orange",
        "OBSTACLE": "pink",
        "AGENT": "black",
        "PICKED": "yellow",
        "DROPPED": "cyan"
    }

    n, m = grid.shape
    fig, ax = plt.subplots(figsize=(m, n))

    # Adjust grid lines to match cell boundaries
    ax.set_xticks(np.arange(0, m + 1, 1))
    ax.set_yticks(np.arange(0, n + 1, 1))
    ax.grid(which='both', color='black', linestyle='-', linewidth=1)

    for i in range(n):
        for j in range(m):
            cell = grid[i, j]

            # Get color based on cell type prefix
            match = re.match(r"(PACKAGE|DROPOFF|PICKED|DROPPED)", cell)
            if match:
                if  "AGENT" in cell:
                  color = "gray"
                else:
                  color_key = match.group(1)  # Get the matched cell type
                  color = color_map.get(color_key, "gray")
            else:
                color_key = cell
                color = color_map.get(color_key, "gray")

            rect = plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor='black', linewidth=1)  # Add edgecolor
            ax.add_patch(rect)

            # Wrap the text using textwrap.fill
            wrapped_text = textwrap.fill(cell, width=10)

            # Text color contrast
            text_color = "black" if color != "black" else "white"
            ax.text(j + 0.5, i + 0.5, wrapped_text , ha="center", va="center", color=text_color, fontsize=8)

    ax.set_xlim(0, m)
    ax.set_ylim(0, n)
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()  # Invert y-axis to match grid orientation

    plt.show()


def ucs(grid, start, goal, obstacle_locations):
    n, m = grid.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    frontier = [(0, start)]  # (cost, position)
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        current_cost, current = heappop(frontier)

        if current == goal:
            break

        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < n and 0 <= ny < m:
                neighbor = (nx, ny)
                move_cost = 6 if neighbor in obstacle_locations else 1
                new_cost = current_cost + move_cost
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    heappush(frontier, (new_cost, neighbor))
                    came_from[neighbor] = current

    # Reconstruct path
    if goal not in came_from:
        return None
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path

class Agent:
    def __init__(self, warehouse):
        self.wh = warehouse
        self.pos = warehouse.start_pos
        self.cost = 0
        self.reward = 0

    def move_step(self, next_pos):
      x, y = next_pos

      # Clear previous agent position unless it's a special cell
      px, py = self.pos
      # Remove "AGENT" from the previous position if present
      if "AGENT" in self.wh.grid[px][py]:
          # If "AGENT" is the only thing in the cell, make it EMPTY
          if self.wh.grid[px][py] == "AGENT":
              self.wh.grid[px][py] = EMPTY
          # Otherwise, remove "AGENT" from the cell content
          else:
              self.wh.grid[px][py] = self.wh.grid[px][py].replace(" + AGENT", "").strip()

      # Update cost based on terrain
      move_cost = 6 if (x, y) in self.wh.obstacle_locations else 1
      self.cost += move_cost
      if move_cost == 6:
          print(f"AGENT Moved through obstacle at {(x, y)}! Cost += 6. (Penalty:5 + Move_Cost:1)")
      else:
          print(f"AGENT Moved to {(x, y)}. Cost += 1.")

      self.wh.grid[x][y] = self.wh.grid[x][y] + " + AGENT" if self.wh.grid[x][y] != EMPTY else AGENT

      self.pos = next_pos

    def nearest_package(self):
        min_cost = float('inf')
        nearest = None
        nearest_path = None

        for pkg in self.wh.package_locations:
            path = ucs(self.wh.grid, self.pos, pkg, self.wh.obstacle_locations)
            if path:
                path_cost = sum(6 if step in self.wh.obstacle_locations else 1 for step in path[1:])
                if path_cost < min_cost:
                    min_cost = path_cost
                    nearest = pkg
                    nearest_path = path

        return nearest, nearest_path

    def deliver_all_packages(self):
        while self.wh.package_locations:
            pkg, path_to_pkg = self.nearest_package()

            if not pkg:
                print("No reachable packages remaining.")
                break

            print(f"\n AGENT Moving to nearest PACKAGE at {pkg} with path {path_to_pkg}")

            # Move to package
            for step in path_to_pkg[1:]:
                self.move_step(step)
                self.wh.display()
                print(f"Path taken: {path_to_pkg}")  # Display path here

            # Update grid to PICKED
            px, py = pkg
            # Get the package number
            package_number = "".join(filter(str.isdigit, self.wh.grid[px][py]))
            self.wh.grid[px][py] = PICKED + package_number  # Concatenate PICKED with package number

            print(f"AGENT Picked up PACKAGE{package_number} at {pkg}.")
            self.wh.display()

            pkg_index = self.wh.package_locations.index(pkg)
            drop = self.wh.dropoff_locations[pkg_index]

            path_to_drop = ucs(self.wh.grid, self.pos, drop, self.wh.obstacle_locations)
            if not path_to_drop:
                print(f"No path to drop-off at {drop}. Aborting.")
                break

            print(f"AGENT Delivering PACKAGE{package_number} to {drop} with path {path_to_drop}")

            # Move to drop-off
            for step in path_to_drop[1:]:
                self.move_step(step)
                self.wh.display()
                print(f"Path taken: {path_to_drop}")  # Display path here

            # Update grid to DROPPED
            dx, dy = drop
            # Get the package number
            package_number = "".join(filter(str.isdigit, self.wh.grid[dx][dy]))

            self.wh.grid[dx][dy] = DROPPED + package_number  # Concatenate Dropped with package number

            print(f"PACKAGE{package_number} delivered successfully! Reward +10.")
            self.wh.display()

            # Update reward and remove package/drop-off
            self.wh.package_locations.remove(pkg)
            self.wh.dropoff_locations.remove(drop)
            self.reward += 10

        self.wh.display()

    def final_score(self):
        return self.reward - self.cost

def main():
    random.seed(42)  # reproducibility

    n, m = 4, 4
    num_packages = 3
    num_obstacles = 5
    start = (0, 0)

    # Check for sufficient space before setup
    if (num_packages * 2) + num_obstacles > (n * m) - 1:
            print(
                "Error: Not enough space in the warehouse for the specified number of "
                "packages, drop-offs, and obstacles. Please adjust the input parameters."
       )
    else:
      global warehouse
      warehouse = Warehouse(n, m, num_packages, num_obstacles, start)

      global grid
      grid = warehouse.grid

      print("\nInitial Warehouse Configuration:")
      warehouse.display()



      print("\nPackage Locations:", warehouse.package_locations)
      print("Drop-Off Locations:", warehouse.dropoff_locations)

      agent = Agent(warehouse)

      print("\nStarting package delivery...")
      agent.deliver_all_packages()

      score = agent.final_score()

      print("\nFinal Results:")
      print(f"Total Movement Cost: {agent.cost}")
      print(f"Total Delivery Reward: {agent.reward}")
      print(f"Final Score: {score}")

if __name__ == "__main__":
  main()
