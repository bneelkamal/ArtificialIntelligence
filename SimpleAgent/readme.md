# Simple Reflex Agent

## Overview

This project implements a **Simple Reflex Agent** that navigates a 2D grid-based environment, cleaning dirty locations as it moves. The agent follows a basic rule-based system to perceive and act on the environment.

## How It Works

- The environment is represented as a **NumPy** array with "DIRTY" and "CLEAN" locations.
- The agent starts at position **(0,0)**.
- It checks the current location:
  - If **DIRTY**, it cleans it.
  - If **CLEAN**, it moves to the next location (right first, then down).
- The process continues until all locations are clean.

## Installation & Requirements

Ensure you have Python installed along with **NumPy**:

```bash
pip install numpy
```

## Running the Agent

Run the script using:

```bash
python simple_reflex_agent.py
```

## Sample Output

```
Initial environment:
[['DIRTY' 'CLEAN' 'DIRTY']
 ['CLEAN' 'DIRTY' 'CLEAN']
 ['DIRTY' 'DIRTY' 'CLEAN']]

Location (0, 0) is DIRTY. Cleaning ...
...
All locations are clean. Stopping.
```

## Code Structure

- **SimpleReflexAgent**: The agent class that perceives and acts.
- **perceive()**: Checks if the current location is dirty.
- **act()**: Cleans if dirty; moves otherwise.
- **move()**: Moves right, then down if at the end of a row.
- **environment**: A 2D grid representing dirty and clean locations.

## Future Enhancements

- Introduce obstacles in the environment.
- Implement a goal-based agent.
- Add reinforcement learning for improved performance.

## License

This project is open-source and available under the MIT License.

