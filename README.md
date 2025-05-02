# Conway's Game of Life

A modular, efficient implementation of Conway's Game of Life in Python, using matrix operations and convolutions for fast computation.

## Features

- Efficient implementation using NumPy arrays and convolutions
- Modular architecture with separate game logic and visualization components
- Includes several pre-defined patterns (glider, blinker, block, beacon, Gosper glider gun)
- Customizable grid size, animation speed, and initial state
- Support for saving animations
- **NEW:** Agent-based mode where each live cell is an intelligent agent

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/game_of_life.git
   cd game_of_life
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the simulation with default settings:
```bash
python -m game_of_life.main
```

### Command-line Arguments

- `--grid-size N`: Set the grid size to N×N (default: 100)
- `--interval MS`: Set the animation update interval in milliseconds (default: 500, higher is slower)
- `--fill-ratio R`: Set the ratio of live cells for random initialization (default: 0.2)
- `--save FILENAME`: Save the animation to the specified file

### Game Modes

#### Standard Mode (Default)
The classic Conway's Game of Life with the standard rules:
1. Any live cell with 2 or 3 live neighbors survives
2. Any dead cell with exactly 3 live neighbors becomes alive
3. All other cells die or stay dead

#### Agent-Based Mode
In this mode, each live cell is an intelligent agent that can take actions:

- `--agent-based`: Enable agent-based mode
- `--random-actions`: Use random action selection for agents (default)
- `--show-agents`: Show agent actions in visualization (arrows and circles)

##### Agent Actions
Each live cell can take one of 9 possible actions:
1. **Grow** in one of 8 directions (N, NE, E, SE, S, SW, W, NW)
2. **Fortify** itself for protection

##### Agent-Based Rules
1. If 3+ cells choose to "grow" into a dead cell, that cell becomes alive
2. If a live cell is "grown into," it increases its effective neighbor count by 1
3. The "fortify" action makes a cell more protected:
   - It can survive with one more neighbor than usual (4 instead of 3)
   - It can survive with one less neighbor than usual (1 instead of 2)

### Pattern Options

- `--random`: Initialize with random cells (default)
- `--glider`: Initialize with a glider pattern
- `--gosper`: Initialize with a Gosper Glider Gun pattern
- `--blinker`: Initialize with a blinker pattern
- `--beacon`: Initialize with a beacon pattern
- `--block`: Initialize with a block pattern

### Examples

Run with a larger grid and slower animation:
```bash
python -m game_of_life.main --grid-size 200 --interval 800
```

Run with a Gosper Glider Gun pattern:
```bash
python -m game_of_life.main --gosper
```

Save an animation of a glider:
```bash
python -m game_of_life.main --glider --save glider.mp4
```

Run in agent-based mode with visualization:
```bash
python -m game_of_life.main --agent-based --show-agents
```

Run a glider in agent-based mode:
```bash
python -m game_of_life.main --agent-based --glider --show-agents
```

## Architecture

The code is organized into three main modules:

- `game_logic.py`: Contains the core game logic, including the `GameOfLife` and `AgentBasedGameOfLife` classes and the `Patterns` class
- `visualization.py`: Handles visualization and animation of the game state
- `main.py`: Serves as the entry point, parsing command-line arguments and setting up the simulation

## Extending the Project

To add new patterns, simply add new static methods to the `Patterns` class in `game_logic.py`.

To customize the visualization, modify the `Visualizer` class in `visualization.py`.

To implement more sophisticated agent behaviors, modify the `_choose_agent_action` method in the `AgentBasedGameOfLife` class.

## Requirements

- Python 3.6+
- NumPy
- Matplotlib
- SciPy

## License

This project is licensed under the MIT License - see the LICENSE file for details.
