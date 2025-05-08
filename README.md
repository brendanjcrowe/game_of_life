# Conway's Game of Life Evolutionary Game Theory Extension

This is a simple extension of Conway's Game of Life to include evolutionary game theory dynamics.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/game_of_life.git
   cd game_of_life
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
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
- `--use-policies`: Use policy-based behavior with evolutionary dynamics (default)
- `--show-agents`: Show agent actions in visualization (arrows and circles)
- `--show-policy-colors`: Show different colors for different policy types

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

#### Evolutionary Game Theory

This mode introduces competing agent policies that evolve over time:

- `--use-policies`: Enable evolutionary dynamics with different agent policies
- `--random-growth-ratio R`: Set the relative ratio of Random Growth policy (default: 1.0)
- `--always-fortify-ratio R`: Set the relative ratio of Always Fortify policy (default: 1.0)
- `--adaptive-ratio R`: Set the relative ratio of Adaptive policy (default: 1.0)

##### Agent Policies

1. **Random Growth (Red)**:
   - Randomly grow into a neighboring unoccupied space
   - If no unoccupied spaces, then fortify

2. **Always Fortify (Blue)**:
   - Always fortify regardless of surroundings

3. **Adaptive (Green)**:
   - If the agent has 2 or 3 neighbors, fortify for stability
   - Otherwise, grow into a random unoccupied space

##### Evolution Mechanism

When new cells are born, they inherit a policy from their parent cells:
- The probability of inheriting a particular policy is proportional to the representation of that policy among neighboring parent cells
- This creates an evolutionary dynamic where more successful policies gradually become more common

### Pattern Options

- `--random`: Initialize with random cells (default)
- `--glider`: Initialize with a glider pattern
- `--gosper`: Initialize with a Gosper Glider Gun pattern
- `--blinker`: Initialize with a blinker pattern
- `--beacon`: Initialize with a beacon pattern
- `--block`: Initialize with a block pattern

Note these patterns do not quite work with the agent-based mode.
### Examples
Note this implementation gets a bit slow with a large number of cells.
I reccomend running it at grid size 25, you can still see the bahavior at this size.

Run with a larger grid and slower animation:
```bash
python -m game_of_life.main --grid-size 25 --interval 100
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

Run with evolutionary dynamics and policy visualization:
```bash
python -m game_of_life.main --agent-based --use-policies --show-policy-colors
```

Run with more Always Fortify policies initially:
```bash
python -m game_of_life.main --agent-based --use-policies --always-fortify-ratio 3.0
```

## Architecture

The code is organized into four main modules:

- `game_logic.py`: Contains the core game logic, including the `GameOfLife` and `AgentBasedGameOfLife` classes and the `Patterns` class
- `policies.py`: Defines the agent policies and evolutionary dynamics
- `visualization.py`: Handles visualization and animation of the game state
- `main.py`: Serves as the entry point, parsing command-line arguments and setting up the simulation

## Extending the Project


To customize the visualization, modify the `Visualizer` class in `visualization.py`.

To implement more sophisticated agent behaviors, add new policy classes to the `policies.py` file.

## Requirements

- Python 3.6+
- NumPy
- Matplotlib
- SciPy

## License

This project is licensed under the MIT License - see the LICENSE file for details.
