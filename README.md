# Conway's Game of Life

A modular, efficient implementation of Conway's Game of Life in Python, using matrix operations and convolutions for fast computation.

## Features

- Efficient implementation using NumPy arrays and convolutions
- Modular architecture with separate game logic and visualization components
- Includes several pre-defined patterns (glider, blinker, block, beacon, Gosper glider gun)
- Customizable grid size, animation speed, and initial state
- Support for saving animations

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

## Architecture

The code is organized into three main modules:

- `game_logic.py`: Contains the core game logic, including the `GameOfLife` class and the `Patterns` class
- `visualization.py`: Handles visualization and animation of the game state
- `main.py`: Serves as the entry point, parsing command-line arguments and setting up the simulation

## Extending the Project

To add new patterns, simply add new static methods to the `Patterns` class in `game_logic.py`.

To customize the visualization, modify the `Visualizer` class in `visualization.py`.

## Requirements

- Python 3.6+
- NumPy
- Matplotlib
- SciPy

## License

This project is licensed under the MIT License - see the LICENSE file for details.
