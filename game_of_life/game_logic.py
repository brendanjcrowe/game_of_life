"""
Game of Life - Core Game Logic

This module contains the core functionality for Conway's Game of Life simulation.
It provides functions for creating and updating the game state.
"""

from typing import Tuple

import numpy as np
from scipy.signal import convolve2d

# Cell states
ON = 255  # Live cell
OFF = 0  # Dead cell


class GameOfLife:
    """
    Conway's Game of Life simulation.

    This class encapsulates the grid state and provides methods for
    updating the state according to Conway's Game of Life rules.
    """

    def __init__(
        self,
        grid_size: int = 100,
        random_init: bool = True,
        random_fill_ratio: float = 0.2,
    ):
        """
        Initialize a Game of Life simulation.

        Args:
            grid_size: The size of the grid (N x N)
            random_init: Whether to initialize the grid with random values
            random_fill_ratio: The ratio of live cells to all cells for random initialization
        """
        self.grid_size = grid_size
        self.grid = np.zeros(grid_size * grid_size, dtype=np.uint8).reshape(
            grid_size, grid_size
        )

        if random_init:
            self._random_init(random_fill_ratio)

    def _random_init(self, fill_ratio: float = 0.2) -> None:
        """
        Initialize the grid with random live/dead cells.

        Args:
            fill_ratio: The ratio of live cells to all cells (between 0 and 1)
        """
        self.grid = np.random.choice(
            [ON, OFF], self.grid_size * self.grid_size, p=[fill_ratio, 1 - fill_ratio]
        ).reshape(self.grid_size, self.grid_size)

    def update(self) -> None:
        """
        Update the grid according to Conway's Game of Life rules:

        1. Any live cell with 2 or 3 live neighbors survives
        2. Any dead cell with exactly 3 live neighbors becomes alive
        3. All other cells die or stay dead

        Uses a convolution operation for efficient neighbor counting.
        """
        # Create binary grid (1 for ON, 0 for OFF)
        binary_grid = (self.grid == ON).astype(int)

        # Define the kernel for counting neighbors (3x3 with center being 0)
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

        # Use convolve2d with 'same' mode and periodic boundary conditions
        # This computes the sum of all 8 neighbors for each cell in one operation
        neighbor_count = convolve2d(binary_grid, kernel, mode="same", boundary="wrap")

        # Create a new grid based on Conway's rules
        new_grid = self.grid.copy()

        # Live cells (value == ON)
        live_cells = self.grid == ON
        # Live cells with <2 or >3 neighbors die
        new_grid[live_cells & ((neighbor_count < 2) | (neighbor_count > 3))] = OFF

        # Dead cells (value == OFF)
        dead_cells = self.grid == OFF
        # Dead cells with exactly 3 neighbors become alive
        new_grid[dead_cells & (neighbor_count == 3)] = ON

        # Update grid
        self.grid = new_grid

    def add_pattern(self, pattern: np.ndarray, position: Tuple[int, int]) -> None:
        """
        Add a pattern to the grid at the specified position.

        Args:
            pattern: A 2D numpy array representing the pattern
            position: (row, column) position where the top-left of the pattern will be placed
        """
        i, j = position
        h, w = pattern.shape
        # Make sure the pattern fits within the grid using modulo for wrapping
        self.grid[i : i + h, j : j + w] = pattern


class Patterns:
    """
    Collection of common patterns for Conway's Game of Life.

    This class provides static methods for creating various patterns
    that can be added to a Game of Life grid.
    """

    @staticmethod
    def glider() -> np.ndarray:
        """
        Returns a glider pattern.

        A glider is a pattern that moves diagonally across the grid.
        """
        return np.array([[OFF, OFF, ON], [ON, OFF, ON], [OFF, ON, ON]])

    @staticmethod
    def blinker() -> np.ndarray:
        """
        Returns a blinker pattern.

        A blinker is a pattern that oscillates between two states.
        """
        return np.array([[OFF, OFF, OFF], [ON, ON, ON], [OFF, OFF, OFF]])

    @staticmethod
    def block() -> np.ndarray:
        """
        Returns a block pattern.

        A block is a still life pattern that doesn't change between generations.
        """
        return np.array([[ON, ON], [ON, ON]])

    @staticmethod
    def beacon() -> np.ndarray:
        """
        Returns a beacon pattern.

        A beacon is a pattern that oscillates between two states.
        """
        return np.array(
            [
                [ON, ON, OFF, OFF],
                [ON, ON, OFF, OFF],
                [OFF, OFF, ON, ON],
                [OFF, OFF, ON, ON],
            ]
        )

    @staticmethod
    def gosper_glider_gun() -> np.ndarray:
        """
        Returns a Gosper Glider Gun pattern.

        A Gosper Glider Gun is a pattern that continuously emits gliders.
        """
        gun = np.zeros((11, 38), dtype=np.uint8)

        gun[5, 1] = gun[5, 2] = ON
        gun[6, 1] = gun[6, 2] = ON

        gun[3, 13] = gun[3, 14] = ON
        gun[4, 12] = gun[4, 16] = ON
        gun[5, 11] = gun[5, 17] = ON
        gun[6, 11] = gun[6, 15] = gun[6, 17] = gun[6, 18] = ON
        gun[7, 11] = gun[7, 17] = ON
        gun[8, 12] = gun[8, 16] = ON
        gun[9, 13] = gun[9, 14] = ON

        gun[1, 25] = ON
        gun[2, 23] = gun[2, 25] = ON
        gun[3, 21] = gun[3, 22] = ON
        gun[4, 21] = gun[4, 22] = ON
        gun[5, 21] = gun[5, 22] = ON
        gun[6, 23] = gun[6, 25] = ON
        gun[7, 25] = ON

        gun[3, 35] = gun[3, 36] = ON
        gun[4, 35] = gun[4, 36] = ON

        return gun
