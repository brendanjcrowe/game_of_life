"""
Game of Life - Constants

This module defines constants used throughout the Game of Life simulation.
"""

# Cell states
ON = 255  # Live cell
OFF = 0  # Dead cell

# Agent actions
GROW_N = 0
GROW_NE = 1
GROW_E = 2
GROW_SE = 3
GROW_S = 4
GROW_SW = 5
GROW_W = 6
GROW_NW = 7
FORTIFY = 8

# Direction offsets (row, col) for the 8 directions
DIRECTIONS = [
    (-1, 0),   # North
    (-1, 1),   # Northeast
    (0, 1),    # East
    (1, 1),    # Southeast
    (1, 0),    # South
    (1, -1),   # Southwest
    (0, -1),   # West
    (-1, -1),  # Northwest
]
