"""
Game of Life - Main Script

This module serves as the entry point for the Game of Life simulation.
It parses command-line arguments and sets up the simulation.
"""

import argparse
from typing import Any, Dict

from game_of_life.game_logic import GameOfLife, Patterns
from game_of_life.visualization import Visualizer


def parse_arguments() -> Dict[str, Any]:
    """
    Parse command-line arguments.

    Returns:
        A dictionary containing the parsed arguments
    """
    parser = argparse.ArgumentParser(description="Conway's Game of Life simulation.")

    # Grid size
    parser.add_argument(
        "--grid-size",
        type=int,
        default=100,
        help="Size of the grid (N x N, default: 100)",
    )

    # Update interval
    parser.add_argument(
        "--interval",
        type=int,
        default=100,
        help="Animation update interval in milliseconds (default: 500, higher is slower)",
    )

    # Save animation
    parser.add_argument(
        "--save", type=str, help="Save the animation to the specified file"
    )

    # Pattern options
    pattern_group = parser.add_argument_group("Patterns")
    pattern_group.add_argument(
        "--random", action="store_true", help="Initialize with random cells (default)"
    )
    pattern_group.add_argument(
        "--glider", action="store_true", help="Initialize with a glider pattern"
    )
    pattern_group.add_argument(
        "--gosper", action="store_true", help="Initialize with a Gosper Glider Gun"
    )
    pattern_group.add_argument(
        "--blinker", action="store_true", help="Initialize with a blinker pattern"
    )
    pattern_group.add_argument(
        "--beacon", action="store_true", help="Initialize with a beacon pattern"
    )
    pattern_group.add_argument(
        "--block", action="store_true", help="Initialize with a block pattern"
    )

    # Fill ratio for random initialization
    parser.add_argument(
        "--fill-ratio",
        type=float,
        default=0.2,
        help="Fill ratio for random initialization (0.0-1.0, default: 0.2)",
    )

    # Parse arguments
    args = parser.parse_args()
    return vars(args)


def setup_game(args: Dict[str, Any]) -> GameOfLife:
    """
    Set up the Game of Life simulation based on command-line arguments.

    Args:
        args: Dictionary of command-line arguments

    Returns:
        A GameOfLife instance initialized according to the arguments
    """
    grid_size = args["grid_size"]

    # Create game with empty grid
    game = GameOfLife(grid_size=grid_size, random_init=False)

    # Add pattern based on arguments
    if args["glider"]:
        game.add_pattern(Patterns.glider(), (1, 1))
    elif args["gosper"]:
        game.add_pattern(Patterns.gosper_glider_gun(), (10, 10))
    elif args["blinker"]:
        game.add_pattern(Patterns.blinker(), (grid_size // 2, grid_size // 2))
    elif args["beacon"]:
        game.add_pattern(Patterns.beacon(), (grid_size // 2, grid_size // 2))
    elif args["block"]:
        game.add_pattern(Patterns.block(), (grid_size // 2, grid_size // 2))
    else:  # Default to random
        game._random_init(args["fill_ratio"])

    return game


def main() -> None:
    """
    Main function that sets up and runs the Game of Life simulation.
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Set up the game
    game = setup_game(args)

    # Set up the visualizer
    viz = Visualizer(game, update_interval=args["interval"])

    # Set title
    if args["glider"]:
        viz.set_title("Conway's Game of Life - Glider")
    elif args["gosper"]:
        viz.set_title("Conway's Game of Life - Gosper Glider Gun")
    elif args["blinker"]:
        viz.set_title("Conway's Game of Life - Blinker")
    elif args["beacon"]:
        viz.set_title("Conway's Game of Life - Beacon")
    elif args["block"]:
        viz.set_title("Conway's Game of Life - Block")
    else:
        viz.set_title("Conway's Game of Life - Random")

    # Start the animation
    viz.start_animation(save_path=args["save"])


if __name__ == "__main__":
    main()
