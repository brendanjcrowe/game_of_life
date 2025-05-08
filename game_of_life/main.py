"""
Game of Life - Main Script

This module serves as the entry point for the Game of Life simulation.
It parses command-line arguments and sets up the simulation.
"""

import argparse
from typing import Any, Dict

from game_of_life.game_logic import AgentBasedGameOfLife, GameOfLife, Patterns
from game_of_life.policies import POLICIES
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

    # Game mode
    mode_group = parser.add_argument_group("Game Mode")
    mode_group.add_argument(
        "--agent-based", 
        action="store_true",
        help="Run in agent-based mode with live cells as agents"
    )
    mode_group.add_argument(
        "--use-policies",
        action="store_true",
        default=True,
        help="Use policy-based behavior with evolutionary dynamics (default)"
    )
    mode_group.add_argument(
        "--no-policies",
        action="store_true",
        help="Disable policy-based behavior"
    )
    mode_group.add_argument(
        "--show-agents",
        action="store_true",
        default=True,
        help="Show agent actions in visualization (arrows and circles)"
    )
    mode_group.add_argument(
        "--hide-agents",
        action="store_true",
        help="Hide agent action visualization"
    )
    mode_group.add_argument(
        "--show-policy-colors",
        action="store_true",
        default=True,
        help="Show different colors for different policy types"
    )
    mode_group.add_argument(
        "--hide-policy-colors",
        action="store_true",
        help="Hide policy color visualization"
    )
    
    # Policy distribution
    policy_group = parser.add_argument_group("Policy Distribution")
    for policy_id, policy in POLICIES.items():
        if policy_id != 0:
            policy_group.add_argument(
                f"--{policy.name.lower().replace(' ', '-')}-ratio",
                type=float,
                default=1.0,
                help=f"Relative ratio of {policy.name} policy (default: 1.0)"
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

    # Max steps
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of steps to run before ending simulation (default: unlimited)"
    )

    # Population tracking
    parser.add_argument(
        "--track-population",
        action="store_true",
        help="Track and plot population statistics over time"
    )

    parser.add_argument(
        "--plot-file",
        type=str,
        default=None,
        help="Save the population plot to this file (requires --track-population)"
    )

    # Parse arguments
    args = parser.parse_args()
    
    # Handle opposite flags
    if args.no_policies:
        args.use_policies = False
    if args.hide_agents:
        args.show_agents = False
    if args.hide_policy_colors:
        args.show_policy_colors = False
        
    return vars(args)


def get_policy_weights(args: Dict[str, Any]) -> Dict[int, float]:
    """
    Get policy weights from command-line arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Dictionary with policy IDs as keys and counts as values
    """
    policy_weights = {}
    
    # Check if any policy weights were explicitly provided
    any_weights_provided = False
    
    for policy_id, policy in POLICIES.items():
        if policy_id != 0:
            arg_name = f"{policy.name.lower().replace(' ', '-')}_ratio"
            if arg_name in args and args[arg_name] is not None:
                policy_weights[policy_id] = args[arg_name]
                any_weights_provided = True
            else:
                policy_weights[policy_id] = 0
    
    # If no weights were provided, create equal counts
    if not any_weights_provided:
        for policy_id in [1, 2, 3]:
            policy_weights[policy_id] = 1 / 3 # Equal weight
    else:
        total_weight = sum(policy_weights.values())
        for policy_id in [1, 2, 3]:
            policy_weights[policy_id] /= total_weight
    print(policy_weights)
    print(f"Policy weights from arguments: {policy_weights}")
    return policy_weights


def setup_game(args: Dict[str, Any]) -> GameOfLife:
    """
    Set up the Game of Life simulation based on command-line arguments.

    Args:
        args: Dictionary of command-line arguments

    Returns:
        A GameOfLife instance initialized according to the arguments
    """
    grid_size = args["grid_size"]
    max_steps = args["max_steps"]
    
    # Get policy counts for evolutionary dynamics
    policy_weights = get_policy_weights(args)
    
    # Determine which game class to use
    if args["agent_based"]:
        # Create agent-based game
        game = AgentBasedGameOfLife(
            grid_size=grid_size,
            random_init=True,
            use_policies=args["use_policies"],
            policy_weights=policy_weights,
            random_fill_ratio=args["fill_ratio"],
            max_steps=max_steps,
        )
    else:
        # Create standard game
        game = GameOfLife(
            grid_size=grid_size, 
            random_init=False,
            max_steps=max_steps
        )

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

    # Enable population tracking if requested
    if args["track_population"]:
        print("Enabling population tracking...")
        game.enable_population_tracking()

    # Print max steps information if set
    if args["max_steps"]:
        print(f"Setting maximum steps to {args['max_steps']}")

    # Set up the visualizer with explicit arguments
    viz = Visualizer(
        game, 
        update_interval=args["interval"],
        show_agents=args["show_agents"],
        show_policy_colors=args["show_policy_colors"],
        max_steps=args["max_steps"]
    )

    # Set title
    title = "Conway's Game of Life"
    
    # Add mode to title
    if args["agent_based"]:
        title += " (Agent-Based Mode)"
        if args["use_policies"]:
            title += " with Evolutionary Dynamics"
    
    # Add pattern to title
    if args["glider"]:
        title += " - Glider"
    elif args["gosper"]:
        title += " - Gosper Glider Gun"
    elif args["blinker"]:
        title += " - Blinker"
    elif args["beacon"]:
        title += " - Beacon"
    elif args["block"]:
        title += " - Block"
    else:
        title += " - Random"
    
    viz.set_title(title)

    # Start the animation
    viz.start_animation(save_path=args["save"])

    # Generate population plot if tracking was enabled
    if args["track_population"] and args["plot_file"]:
        game.save_population_plot(args["plot_file"])


if __name__ == "__main__":
    main()
