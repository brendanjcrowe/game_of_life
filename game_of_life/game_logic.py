"""
Game of Life - Core Game Logic

This module contains the core functionality for Conway's Game of Life simulation.
It provides functions for creating and updating the game state.
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from collections import defaultdict

from game_of_life.constants import DIRECTIONS, FORTIFY, ON, OFF
from game_of_life.policies import (
    POLICIES,
    get_policy_by_id,
    get_random_policy,
    inherit_policy_from_neighbors,
    POLICY_NAMES
)


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
        max_steps: Optional[int] = None,
    ):
        """
        Initialize a Game of Life simulation.

        Args:
            grid_size: The size of the grid (N x N)
            random_init: Whether to initialize the grid with random values
            random_fill_ratio: The ratio of live cells to all cells for random initialization
            max_steps: Maximum number of steps to run before ending simulation
        """
        self.grid_size = grid_size
        self.grid = np.zeros(grid_size * grid_size, dtype=np.uint8).reshape(
            grid_size, grid_size
        )
        self.max_steps = max_steps

        if random_init:
            self._random_init(random_fill_ratio)

        self.track_population = False
        self.population_history = []
        self.step_count = 0
        self.is_complete = False

    def _random_init(self, fill_ratio: float = 0.2) -> None:
        """
        Initialize the grid with random live/dead cells.

        Args:
            fill_ratio: The ratio of live cells to all cells (between 0 and 1)
        """
        self.grid = np.random.choice(
            [ON, OFF], self.grid_size * self.grid_size, p=[fill_ratio, 1 - fill_ratio]
        ).reshape(self.grid_size, self.grid_size)

    def enable_population_tracking(self) -> None:
        """
        Enable tracking of population statistics over time.
        """
        self.track_population = True
        self.population_history = []
        self.step_count = 0
        
        self._record_population()

    def _record_population(self) -> None:
        """
        Record the current population statistics.
        """
        if not self.track_population:
            return
            
        total_population = np.sum(self.grid > 0)
        
        population_data = {
            'step': self.step_count,
            'total': total_population
        }
        

        if self.use_policies:
            # Verify counts match actual grid
            # if self.step_count % 10 == 0:  # Check every 10 steps to save performance
            #     self._verify_policy_counts()
            
            for policy_id, count in self.policy_counts.items():
                policy_name = POLICIES[policy_id].name
                population_data[policy_name] = max(0, count)  # Ensure no negative values
        
        # Record data
        self.population_history.append(population_data)

    # def _verify_policy_counts(self) -> None:
    #     """
    #     Verify that policy counts match the actual grid state.
    #     This is a debugging function to ensure accounting is correct.
    #     """
    #     actual_counts = {policy_id: 0 for policy_id in POLICIES.keys()}
        
    #     # Count cells with each policy
    #     for row, col in np.argwhere(self.grid == ON):
    #         policy_id = self.policy_grid[row, col]
    #         if policy_id in actual_counts:
    #             actual_counts[policy_id] += 1
        
    #     # Check for discrepancies
    #     for policy_id, count in actual_counts.items():
    #         if count != self.policy_counts[policy_id]:
    #             # Fix the discrepancy
    #             print(f"Correcting count for {POLICIES[policy_id].name}: {self.policy_counts[policy_id]} -> {count}")
    #             self.policy_counts[policy_id] = count

    def update(self) -> bool:
        """
        Update the grid according to Conway's Game of Life rules.
        
        Returns:
            Boolean indicating if the simulation should continue
        """
        # Check if simulation is complete
        if self.is_complete:
            return False
            
        # Check if we've reached max steps
        if self.max_steps is not None and self.step_count >= self.max_steps:
            self.is_complete = True
            print(f"Game reached maximum steps ({self.max_steps})")
            return False
            
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

        # Increment step counter and record population if tracking is enabled
        if self.track_population:
            self.step_count += 1
            self._record_population()
            
        return True

    def is_simulation_complete(self) -> bool:
        """
        Check if the simulation is complete.
        
        Returns:
            True if the simulation is complete, False otherwise
        """
        return self.is_complete
        
    def set_max_steps(self, max_steps: Optional[int]) -> None:
        """
        Set the maximum number of steps for the simulation.
        
        Args:
            max_steps: Maximum number of steps, or None for unlimited
        """
        self.max_steps = max_steps

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

    def save_population_plot(self, filename: str) -> None:
        """
        Generate and save a plot of population statistics over time.
        
        Args:
            filename: Path to save the plot image
        """
        if not self.track_population or not self.population_history:
            print("No population data to plot. Check that --track-population is enabled.")
            return
            
        # Print diagnostic info
        print(f"Plotting population data with {len(self.population_history)} data points")
        
        steps = [data['step'] for data in self.population_history]
        total = [data['total'] for data in self.population_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, total, 'b-', linewidth=2, label='Total Population')
        plt.xlabel('Steps')
        plt.ylabel('Population')
        plt.title('Population Over Time')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(filename)
        print(f"Population plot saved to {filename}")
        
        # Close the figure to prevent memory leaks
        plt.close()


class AgentBasedGameOfLife(GameOfLife):
    """
    Agent-based extension of Conway's Game of Life.
    
    In this version, each live cell is an agent that can take actions:
    - Grow in one of 8 directions
    - Fortify itself
    
    The rules are modified:
    - If 3+ cells choose to grow into a dead cell, it becomes alive
    - If a live cell is grown into, it increases its effective neighbor count by 1
    - Fortifying makes a cell survive with one more or one less neighbor
    
    With evolutionary dynamics, agents can follow different policies, and 
    new agents inherit policies from their neighbors.
    """
    
    def __init__(
        self,
        grid_size: int = 100,
        random_init: bool = True,
        random_fill_ratio: float = 0.2,
        use_policies: bool = True,
        policy_weights: Optional[Dict[int, float]] = None,
        max_steps: Optional[int] = None,
        which_policies: List[str] = None,
    ):
        """
        Initialize an agent-based Game of Life simulation.
        
        Args:
            grid_size: The size of the grid (N x N)
            random_init: Whether to initialize the grid with random values
            random_fill_ratio: The ratio of live cells to all cells for random initialization
            use_policies: Whether to use the policy-based behavior
            policy_counts: Initial distribution of policies (if None, equal distribution)
            max_steps: Maximum number of steps to run before ending simulation
            which_policies: List of policy names to use (if None, use all)
        """
        super().__init__(grid_size, random_init, random_fill_ratio, max_steps)
        
        # Track which cells are fortified
        self.fortified = np.zeros((grid_size, grid_size), dtype=bool)
        
        # Track cell actions
        self.actions = np.zeros((grid_size, grid_size), dtype=np.uint8)
        
        # Set up policy-based behavior
        self.use_policies = use_policies
        self.policy_grid = np.zeros((grid_size, grid_size), dtype=np.int8)
        
        # Define which policies to use
        if which_policies is None:
            which_policies = ["random-growth", "always-fortify", "adaptive"]
        
        # Create active policies dictionary
        valid_policy_ids = [POLICY_NAMES[p] for p in which_policies if p in POLICY_NAMES]
        self.active_policies = {k: POLICIES[k] for k in valid_policy_ids}
        
        print(f"Active policies: {[POLICIES[pid].name for pid in valid_policy_ids]}")
        
        # Track stats about policies - initialize with zeros
        self.policy_weights = policy_weights
        
        # If we're using random initialization, assign policies
        if random_init and use_policies:
            self._init_policies(policy_weights)
            self._record_population()
    
    def _get_policy_weights(self, policy_counts: Optional[Dict[int, int]] = None) -> Dict[int, float]:
        """
        Convert policy counts to probability weights.
        
        Args:
            policy_counts: Dictionary with policy IDs as keys and counts/weights as values
            
        Returns:
            Dictionary with policy IDs as keys and probability weights as values
        """
        weights = {}
        
        # If specific counts are provided, use them
        if policy_counts:
            # Calculate total to normalize
            total = sum(policy_counts.values())
            
            # Convert counts to weights
            for policy_id in POLICIES.keys():
                weights[policy_id] = policy_counts.get(policy_id, 1) / total
        else:
            # If no counts provided, use equal distribution
            equal_weight = 1.0 / len(POLICIES)
            weights = {policy_id: equal_weight for policy_id in POLICIES.keys()}
        
        # Verify weights sum to 1.0
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 0.001:
            print(f"Warning: Policy weights sum to {weight_sum}, normalizing")
            for policy_id in weights:
                weights[policy_id] /= weight_sum
        
        print("Policy weights for initialization:")
        for policy_id, weight in weights.items():
            print(f"{POLICIES[policy_id].name}: {weight:.3f}")
        
        return weights

    def _init_policies(self, policy_weights: Optional[Dict[int, int]] = None) -> None:
        """
        Initialize policy-related data structures.
        
        Args:
            policy_counts: Dictionary with policy IDs as keys and weights as values
        """
        # Create a grid to track policy assignments
        self.policy_grid = np.zeros_like(self.grid, dtype=np.int8)
        
        # Initialize policy counts
        self.policy_counts = {policy_id: 0 for policy_id in self.active_policies.keys()}
        
        # Get policy IDs for random assignment
        policy_ids = list(self.active_policies.keys())
        
        # Count live cells
        live_cells = np.argwhere(self.grid == ON)
        total_cells = len(live_cells)
        print(f"Initializing policies for {total_cells} live cells")
        
        # Assign policies randomly to all live cells
        for row, col in live_cells:
            # Choose a random policy (equal chance for each)
            policy_id = np.random.choice(policy_ids, p=[self.policy_weights[pid] for pid in policy_ids])
            
            # Assign the policy
            self.policy_grid[row, col] = policy_id
            
            # Update counts
            self.policy_counts[policy_id] += 1
        
        # Print policy distribution
        print("Initial policy distribution:")
        for policy_id, count in sorted(self.policy_counts.items()):
            policy_name = POLICIES[policy_id].name
            print(f"{policy_name}: {count}")

    def _get_wrapped_coords(self, row: int, col: int) -> Tuple[int, int]:
        """
        Get the wrapped coordinates for a given position.
        
        Args:
            row: The row coordinate
            col: The column coordinate
            
        Returns:
            The wrapped (row, col) coordinates
        """
        return row % self.grid_size, col % self.grid_size
    
    def _assign_agent_action(self, row: int, col: int) -> int:
        """
        Assign an action to an agent based on its policy.
        
        Args:
            row: The row coordinate of the agent
            col: The column coordinate of the agent
            
        Returns:
            The assigned action
        """
        # Get the agent's policy
        policy_id = self.policy_grid[row, col]
        
        # If no policy or invalid policy, use random action
        if policy_id == 0 or policy_id not in self.active_policies:
            return random.randint(0, 8)
        
        # Get the policy object
        policy = self.active_policies[policy_id]
        
        # Let the policy choose the action
        return policy.choose_action(row, col, self.grid, self.grid_size)
    
    def _assign_agent_actions(self) -> None:
        """
        Assign actions to all live cells (agents).
        """
        # Reset actions
        self.actions.fill(0)
        
        # Find all live cells
        live_cells = np.argwhere(self.grid == ON)
        
        # Assign actions to each live cell
        for row, col in live_cells:
            self.actions[row, col] = self._assign_agent_action(row, col)
    
    def _get_neighbor_cells(self, row: int, col: int):
        """
        Get the states of the neighboring cells with proper wrapping.
        
        Args:
            row: Row index
            col: Column index
            
        Returns:
            A 3x3 array containing the neighboring cells (including the center cell)
        """
        neighbors = np.zeros((3, 3), dtype=int)
        neighbors_policy = np.zeros((3, 3), dtype=int)
        
        for i in range(3):
            for j in range(3):
                # Calculate wrapped coordinates
                r = (row - 1 + i) % self.grid_size
                c = (col - 1 + j) % self.grid_size
                
                # Copy grid and policy data
                neighbors[i, j] = self.grid[r, c]
                if self.use_policies:
                    neighbors_policy[i, j] = self.policy_grid[r, c]
                
        return neighbors, neighbors_policy
    
    def _record_population(self) -> None:
        """
        Record the current population statistics.
        """
        if not self.track_population:
            return
            
        # Total population
        total_population = np.sum(self.grid > 0)
        
        # Create population record
        population_data = {
            'step': self.step_count,
            'total': total_population
        }
        
        # Add policy-specific data if using policies
        if self.use_policies:
            # Verify counts match actual grid
            # if self.step_count % 10 == 0:  # Check every 10 steps to save performance
            #     self._verify_policy_counts()
            
            for policy_id, count in self.policy_counts.items():
                policy_name = POLICIES[policy_id].name
                population_data[policy_name] = max(0, count)  # Ensure no negative values
        
        # Record data
        self.population_history.append(population_data)
    
    # def _verify_policy_counts(self) -> None:
    #     """
    #     Verify that policy counts match the actual grid state.
    #     This is a debugging function to ensure accounting is correct.
    #     """
    #     actual_counts = {policy_id: 0 for policy_id in POLICIES.keys()}
        
    #     # Count cells with each policy
    #     for row, col in np.argwhere(self.grid == ON):
    #         policy_id = self.policy_grid[row, col]
    #         if policy_id in actual_counts:
    #             actual_counts[policy_id] += 1
        
    #     # Check for discrepancies
    #     for policy_id, count in actual_counts.items():
    #         if count != self.policy_counts[policy_id]:
    #             # Fix the discrepancy
    #             print(f"Correcting count for {POLICIES[policy_id].name}: {self.policy_counts[policy_id]} -> {count}")
    #             self.policy_counts[policy_id] = count
    
    def save_population_plot(self, filename: str) -> None:
        """
        Generate and save a plot of population statistics over time,
        including policy-specific populations if using policies.
        
        Args:
            filename: Path to save the plot image
        """
        if not self.track_population or not self.population_history:
            print("No population data to plot. Check that --track-population is enabled.")
            return
        
        # Print diagnostic info
        print(f"Plotting population data with {len(self.population_history)} data points")
        
        steps = [data['step'] for data in self.population_history]
        total = [data['total'] for data in self.population_history]
        
        # Create a single plot with everything overlaid
        plt.figure(figsize=(12, 8))
        
        # Plot total population with thick black line
        plt.plot(steps, total, 'k-', linewidth=3, label='Total Population')
        
        # If using policies, plot policy-specific populations on the same chart
        if self.use_policies and len(self.population_history) > 0:
            policy_names = []
            for policy_id in self.active_policies:
                policy_name = POLICIES[policy_id].name
                if policy_name in self.population_history[0]:
                    policy_names.append(policy_name)
                
                # Extract data for this policy
                policy_data = [data.get(policy_name, 0) for data in self.population_history]
                
                # Get policy color from the policy object
                policy_color = POLICIES[policy_id].color
                
                plt.plot(steps, policy_data, linewidth=2, label=policy_name, color=policy_color)
        
        plt.xlabel('Steps')
        plt.ylabel('Population')
        plt.title('Population Over Time')
        plt.grid(True)
        plt.legend(loc='upper right')
        
        # Save the plot
        plt.savefig(filename)
        print(f"Population plot saved to {filename}")
        
        # Close the figure to prevent memory leaks
        plt.close()
    
    def update(self) -> bool:
        """
        Update the grid according to the agent-based rules.
        
        Returns:
            Boolean indicating if the simulation should continue
        """
        # Check if simulation is complete
        if self.is_complete:
            return False
            
        # Check if we've reached max steps
        if self.max_steps is not None and self.step_count >= self.max_steps:
            self.is_complete = True
            print(f"Game reached maximum steps ({self.max_steps})")
            return False
            
        # Assign actions to all agents (live cells)
        self._assign_agent_actions()
        
        # Create binary grid (1 for ON, 0 for OFF)
        binary_grid = (self.grid == ON).astype(int)
        
        # Count natural neighbors using convolution
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        neighbor_count = convolve2d(binary_grid, kernel, mode="same", boundary="wrap")
        
        # Track cells that are grown into
        grown_into = np.zeros_like(self.grid, dtype=np.int8)
        
        # Process growth actions
        for row, col in np.argwhere(self.grid == ON):
            action = self.actions[row, col]
            
            # Skip fortify actions
            if action == FORTIFY:
                continue
                
            # Get the direction to grow
            dr, dc = DIRECTIONS[action]
            
            # Get the target cell (with wrapping)
            target_row, target_col = self._get_wrapped_coords(row + dr, col + dc)
            
            # Increment the grown_into counter for the target cell
            grown_into[target_row, target_col] += 1
        
        # Create a new grid
        new_grid = self.grid.copy()
        
        # Track new fortified cells
        new_fortified = self.fortified.copy()
        
        new_policy_grid = self.policy_grid.copy()
        # For evolutionary dynamics, track new cells to assign policies
        new_cells = []
        
        # Update fortified status
        for row, col in np.argwhere(self.grid == ON):
            if self.actions[row, col] == FORTIFY:
                new_fortified[row, col] = True
            else:
                # If not fortifying now, lose fortified status
                new_fortified[row, col] = False
        
        # Apply rules for live cells
        for row, col in np.argwhere(self.grid == ON):
            # Calculate effective neighbor count
            effective_neighbors = neighbor_count[row, col] + grown_into[row, col]
            
            # Check if the cell survives
            survives = False
            
            if self.fortified[row, col]:
                # Fortified cells survive with 1-4 neighbors (instead of 2-3)
                survives = 1 <= effective_neighbors <= 4
            else:
                # Normal cells survive with 2-3 neighbors
                survives = 2 <= effective_neighbors <= 3
                
            # Update the cell state
            if not survives:
                new_grid[row, col] = OFF
                
                # If using policies, decrement the count for this policy
                if self.use_policies:
                    policy_id = self.policy_grid[row, col]
                    #if policy_id in self.policy_counts and self.policy_counts[policy_id] > 0:
                    # if policy_id in self.policy_counts:
                    #     self.policy_counts[policy_id] -= 1
                    new_policy_grid[row, col] = 0
                    #else:
                        # pass
        
        # Apply rules for dead cells
        for row, col in np.argwhere(self.grid == OFF):
            # Check if enough cells are growing into this cell
            if grown_into[row, col] + neighbor_count[row, col] == 3:
                new_grid[row, col] = ON
                
                # Mark for policy assignment
                if self.use_policies:
                    new_cells.append((row, col))
        
        # Update grid and fortified status
        self.grid = new_grid
        self.fortified = new_fortified
        
        # Assign policies to new cells
        if self.use_policies:
            for row, col in new_cells:
                # Get properly wrapped neighborhood
                neighbors, neighbors_policy = self._get_neighbor_cells(row, col)
                
                # Inherit policy from neighbors
                policy_id = inherit_policy_from_neighbors(
                    1, 1, neighbors, neighbors_policy, 3, self.policy_counts
                )
                
                # Assign the policy and update counts
                # self.policy_grid[row, col] = policy_id
                new_policy_grid[row, col] = policy_id
                # self.policy_counts[policy_id] += 1
        
        self.policy_grid = new_policy_grid
        self.policy_counts = {pid: np.ones_like(self.policy_grid)[self.policy_grid == pid].sum() for pid in self.active_policies.keys()}
        # Increment step counter and record population if tracking is enabled
        if self.track_population:
            self.step_count += 1
            self._record_population()
        
        return True
    
    def get_agent_action(self, row: int, col: int) -> Optional[int]:
        """
        Get the action of the agent at the specified position.
        
        Args:
            row: The row coordinate
            col: The column coordinate
            
        Returns:
            The action index (0-8) or None if no agent at the position
        """
        if self.grid[row, col] == ON:
            return self.actions[row, col]
        return None
    
    def is_fortified(self, row: int, col: int) -> bool:
        """
        Check if the cell at the specified position is fortified.
        
        Args:
            row: The row coordinate
            col: The column coordinate
            
        Returns:
            True if the cell is fortified, False otherwise
        """
        return self.fortified[row, col]
    
    def get_policy_id(self, row: int, col: int) -> Optional[int]:
        """
        Get the policy ID of the agent at the specified position.
        
        Args:
            row: The row coordinate
            col: The column coordinate
            
        Returns:
            The policy ID or None if no agent or not using policies
        """
        if not self.use_policies or self.grid[row, col] != ON:
            return None
        return self.policy_grid[row, col]
    
    def get_policy_counts(self) -> Dict[int, int]:
        """
        Get the current counts of each policy type.
        
        Returns:
            Dictionary with policy IDs as keys and counts as values
        """
        return self.policy_counts.copy()
    
    def set_use_policies(self, use_policies: bool) -> None:
        """
        Set whether to use policies.
        
        Args:
            use_policies: Whether to use policies
        """
        # If turning on policies and they weren't being used before
        if use_policies and not self.use_policies:
            self._init_policies()
            
        self.use_policies = use_policies


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
