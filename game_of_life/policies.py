"""
Game of Life - Agent Policies

This module defines the policies that agents can follow in the 
agent-based Conway's Game of Life simulation with evolutionary dynamics.
"""

import random
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from game_of_life.constants import DIRECTIONS, FORTIFY, OFF


class Policy(ABC):
    """
    Abstract base class for agent policies.
    
    Policies determine how agents choose actions in the agent-based
    Game of Life simulation.
    """
    
    @abstractmethod
    def choose_action(
        self, 
        row: int, 
        col: int, 
        grid: np.ndarray, 
        grid_size: int
    ) -> int:
        """
        Choose an action for the agent based on the current state.
        
        Args:
            row: Row position of the agent
            col: Column position of the agent
            grid: The current game grid
            grid_size: Size of the grid
            
        Returns:
            The chosen action (0-8)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of the policy.
        
        Returns:
            The policy name
        """
        pass
    
    @property
    @abstractmethod
    def color(self) -> str:
        """
        Get the display color for agents using this policy.
        
        Returns:
            Color string for visualization
        """
        pass
    
    def _get_valid_growth_directions(
        self, 
        row: int, 
        col: int, 
        grid: np.ndarray, 
        grid_size: int
    ) -> List[int]:
        """
        Get the list of valid directions for growth (unoccupied spaces).
        
        Args:
            row: Row position of the agent
            col: Column position of the agent
            grid: The current game grid
            grid_size: Size of the grid
            
        Returns:
            List of valid growth directions (0-7)
        """
        valid_directions = []
        
        for direction_idx, (dr, dc) in enumerate(DIRECTIONS):
            # Calculate target position with wrapping
            target_row = (row + dr) % grid_size
            target_col = (col + dc) % grid_size
            
            # Check if target cell is empty
            if grid[target_row, target_col] == OFF:
                valid_directions.append(direction_idx)
                
        return valid_directions
    
    def _count_neighbors(
        self, 
        row: int, 
        col: int, 
        grid: np.ndarray, 
        grid_size: int
    ) -> int:
        """
        Count the number of living neighbor cells.
        
        Args:
            row: Row position of the agent
            col: Column position of the agent
            grid: The current game grid
            grid_size: Size of the grid
            
        Returns:
            Number of living neighbors
        """
        count = 0
        
        for dr, dc in DIRECTIONS:
            # Calculate neighbor position with wrapping
            neighbor_row = (row + dr) % grid_size
            neighbor_col = (col + dc) % grid_size
            
            # Check if neighbor is alive
            if grid[neighbor_row, neighbor_col] > 0:
                count += 1
                
        return count


class RandomGrowPolicy(Policy):
    """
    Policy: Randomly grow into a neighboring unoccupied space.
    If none are available, then fortify.
    """
    
    def choose_action(
        self, 
        row: int, 
        col: int, 
        grid: np.ndarray, 
        grid_size: int
    ) -> int:
        # Find valid growth directions
        valid_directions = self._get_valid_growth_directions(row, col, grid, grid_size)
        
        # If there are valid directions, choose one randomly
        if valid_directions:
            return random.choice(valid_directions)
        
        # Otherwise, fortify
        return FORTIFY
    
    @property
    def name(self) -> str:
        return "RandomGrowth"
    
    @property
    def color(self) -> str:
        return "red"


class AlwaysFortifyPolicy(Policy):
    """
    Policy: Always fortify no matter what.
    """
    
    def choose_action(
        self, 
        row: int, 
        col: int, 
        grid: np.ndarray, 
        grid_size: int
    ) -> int:
        # Always choose to fortify
        return FORTIFY
    
    @property
    def name(self) -> str:
        return "AlwaysFortify"
    
    @property
    def color(self) -> str:
        return "blue"


class AdaptivePolicy(Policy):
    """
    Policy: If the agent has 2 or 3 neighbors, fortify.
    Otherwise, grow into a random unoccupied space.
    """
    
    def choose_action(
        self, 
        row: int, 
        col: int, 
        grid: np.ndarray, 
        grid_size: int
    ) -> int:
        # Count neighbors
        neighbor_count = self._count_neighbors(row, col, grid, grid_size)
        
        # If we have 2 or 3 neighbors, fortify
        if neighbor_count in [2, 3]:
            return FORTIFY
            
        # Otherwise, try to grow
        valid_directions = self._get_valid_growth_directions(row, col, grid, grid_size)
        
        # If there are valid directions, choose one randomly
        if valid_directions:
            return random.choice(valid_directions)
        
        # If no valid directions, fortify anyway
        return FORTIFY
    
    @property
    def name(self) -> str:
        return "Adaptive"
    
    @property
    def color(self) -> str:
        return "green"


# Dictionary of all policies for easy access
POLICIES = {
    0: None,
    1: RandomGrowPolicy(),
    2: AlwaysFortifyPolicy(),
    3: AdaptivePolicy()
}

POLICY_NAMES = {
    "random-growth": 1,
    "always-fortify": 2,
    "adaptive": 3
}


def get_random_policy() -> Policy:
    """
    Get a random policy from the available policies.
    
    Returns:
        A randomly chosen policy
    """
    policy_id = random.randint(0, len(POLICIES) - 1)
    return POLICIES[policy_id]


def get_policy_by_id(policy_id: int) -> Policy:
    """
    Get a policy by its ID.
    
    Args:
        policy_id: The ID of the policy to get
        
    Returns:
        The corresponding policy object
    """
    return POLICIES[policy_id]


def inherit_policy_from_neighbors(
    row: int, 
    col: int, 
    grid: np.ndarray, 
    policy_grid: np.ndarray,
    grid_size: int,
    policy_counts_total = None
) -> int:
    """
    Inherit a policy from neighbors proportional to their representation.
    
    Args:
        row: Row position within neighborhood (typically 1 for center)
        col: Column position within neighborhood (typically 1 for center)
        grid: 3x3 neighborhood grid
        policy_grid: 3x3 neighborhood policy grid
        grid_size: Size of the neighborhood (typically 3)
        
    Returns:
        The policy ID to assign to the new agent
    """
    # Count occurrences of each policy among neighboring cells
    # Only count policies with ID > 0 (not the None policy)
    policy_counts = {}
    
    for dr, dc in DIRECTIONS:
        # Calculate neighbor position within the 3x3 neighborhood
        neighbor_row = (row + dr) % grid_size
        neighbor_col = (col + dc) % grid_size
        
        # If neighbor is alive and has a valid policy, count it
        if grid[neighbor_row, neighbor_col] > 0:
            neighbor_policy = policy_grid[neighbor_row, neighbor_col]
            if neighbor_policy > 0:  # Skip the None policy (0)
                if neighbor_policy not in policy_counts:
                    policy_counts[neighbor_policy] = 0
                policy_counts[neighbor_policy] += 1
    
    valid_policy_ids = [1, 2, 3]  # Assuming these are the valid policy IDs
    
    total_neighbors = sum(policy_counts.values())
    if total_neighbors == 0:
        pol = [pid for pid in valid_policy_ids]
        p = np.array([policy_counts_total[pid] for pid in valid_policy_ids]).astype(float)
        p /= p.sum()
        return np.random.choice(pol, p=p)
    
    # Choose policy proportional to neighbor policies
    policies = list(policy_counts.keys())
    weights = [policy_counts[p] / total_neighbors for p in policies]
    
    return random.choices(policies, weights=weights, k=1)[0] 