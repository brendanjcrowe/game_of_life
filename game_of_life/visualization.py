"""
Game of Life - Visualization

This module contains visualization functionality for Conway's Game of Life.
It provides the Visualizer class to render and animate the game state.
"""

from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arrow, Circle, Rectangle

from game_of_life.constants import DIRECTIONS, FORTIFY, ON
from game_of_life.game_logic import AgentBasedGameOfLife, GameOfLife
from game_of_life.policies import POLICIES, get_policy_by_id


class Visualizer:
    """
    Visualizer for Conway's Game of Life.

    This class handles the rendering and animation of the Game of Life simulation.
    """

    def __init__(
        self,
        game: Union[GameOfLife, AgentBasedGameOfLife],
        update_interval: int = 50,
        cmap: str = "viridis",
        interpolation: str = "nearest",
        show_agents: bool = True,
        show_policy_colors: bool = True
    ):
        """
        Initialize the visualizer.

        Args:
            game: The GameOfLife instance to visualize
            update_interval: Animation update interval in milliseconds
            cmap: Matplotlib colormap to use for rendering
            interpolation: Interpolation method for displaying the grid
            show_agents: Whether to show agent visualizations (for AgentBasedGameOfLife)
            show_policy_colors: Whether to show different colors for different policies
        """
        self.game = game
        self.update_interval = update_interval
        self.cmap = cmap
        self.interpolation = interpolation
        self.show_agents = show_agents and isinstance(game, AgentBasedGameOfLife)
        self.show_policy_colors = show_policy_colors and isinstance(game, AgentBasedGameOfLife)

        # Initialize the figure and axes
        self.fig, self.ax = plt.subplots(figsize=(10, 10))  # Increased figure size
        
        # Create a grid visualization
        if isinstance(game, AgentBasedGameOfLife):
            # For agent-based, we'll use a custom visualization
            grid_display = self._create_agent_grid(self.game)
            self.img = self.ax.imshow(
                grid_display, interpolation=self.interpolation, cmap='gray'  # Use grayscale for base grid
            )
            
            # Add arrow indicators for actions and policy colors
            self.arrow_patches = []
            self.fortify_patches = []
            self.policy_patches = []
            self._update_agent_indicators()
            
            # Add a legend for policies if using them
            if self.show_policy_colors and hasattr(game, 'use_policies') and game.use_policies:
                self._add_policy_legend()
        else:
            # For standard visualization, just show the grid
            self.img = self.ax.imshow(
                self.game.grid, interpolation=self.interpolation, cmap=self.cmap
            )

        # Initialize the animation object
        self.animation = None
        
        # Set axis labels and grid
        self.ax.set_xlabel('Column')
        self.ax.set_ylabel('Row')
        self.ax.grid(False)

    def _create_agent_grid(self, game: AgentBasedGameOfLife) -> np.ndarray:
        """
        Create a grid visualization for agent-based game.
        
        Args:
            game: The AgentBasedGameOfLife instance
            
        Returns:
            A 2D array suitable for visualization
        """
        # Create a grayscale grid display
        # We'll overlay colored patches for policies
        grid_display = game.grid.copy()
        
        # Enhance fortified cells to be brighter
        for row, col in np.argwhere(game.fortified):
            if game.grid[row, col] == ON:
                # Make fortified cells brighter
                grid_display[row, col] = 255
                
        return grid_display

    def _add_policy_legend(self) -> None:
        """
        Add a legend showing the different policy types.
        """
        # Create legend handles
        handles = []
        labels = []
        
        # Add a legend entry for each policy
        for policy_id, policy in POLICIES.items():
            patch = Rectangle((0, 0), 1, 1, color=policy.color, alpha=0.7)
            handles.append(patch)
            labels.append(policy.name)
        
        # Add the legend to the plot
        self.ax.legend(
            handles=handles, 
            labels=labels,
            loc='upper right',
            fontsize='medium'
        )

    def _update_agent_indicators(self) -> None:
        """
        Update the visual indicators for agent actions and policies.
        """
        # Only applicable for agent-based game
        if not isinstance(self.game, AgentBasedGameOfLife):
            return
            
        # Clear existing patches
        for patch in self.arrow_patches:
            patch.remove()
        self.arrow_patches.clear()
        
        for patch in self.fortify_patches:
            patch.remove()
        self.fortify_patches.clear()
        
        for patch in self.policy_patches:
            patch.remove()
        self.policy_patches.clear()
        
        agent_game = self.game  # type: AgentBasedGameOfLife
        
        # Get grid dimensions for proper scaling
        ny, nx = agent_game.grid.shape
        
        # Draw policy colors first (as background)
        if self.show_policy_colors and hasattr(agent_game, 'use_policies') and agent_game.use_policies:
            for row, col in np.argwhere(agent_game.grid == ON):
                policy_id = agent_game.get_policy_id(row, col)
                if policy_id is not None:
                    policy = get_policy_by_id(policy_id)
                    rect = Rectangle(
                        (col - 0.5, row - 0.5),
                        1, 1,
                        color=policy.color,
                        alpha=0.7,  # Increased alpha for better visibility
                        linewidth=0
                    )
                    self.ax.add_patch(rect)
                    self.policy_patches.append(rect)
        
        # Draw agent actions next (as foreground)
        if self.show_agents:
            for row, col in np.argwhere(agent_game.grid == ON):
                action = agent_game.get_agent_action(row, col)
                
                if action == FORTIFY:
                    # Draw a circle for fortify action
                    circle = Circle(
                        (col, row), 
                        0.3, 
                        color='yellow', 
                        alpha=0.8,  # Increased alpha
                        fill=True
                    )
                    self.ax.add_patch(circle)
                    self.fortify_patches.append(circle)
                else:
                    # Draw an arrow for growth direction
                    dr, dc = DIRECTIONS[action]
                    arrow = Arrow(
                        col, row, 
                        dc * 0.5, dr * 0.5,  # Shorter arrow
                        width=0.3,
                        color='white',  # White arrows for better visibility
                        alpha=0.9  # Increased alpha
                    )
                    self.ax.add_patch(arrow)
                    self.arrow_patches.append(arrow)

    def update_frame(self, frame_num) -> Tuple:
        """
        Update function for animation.

        Args:
            frame_num: Current frame number

        Returns:
            The updated image
        """
        # Update the game state
        self.game.update()

        # Update the display
        if isinstance(self.game, AgentBasedGameOfLife):
            # Update agent-based visualization
            grid_display = self._create_agent_grid(self.game)
            self.img.set_data(grid_display)
            
            # Update agent indicators (ensure this is called on every frame update)
            self._update_agent_indicators()
            
            # Update title with policy statistics if using policies
            if hasattr(self.game, 'use_policies') and self.game.use_policies:
                policy_counts = self.game.get_policy_counts()
                stats_str = " | ".join([
                    f"{POLICIES[pid].name}: {count}" 
                    for pid, count in policy_counts.items()
                ])
                self.ax.set_title(f"{self.title}\n{stats_str}", fontsize=10)
                
            # Force the figure to redraw to ensure policy patches are visible
            self.fig.canvas.draw_idle()
        else:
            # Standard visualization
            self.img.set_data(self.game.grid)
            
        return (self.img,)

    def start_animation(
        self, frames: int = None, save_path: Optional[str] = None
    ) -> None:
        """
        Start the animation.

        Args:
            frames: Number of frames to run (None for infinite)
            save_path: Path to save the animation to (None to not save)
        """
        # Set up animation with a finite number of frames if saving
        if save_path:
            frames = 100  # Use a reasonable number of frames for saving
            
        self.animation = animation.FuncAnimation(
            self.fig,
            self.update_frame,
            frames=frames,
            interval=self.update_interval,
            blit=False,  # Changed to False to ensure proper redrawing
            save_count=frames if save_path else None,
        )

        # Save animation if requested
        if save_path:
            try:
                # Try to use ffmpeg first with specific extra args
                writer = animation.FFMpegWriter(fps=30, bitrate=1800)
                self.animation.save(save_path, writer=writer)
            except Exception as e:
                print(f"FFmpeg writer failed: {e}")
                try:
                    # Fallback to default writer with fewer options
                    self.animation.save(save_path, fps=30)
                except Exception as e2:
                    print(f"Animation saving failed: {e2}")
                    print("Unable to save animation. Displaying only.")

        # Show the plot
        plt.show()

    def set_title(self, title: str) -> None:
        """
        Set the title of the plot.

        Args:
            title: The title to set
        """
        self.title = title  # Store the original title
        self.ax.set_title(title)

    def set_update_interval(self, interval: int) -> None:
        """
        Set the update interval for the animation.

        Args:
            interval: Update interval in milliseconds
        """
        self.update_interval = interval
        if self.animation:
            self.animation.event_source.interval = interval
    
    def toggle_agent_visualization(self) -> None:
        """
        Toggle the visualization of agent actions.
        Only applicable for agent-based game.
        """
        if isinstance(self.game, AgentBasedGameOfLife):
            self.show_agents = not self.show_agents
            
            # Redraw the frame
            self.update_frame(0)
            
    def toggle_policy_colors(self) -> None:
        """
        Toggle the visualization of policy colors.
        Only applicable for agent-based game with policies.
        """
        if isinstance(self.game, AgentBasedGameOfLife) and hasattr(self.game, 'use_policies'):
            self.show_policy_colors = not self.show_policy_colors
            
            # Redraw the frame
            self.update_frame(0)
