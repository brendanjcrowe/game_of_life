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
from matplotlib.patches import Arrow, Circle

from game_of_life.game_logic import (
    DIRECTIONS,
    FORTIFY,
    ON,
    AgentBasedGameOfLife,
    GameOfLife,
)


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
        show_agents: bool = True
    ):
        """
        Initialize the visualizer.

        Args:
            game: The GameOfLife instance to visualize
            update_interval: Animation update interval in milliseconds
            cmap: Matplotlib colormap to use for rendering
            interpolation: Interpolation method for displaying the grid
            show_agents: Whether to show agent visualizations (for AgentBasedGameOfLife)
        """
        self.game = game
        self.update_interval = update_interval
        self.cmap = cmap
        self.interpolation = interpolation
        self.show_agents = show_agents and isinstance(game, AgentBasedGameOfLife)

        # Initialize the figure and axes
        self.fig, self.ax = plt.subplots()
        
        # Create a grid visualization
        if self.show_agents:
            # For agent-based, we'll use a custom visualization
            grid_display = self._create_agent_grid(self.game)
            self.img = self.ax.imshow(
                grid_display, interpolation=self.interpolation, cmap=self.cmap
            )
            
            # Add arrow indicators for actions
            self.arrow_patches = []
            self.fortify_patches = []
            self._update_agent_indicators()
        else:
            # For standard visualization, just show the grid
            self.img = self.ax.imshow(
                self.game.grid, interpolation=self.interpolation, cmap=self.cmap
            )

        # Initialize the animation object
        self.animation = None

    def _create_agent_grid(self, game: AgentBasedGameOfLife) -> np.ndarray:
        """
        Create a grid visualization for agent-based game.
        
        Args:
            game: The AgentBasedGameOfLife instance
            
        Returns:
            A 2D array suitable for visualization
        """
        # Create a copy of the grid for display
        grid_display = game.grid.copy()
        
        # Enhance fortified cells to be brighter
        for row, col in np.argwhere(game.fortified):
            if game.grid[row, col] == ON:
                # Make fortified cells brighter
                grid_display[row, col] = 255
                
        return grid_display

    def _update_agent_indicators(self) -> None:
        """
        Update the visual indicators for agent actions.
        """
        # Only applicable for agent-based game
        if not self.show_agents or not isinstance(self.game, AgentBasedGameOfLife):
            return
            
        # Clear existing patches
        for patch in self.arrow_patches:
            patch.remove()
        self.arrow_patches.clear()
        
        for patch in self.fortify_patches:
            patch.remove()
        self.fortify_patches.clear()
        
        agent_game = self.game  # type: AgentBasedGameOfLife
        
        # Get grid dimensions for proper scaling
        ny, nx = agent_game.grid.shape
        
        # Draw agent actions
        for row, col in np.argwhere(agent_game.grid == ON):
            action = agent_game.get_agent_action(row, col)
            
            if action == FORTIFY:
                # Draw a circle for fortify action
                circle = Circle(
                    (col, row), 
                    0.3, 
                    color='yellow', 
                    alpha=0.7,
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
                    color='red',
                    alpha=0.7
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
        if self.show_agents and isinstance(self.game, AgentBasedGameOfLife):
            # Update agent-based visualization
            grid_display = self._create_agent_grid(self.game)
            self.img.set_data(grid_display)
            
            # Update agent indicators
            self._update_agent_indicators()
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
            blit=True,
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
