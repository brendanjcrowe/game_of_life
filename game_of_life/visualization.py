"""
Game of Life - Visualization

This module contains visualization functionality for Conway's Game of Life.
It provides the Visualizer class to render and animate the game state.
"""

from typing import Optional, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt

from game_of_life.game_logic import GameOfLife


class Visualizer:
    """
    Visualizer for Conway's Game of Life.

    This class handles the rendering and animation of the Game of Life simulation.
    """

    def __init__(
        self,
        game: GameOfLife,
        update_interval: int = 50,
        cmap: str = "viridis",
        interpolation: str = "nearest",
    ):
        """
        Initialize the visualizer.

        Args:
            game: The GameOfLife instance to visualize
            update_interval: Animation update interval in milliseconds
            cmap: Matplotlib colormap to use for rendering
            interpolation: Interpolation method for displaying the grid
        """
        self.game = game
        self.update_interval = update_interval
        self.cmap = cmap
        self.interpolation = interpolation

        # Initialize the figure and axes
        self.fig, self.ax = plt.subplots()
        self.img = self.ax.imshow(
            self.game.grid, interpolation=self.interpolation, cmap=self.cmap
        )

        # Initialize the animation object
        self.animation = None

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
        self.animation = animation.FuncAnimation(
            self.fig,
            self.update_frame,
            frames=frames,
            interval=self.update_interval,
            blit=True,
            save_count=50 if save_path else None,
        )

        # Save animation if requested
        if save_path:
            self.animation.save(save_path, fps=30, extra_args=["-vcodec", "libx264"])

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
