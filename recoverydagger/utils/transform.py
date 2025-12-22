"""
transform.py
Utility functions for coordinate transformations in maze environments.
"""

import numpy as np


def xy_to_cell_rowcol(self, x, y):

    i = int(-y + self.maze_height / 2)
    j = int(x + self.maze_width / 2)

    i = np.clip(i, 0, self.maze_height - 1)
    j = np.clip(j, 0, self.maze_width - 1)

    return i, j


def cell_rowcol_to_xy(self, i, j):
    y = -(i - self.maze_height / 2 + 0.5)
    x = j - self.maze_width / 2 + 0.5
    return x, y
