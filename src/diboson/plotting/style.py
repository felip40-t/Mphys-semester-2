"""Shared visual constants for physics heatmap and contour plots."""

from matplotlib.colors import LinearSegmentedColormap

FIGSIZE_HEATMAP = (12, 10)
FONTSIZE_LABEL = 16
FONTSIZE_TICK = 14
FONTSIZE_ANNOTATION = 12

CONCURRENCE_CMAP = LinearSegmentedColormap.from_list(
    'concurrence_cmap', ['darkblue', 'blue', 'purple', 'red'], N=256
)
