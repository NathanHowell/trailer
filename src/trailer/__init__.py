"""trailer -- LiDAR trail detection for OpenStreetMap.

Builds training tiles from USGS 3DEP point clouds and OSM trail geometry, for a
segmentation model whose output is a probability heatmap a human reviews in
JOSM rather than an automatic import.
"""

__version__ = "0.1.0"
