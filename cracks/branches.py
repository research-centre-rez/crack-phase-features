import logging
from typing import List, Optional, Tuple, Union, Sequence
import numpy as np

logger = logging.getLogger(__name__)


def _find_path_start(binary_mask: np.ndarray) -> Union[np.ndarray, Tuple[int, int]]:
    """
    Find an endpoint pixel (having exactly 1 neighbor) to use as the starting pixel of a branch path.

    :param binary_mask: 2D binary numpy array of the branch.
    :return: (y, x) coordinate of the start pixel.
    """
    path_pixels = np.argwhere(binary_mask == 1)
    # 8-connected neighborhood offset directions
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (-1, 1), (1, -1), (-1, -1)]

    for pixel in path_pixels:
        neighbors = 0
        for d in directions:
            neighbor = (pixel[0] + d[0], pixel[1] + d[1])
            if (0 <= neighbor[0] < binary_mask.shape[0] and
                    0 <= neighbor[1] < binary_mask.shape[1] and
                    binary_mask[neighbor] == 1):
                neighbors += 1
        # Pixel with only one neighbor is a branch endpoint/tip
        if neighbors == 1:
            return pixel

    # Fallback to first path pixel if no endpoint is found (e.g., closed cycle/loop)
    return path_pixels[0]


def sort_branch_pixels(binary_mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Order the connected pixels of a 1px wide branch mask into a sequential coordinate path.

    :param binary_mask: 2D binary numpy array of the branch pixels.
    :return: List of (y, x) tuple coordinates ordered sequentially along the branch path.
    """
    # Find initial starting point (endpoint if available)
    start_pixel = _find_path_start(binary_mask)
    # 8-connected neighborhood directions
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (-1, 1), (1, -1), (-1, -1)]

    ordered_pixels = []
    visited = set()

    # Depth-First Search (DFS) stack traversal to follow the branch sequentially
    stack = [start_pixel]
    while stack:
        px = stack.pop()
        px_tuple = tuple(px)
        if px_tuple in visited:
            continue
        visited.add(px_tuple)
        ordered_pixels.append(px_tuple)

        # Inspect all 8 surrounding neighbors
        for d in directions:
            neighbor = (px[0] + d[0], px[1] + d[1])
            if (0 <= neighbor[0] < binary_mask.shape[0] and
                    0 <= neighbor[1] < binary_mask.shape[1] and
                    binary_mask[neighbor] == 1 and
                    neighbor not in visited):
                stack.append(neighbor)

    return ordered_pixels


def path_direction(branch_ordered_pixels: Union[List[Tuple[int, int]], np.ndarray, Sequence[Tuple[int, int]]], mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Compute tangent gradient vectors along an ordered sequence of branch pixels.

    For paths with at least 2 pixels, central differences are computed using numpy.gradient.
    For single-pixel paths, local neighboring crack pixels in the mask are used to estimate orientation.

    :param branch_ordered_pixels: List of (y, x) coordinates of ordered pixels along the branch.
    :param mask: Reference mask image (crack mask or phase map) for boundary inspection.
    :return: Numpy array of shape (N, 2) containing gradient vectors (dy, dx), or None if undefined.
    """
    if len(branch_ordered_pixels) < 2:
        # Single pixel case: cannot compute direct gradient, look at 3x3 local patch in the mask
        y_min = max(0, branch_ordered_pixels[0][0] - 1)
        y_max = min(branch_ordered_pixels[0][0] + 2, mask.shape[0])
        x_min = max(0, branch_ordered_pixels[0][1] - 1)
        x_max = min(branch_ordered_pixels[0][1] + 2, mask.shape[1])

        crack_nearby = mask[y_min:y_max, x_min:x_max]
        nearby_crack_coords = np.stack(np.where(crack_nearby == 0), axis=1)
        nearby_crack_pixels_count = nearby_crack_coords.shape[0]

        if nearby_crack_pixels_count == 1:
            # Isolated pixel without orientation
            logger.error("Invalid crack branch (single pixel). No gradient.")
            return None
        elif nearby_crack_pixels_count == 2:
            # 2 nearby pixels: forward difference
            gradient = np.gradient(nearby_crack_coords, axis=0)[:1]
        elif nearby_crack_pixels_count == 3:
            # 3 nearby pixels: central difference around the middle pixel
            gradient = np.gradient(nearby_crack_coords, axis=0)[1:2]
        else:
            # Ambiguous orientation (more than 2 neighbors)
            return None
    else:
        # Standard gradient computation along ordered pixel path
        gradient = np.gradient(branch_ordered_pixels, axis=0)

    return gradient
