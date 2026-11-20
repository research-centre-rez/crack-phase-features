import logging
from typing import Any, Dict, List, Tuple
import numpy as np
from skimage.measure import label
from skimage.segmentation import mark_boundaries
from tqdm.auto import tqdm

from cracks.branches import sort_branch_pixels, path_direction
import cracks.derivates

logger = logging.getLogger(__name__)


def attach_neigh_phases_to_skeleton_branch(branch_mask: np.ndarray, phase_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find left and right adjacent phase IDs for each pixel along a skeleton branch.

    For each skeleton pixel, marches outwards along the orthogonal normal vector
    to the branch tangent until non-crack phase pixels are hit or image bounds are reached.

    :param branch_mask: 2D binary numpy array representing a single skeleton branch.
    :param phase_map: 2D int numpy array of phase IDs (-2 represents crack areas).
    :return: Tuple of:
        - phases_found: Array of shape (N, 2) containing left and right phase IDs (-1 for invalid/out-of-bounds).
        - neighbors: Array of shape (N, 2, 2) containing left and right neighbor (y, x) coordinates.
        - gradient: Array of shape (N, 2) containing normalized tangent unit vectors along the branch.
    """
    branch_ordered_pixels = sort_branch_pixels(branch_mask)
    gradient = path_direction(branch_ordered_pixels, phase_map)
    if gradient is None:
        return np.array([]), np.array([]), np.array([])

    # Normalize tangent gradient vectors to unit length
    gradient_norms = np.linalg.norm(gradient, axis=1, keepdims=True)
    # Avoid division by zero
    gradient_norms[gradient_norms == 0] = 1.0
    gradient = gradient / gradient_norms

    # Compute orthogonal normal vectors by 90-degree 2D rotation: [dy, dx] @ [[0, -1], [1, 0]] = [dx, -dy]
    norm = np.matmul(gradient, [[0, -1], [1, 0]])

    # Initialize tracking arrays: -2 indicates search still in progress, -1 indicates invalid/out-of-bounds
    phases_found = -2 * np.ones((len(branch_ordered_pixels), 2))
    neighbors = -np.ones((len(branch_ordered_pixels), 2, 2))  # shape: (N, 2 [left/right], 2 [y, x])

    # Step outwards iteratively along the normal vectors until non-crack phase pixels are reached
    steps = 1
    while not np.all(phases_found != -2):
        neighbors_done = phases_found != -2
        left_neighbor = np.round(branch_ordered_pixels + steps * norm).astype(int)
        right_neighbor = np.round(branch_ordered_pixels - steps * norm).astype(int)

        steps += 1
        for nid, neighbor in enumerate([left_neighbor, right_neighbor]):
            # Check for coordinates falling outside image bounds
            out_of_range = np.logical_or(
                np.logical_or(neighbor[:, 0] < 0, neighbor[:, 0] >= phase_map.shape[0]),
                np.logical_or(neighbor[:, 1] < 0, neighbor[:, 1] >= phase_map.shape[1])
            )

            # If neighbor reached boundary without hitting a valid phase, mark as invalid (-1)
            to_be_set_invalid = np.logical_and(out_of_range, ~neighbors_done[:, nid])
            phases_found[to_be_set_invalid, nid] = -1
            neighbors[to_be_set_invalid, nid, :] = -1

            # If inside bounds, check if a valid phase (> -2, i.e., not a crack) was found
            to_be_set = np.logical_and(~out_of_range, ~neighbors_done[:, nid])
            found_phases = phase_map[neighbor[to_be_set, 0], neighbor[to_be_set, 1]]
            # Only finalize if the pixel is not another crack pixel (-2)
            valid_phase_mask = found_phases != -2
            if np.any(valid_phase_mask):
                valid_indices = np.where(to_be_set)[0][valid_phase_mask]
                phases_found[valid_indices, nid] = found_phases[valid_phase_mask]
                neighbors[valid_indices, nid, :] = neighbor[valid_indices, :]

    return phases_found.astype(int), neighbors.astype(int), gradient


def phase_analysis(crack_mask_smooth: np.ndarray, phase_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform full neighbor phase analysis across all skeleton branches.

    Classifies skeleton pixels into:
    - 'through': crack runs inside a single phase (left phase == right phase).
    - 'edge': crack runs along phase boundary (left phase != right phase).

    :param crack_mask_smooth: Smoothed 2D binary crack mask.
    :param phase_map: 2D integer phase map image.
    :return: Tuple of (through, edge, neighbors_map, branches_map, gradient_map).
    """
    # Extract skeleton and branch components
    skeleton_img, _ = cracks.derivates.skeleton(crack_mask_smooth)
    branches_mask, _, _ = cracks.derivates.skeleton_branches(skeleton_img)
    branches_map = label(branches_mask, background=0)
    branches_list = np.unique(branches_map)

    neighbors_map = -2 * np.ones(branches_map.shape + (2, 2))
    gradient_map = np.zeros(branches_map.shape + (2,))

    # Erase phase information at crack locations to avoid self-referencing
    phase_map_copy = phase_map.copy()
    phase_map_copy[crack_mask_smooth.astype(bool)] = -2

    through = -np.ones_like(phase_map_copy, dtype=int)
    edge = -np.ones(phase_map_copy.shape + (2,), dtype=int)

    # First ID (0) corresponds to background, so iterate from 1 upwards
    for branch_id in tqdm(branches_list[1:], total=len(branches_list) - 1, desc="Phase left/right (branches)"):
        branch_mask = branches_map == branch_id
        phases_per_px, neigh_coords_per_px, gradient = attach_neigh_phases_to_skeleton_branch(
            branch_mask, phase_map_copy
        )
        if phases_per_px.size > 0:
            coords = np.where(branch_mask)
            xx, yy = coords
            neighbors_map[coords] = neigh_coords_per_px
            gradient_map[coords] = gradient
            for phases, x, y in zip(phases_per_px, xx, yy):
                if phases[0] == phases[1]:
                    # Crack passes through a single phase
                    through[x, y] = phases[0]
                else:
                    # Crack lies on the edge between two distinct phases
                    edge[x, y, :] = phases

    return through, edge, neighbors_map.astype(int), branches_map, gradient_map


def compute_cracks(crack_mask_smooth: np.ndarray) -> List[Dict[str, Any]]:
    """
    Compute geometric and topological features for all connected crack components.

    :param crack_mask_smooth: Binary mask indicating location of cracks (1 = crack, 0 = background).
    :return: List of feature dicts for each connected crack component:
        - label: Unique integer identifier of the crack component.
        - crackSize_px: Total area of the crack in pixels.
        - skeleton_px: Total number of skeleton pixels in the crack.
        - maxWidth_px: Maximum estimated width (2 * distance transform value).
        - avgWidth_px: Mean estimated width along the skeleton.
        - boundaryLength_px: Perimeter boundary length in pixels.
        - farthestPoints_px: Euclidean distance between bounding box extreme corners.
    """
    if np.sum(crack_mask_smooth) == 0:
        return []

    cracks_skeletons, skeleton_to_boundary_distance = cracks.derivates.skeleton(crack_mask_smooth)
    mask_labeled = label(crack_mask_smooth, background=0)
    crack_labels, cracks_areas_px = np.unique(mask_labeled, return_counts=True)

    if len(crack_labels) <= 1:
        return []

    out = []
    # Skip background label (index 0)
    for crack_label, crack_area_px in tqdm(
        zip(crack_labels[1:], cracks_areas_px[1:]),
        total=len(crack_labels) - 1,
        desc="Computing crack features"
    ):
        crack_smooth = mask_labeled == crack_label
        skeleton_comp = np.logical_and(cracks_skeletons, crack_smooth)
        distance = skeleton_to_boundary_distance * skeleton_comp

        if not np.any(skeleton_comp):
            avg_width = 0.0
            max_width = 0.0
        else:
            max_width = float(np.max(distance))
            avg_width = float(np.mean(distance[distance != 0]))

        # Calculate bounding diagonal length across extreme crack pixels
        y_coords, x_coords = np.where(crack_smooth)
        length = float(np.sqrt(
            (np.max(x_coords) - np.min(x_coords)) ** 2 +
            (np.max(y_coords) - np.min(y_coords)) ** 2
        ))

        # Crop patch around the crack component to speed up boundary extraction
        top_left_y, top_left_x = np.min(y_coords), np.min(x_coords)
        bottom_right_y, bottom_right_x = np.max(y_coords) + 1, np.max(x_coords) + 1

        crack_patch = crack_smooth[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        label_patch = mask_labeled[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

        # Use scikit-image mark_boundaries on the padded patch to extract boundary pixel count
        marked = mark_boundaries(
            np.pad(crack_patch, ((1, 1), (1, 1))),
            np.pad(label_patch, ((1, 1), (1, 1))),
            outline_color=(0.5, 0, 0),
            mode="outer"
        )
        boundary_coords = np.where(marked[:, :, 0] == 0.5)

        out.append({
            "label": int(crack_label),
            "crackSize_px": int(crack_area_px),
            "skeleton_px": int(np.sum(skeleton_comp)),
            "maxWidth_px": max_width,
            "avgWidth_px": avg_width,
            "boundaryLength_px": int(len(boundary_coords[0])),
            "farthestPoints_px": length
        })

    return out
