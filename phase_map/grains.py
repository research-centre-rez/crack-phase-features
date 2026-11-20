import logging
import os
from typing import Optional, Tuple
import numpy as np
import skimage.measure
from skimage.morphology import disk, opening, dilation, label
from tqdm.auto import tqdm

import utils.image_logger as image_logger

GRAIN_STRUCT_ELEMENT = disk(10)
SUBGRAIN_STRUCT_ELEMENT = disk(5)
logger = logging.getLogger(__name__)


def segment_grains(phase_map: np.ndarray, output_dir_path: Optional[str] = None, output_file_prefix: str = "") -> np.ndarray:
    """
    Segment individual grains in phase map and refine boundaries using mathematical morphology.

    Processing steps:
    1. Label connected grain components for each phase.
    2. Apply morphological opening to eliminate narrow thread artifacts.
    3. Dilate separated subgrains to restore natural grain boundaries.
    4. Detect residual unattached leftovers via XOR; assign them to the neighboring grain with maximum contact outline.

    :param phase_map: 2D numpy array of the segmented phase map.
    :param output_dir_path: Output directory for image dumps.
    :param output_file_prefix: Prefix for saved file names.
    :return: 2D uint16 numpy array representing refined grain IDs.
    """
    logger.info("Computation of the grain set started.")

    # Step 1: Label individual grains in the input phase map (excluding 0 background)
    labeled_phase_map = label(phase_map, background=0)

    # Step 2: Morphological opening (erosion followed by dilation) to detach narrow connecting threads
    grains_open = opening(labeled_phase_map, GRAIN_STRUCT_ELEMENT)

    # Step 3: Relabel opened grains and dilate back to fill boundaries (masked by original phase regions)
    sub_grains = label(grains_open, background=0)
    the_grains = dilation(sub_grains, SUBGRAIN_STRUCT_ELEMENT) * (labeled_phase_map > 0).astype(int)

    # Step 4: Identify unattached leftover fragments (regions in labeled_phase_map not covered by dilated grains)
    leftovers = label(np.logical_xor(labeled_phase_map > 0, the_grains.astype(bool)))
    leftovers_dilated = dilation(leftovers, disk(2))
    leftovers_outline = np.logical_xor(leftovers, leftovers_dilated)
    leftovers_outline_grain_overlap = the_grains * leftovers_outline

    # Zero out unattached leftover positions before reassigning
    the_grains[leftovers != 0] = 0

    # Step 5: For each leftover fragment with outline intersecting surrounding grains, assign to majority grain
    attachable_leftovers = np.unique((leftovers_outline_grain_overlap != 0) * leftovers_dilated)[1:]
    for leftover_id in tqdm(attachable_leftovers, total=attachable_leftovers.size, desc="Grain refine (filling leftovers)"):
        leftover_mask = leftovers_dilated == leftover_id
        grain_ids, counts = np.unique(leftover_mask * leftovers_outline_grain_overlap, return_counts=True)
        assert grain_ids.size != 1, f"{grain_ids} attached to this leftover outline do not match outline intersection with grains"
        # Select neighbor grain ID with the highest outline contact pixel count
        valid_grain_ids = grain_ids[grain_ids != 0]
        valid_counts = counts[grain_ids != 0]
        attached_grain_id = valid_grain_ids[np.argsort(valid_counts)[-1]]
        the_grains[leftover_mask] = attached_grain_id

    max_grain = int(np.max(the_grains)) if the_grains.size > 0 else 0
    logger.info(f"Computation of the grain set done. Total {max_grain} grains.")
    image_logger.info(the_grains, output_dir_path, f"{output_file_prefix}[user]grain_map.tiff")
    if output_dir_path is not None:
        image_logger.dump_image(
            os.path.join(output_dir_path, f"{output_file_prefix}grain_map.tiff"),
            the_grains.astype(np.uint16)
        )

    return the_grains


def grain_size_filter(grain_map: np.ndarray, grain_size_px_limit: int = 9000, output_dir_path: Optional[str] = None, output_file_prefix: str = "") -> Tuple[np.ndarray, int, int]:
    """
    Filter out grains smaller than a given pixel threshold and reclassify them as matrix phase.

    :param grain_map: 2D numpy array of grain IDs.
    :param grain_size_px_limit: Minimum area threshold in pixels for retaining a grain.
    :param output_dir_path: Output directory for image dumps.
    :param output_file_prefix: Prefix for saved file names.
    :return: Tuple of:
        - grain_map_size_limit: 2D array of filtered grains with small grains converted to matrix IDs.
        - grain_size_px_limit: The threshold value used.
        - non_matrix_grains_count: Total count of retained large (non-matrix) grains.
    """
    logger.info("Grain filter started.")

    grain_idx, grain_sizes_px = np.unique(grain_map.ravel(), return_counts=True)

    # Filter grain IDs with area >= limit
    grain_ids_bigger_than_limit = [
        grain_id
        for grain_id, grain_size_px in zip(grain_idx, grain_sizes_px)
        if grain_size_px >= grain_size_px_limit
    ]

    # Retain only grains larger than limit and relabel them sequentially
    grain_map_size_limit = np.zeros_like(grain_map)
    grain_map_size_limit[np.isin(grain_map, grain_ids_bigger_than_limit)] = 1
    grain_map_size_limit = label(grain_map_size_limit, background=0)
    non_matrix_grains_count = int(np.max(grain_map_size_limit))
    image_logger.info(
        (grain_map_size_limit != 0).astype(float),
        output_dir_path,
        f"{output_file_prefix}[user]non_matrix_grains.png"
    )

    # The background / small grain regions form the matrix phase; label distinct matrix zones
    matrix_grains, matrix_grains_count = skimage.measure.label(
        grain_map_size_limit == 0,
        background=0,
        return_num=True
    )
    image_logger.info(
        (matrix_grains != 0).astype(float),
        output_dir_path,
        f"{output_file_prefix}[user]matrix_grains.png"
    )

    # Shift matrix grain IDs so they don't collide with retained non-matrix grain IDs
    matrix_grains[matrix_grains != 0] += non_matrix_grains_count
    grain_map_size_limit[grain_map_size_limit == 0] = matrix_grains[matrix_grains > 0]

    logger.info(
        f"Grain filter done. Total {non_matrix_grains_count} grains and {matrix_grains_count} matrix grains."
    )
    image_logger.info(grain_map_size_limit, output_dir_path, f"{output_file_prefix}[user]grain_map_filtered.tiff")
    if output_dir_path is not None:
        image_logger.dump_image(
            os.path.join(output_dir_path, f"{output_file_prefix}grain_map_filtered.tiff"),
            grain_map_size_limit.astype(np.uint16)
        )

    return grain_map_size_limit, grain_size_px_limit, non_matrix_grains_count
