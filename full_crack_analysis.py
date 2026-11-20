import argparse
import logging
import os
from typing import Any, List, Optional, Sequence, Tuple, Union
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.measure import label

import cracks.derivates
import cracks.features
import utils

output_dir_path = "."


def through_chart(phase_through_ids: np.ndarray, phase_through_count: np.ndarray,
                  labels: Union[np.ndarray, List[str], Sequence[str]], sample_name: str,
                  output_file_prefix: str = "", output_dir_path: Optional[str] = None) -> None:
    """
    Draw and save a bar chart showing the number of crack pixels passing through each phase.

    :param phase_through_ids: Array of phase indices where cracks pass through.
    :param phase_through_count: Array of pixel counts corresponding to each phase index.
    :param labels: List or array of phase name strings.
    :param sample_name: Name of the sample to include in the plot title.
    :param output_file_prefix: Prefix for the output plot file name.
    :param output_dir_path: Output directory path to save the chart.
    """
    if output_dir_path is None:
        output_dir_path = globals().get("output_dir_path", ".")

    plt.figure(figsize=(7, 7))
    # Exclude invalid/background index at index 0 (phase id == -1)
    plt.bar(labels[phase_through_ids[1:]], phase_through_count[1:], width=0.7)
    plt.title(f"Crack pixels through phase {sample_name}")
    plt.xlabel("Phase name")
    plt.ylabel("Pixel count")
    plt.savefig(os.path.join(output_dir_path, f"{output_file_prefix}phase-through.png"))
    plt.close()


def edge_stacked_chart(phase_edge_pairs: Union[np.ndarray, List[Tuple[int, int]], Sequence[Any]],
                       phase_edge_count: Union[np.ndarray, Sequence[int]],
                       labels: Union[np.ndarray, List[str], Sequence[str]], sample_name: str,
                       output_file_prefix: str = "", output_dir_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draw and save a stacked bar chart showing crack pixels along boundaries between phase pairs.

    :param phase_edge_pairs: List or 2D array of (phase_A, phase_B) ID tuples representing phase boundaries.
    :param phase_edge_count: Pixel count for each phase edge pair.
    :param labels: List of phase name strings.
    :param sample_name: Name of the sample used in chart title.
    :param output_file_prefix: Prefix for the saved plot filename.
    :param output_dir_path: Directory path where the chart will be saved.
    :return: Tuple containing:
        - all_phases_used: Numpy array of all phase indices including -1 background.
        - edge_counts: 2D symmetric numpy array of edge pixel counts between all phase pairs.
    """
    if output_dir_path is None:
        output_dir_path = globals().get("output_dir_path", ".")

    all_phases_used = np.arange(-1, len(labels))
    edge_counts = np.zeros((all_phases_used.size, all_phases_used.size))

    # Populate symmetric adjacency count matrix for every pair of adjacent phases
    for phase_edge_pair, count in zip(phase_edge_pairs, phase_edge_count):
        if phase_edge_pair[0] in all_phases_used and phase_edge_pair[1] in all_phases_used:
            phase_a_idx = np.where(all_phases_used == phase_edge_pair[0])[0][0]
            phase_b_idx = np.where(all_phases_used == phase_edge_pair[1])[0][0]
            edge_counts[phase_a_idx, phase_b_idx] += count
            edge_counts[phase_b_idx, phase_a_idx] += count

    # Accumulate stacked bar offsets for each phase bar
    bars_bottom = np.zeros((len(labels),))
    plt.figure(figsize=(7, 7))
    for phase_order, phase_id in enumerate(all_phases_used):
        # Skip the invalid/unassigned background phase (-1)
        if phase_id < 0:
            continue
        plt.bar(
            np.array(labels),
            edge_counts[1:, phase_order],
            width=0.7,
            bottom=bars_bottom,
            label=labels[phase_id]
        )
        bars_bottom += edge_counts[1:, phase_order]

    plt.title(f"Crack neighbors - phase pairs {sample_name}", fontsize="xx-large")
    plt.xlabel("Phase name", fontsize="xx-large")
    plt.ylabel("Pixel count", fontsize="xx-large")
    plt.legend(fontsize="xx-large")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_path, f"{output_file_prefix}phase-edge.png"), transparent=True)
    plt.close()

    return all_phases_used, edge_counts


def phases_cracks_to_csv(labels: Union[np.ndarray, List[str], Sequence[str]], edge_counts: np.ndarray,
                         phase_through_ids: np.ndarray, phase_through_count: np.ndarray,
                         all_phases_used: np.ndarray, output_dir_path: str, output_file_prefix: str = "") -> None:
    """
    Export a 2D matrix of phase-to-phase crack pixel counts to a CSV file.

    Diagonal cells contain crack pixels passing through a single phase;
    off-diagonal cells contain crack pixels along the boundary between two phases.

    :param labels: Array of phase name strings.
    :param edge_counts: 2D array of edge pixel counts between phases.
    :param phase_through_ids: Array of phase IDs through which cracks pass.
    :param phase_through_count: Pixel counts corresponding to phase_through_ids.
    :param all_phases_used: Array of all phase indices including -1 background.
    :param output_dir_path: Output directory path to save the CSV.
    :param output_file_prefix: Prefix string for the CSV file name.
    """
    # Initialize count matrix excluding background row/col (-1)
    phases_cracks_count = edge_counts[1:, 1:].copy()

    # Set diagonal elements with through-phase pixel counts
    for phase_id, count in zip(phase_through_ids[1:], phase_through_count[1:]):
        if 0 <= phase_id < phases_cracks_count.shape[0]:
            phases_cracks_count[phase_id, phase_id] = count

    # Assemble output dataframe with columns: [phase id, label, <phase1>, <phase2>, ...]
    df_data = np.concatenate([
        all_phases_used[1:].reshape(1, -1),
        labels[all_phases_used[1:]].reshape(1, -1),
        phases_cracks_count.astype(int)
    ], axis=0).T

    col_names = ["phase id", "label"] + labels[all_phases_used[1:]].tolist()
    pd.DataFrame(df_data, columns=col_names).to_csv(
        os.path.join(output_dir_path, f"{output_file_prefix}phases_cracks.csv"),
        index=False
    )


def skeleton_neighbors_to_csv(gradient_map: np.ndarray, neigh_coords_per_px: np.ndarray,
                              smooth_mask: np.ndarray, branches_map: np.ndarray,
                              phase_map_img: np.ndarray, output_dir_path: str,
                              output_file_prefix: str = "") -> None:
    """
    Export detailed per-skeleton-pixel neighbor evaluation data to a CSV file.

    Outputs table containing: skeleton coordinates, crack component ID, branch ID,
    tangent gradient, left/right neighbor coordinates, and left/right neighbor phase IDs.

    :param gradient_map: 3D float numpy array (H, W, 2) of skeleton tangent gradients.
    :param neigh_coords_per_px: 4D int numpy array (H, W, 2 [left/right], 2 [y, x]) of neighbor coordinates.
    :param smooth_mask: 2D binary numpy array of smoothed cracks.
    :param branches_map: 2D integer numpy array of labeled branch IDs.
    :param phase_map_img: 2D integer numpy array of phase map labels.
    :param output_dir_path: Output directory path for the CSV.
    :param output_file_prefix: Prefix for the output CSV file name.
    """
    # Select skeleton pixels where both left (0) and right (1) neighbor coordinates are valid (>= 0)
    valid_mask = np.logical_and(
        np.logical_and(neigh_coords_per_px[:, :, 0, 0] >= 0, neigh_coords_per_px[:, :, 0, 1] >= 0),
        np.logical_and(neigh_coords_per_px[:, :, 1, 0] >= 0, neigh_coords_per_px[:, :, 1, 1] >= 0)
    )
    valid_neigh_coords = np.where(valid_mask)
    vy, vx = valid_neigh_coords

    # Extract left neighbor coordinates and phase IDs
    left_y = neigh_coords_per_px[vy, vx, 0, 0]
    left_x = neigh_coords_per_px[vy, vx, 0, 1]
    left_phase = phase_map_img[left_y, left_x]

    # Extract right neighbor coordinates and phase IDs
    right_y = neigh_coords_per_px[vy, vx, 1, 0]
    right_x = neigh_coords_per_px[vy, vx, 1, 1]
    right_phase = phase_map_img[right_y, right_x]

    crack_labels = label(smooth_mask, background=0)[valid_neigh_coords]
    branch_labels = branches_map[valid_neigh_coords]

    df = pd.DataFrame(np.stack([
        vx,  # crack skeleton coord X
        vy,  # crack skeleton coord Y
        crack_labels,
        branch_labels,
        gradient_map[vy, vx, 0],
        gradient_map[vy, vx, 1],
        left_x,
        left_y,
        right_x,
        right_y,
        left_phase,
        right_phase
    ], axis=1), columns=[
        "crack skeleton coord X",
        "crack skeleton coord Y",
        "crack ID",
        "branch ID",
        "gradient X",
        "gradient Y",
        "left neighbor X",
        "left neighbor Y",
        "right neighbor X",
        "right neighbor Y",
        "left neighbor phase ID",
        "right neighbor phase ID"
    ])

    df.to_csv(os.path.join(output_dir_path, f"{output_file_prefix}skeleton-neighbors.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    Toolbox for analysis of cracks in microscopic samples.
    """)

    parser.add_argument(
        "cracks_mask_path",
        type=str,
        help="Path to the cracks mask file."
    )
    parser.add_argument(
        "-f",
        "--features",
        action="append",
        type=str,
        required=True,
        help="Specify features to measure: [crack_features, phase_features]"
    )
    parser.add_argument(
        "--phase-map-path",
        type=str,
        help="Path to matching phase map file. Required if feature phase_features is enabled. "
             "Filtered phase map can be created by phase_map module.",
    )
    utils.cli_arguments.add_sample_name(parser)
    utils.config_loader.add_argparse_argument_phase_config_path(parser)
    utils.output_dir_generator.add_argparse_argument(parser)
    args = parser.parse_args()

    output_dir_path, output_file_prefix = utils.output_dir_generator.prepare_output_path(
        args.output_dir_path, args.sample_name
    )
    utils.configure_logger(os.path.join(output_dir_path, f"{output_file_prefix}full_crack_analysis.log"))
    logger = logging.getLogger(__name__)

    PHASES_CONFIG = utils.config_loader.Config.load_config(args.phase_config_path, "phases")

    cracks_mask = iio.imread(args.cracks_mask_path)
    smooth_mask = cracks.derivates.smooth_mask(cracks_mask)

    if "crack_features" in args.features:
        features = cracks.features.compute_cracks(smooth_mask)
        pd.DataFrame(features).to_csv(
            os.path.join(output_dir_path, f"{output_file_prefix}cracks_features.csv"),
            index=False
        )

    if "phase_features" in args.features:
        assert args.phase_map_path is not None, "Please specify --phase-map-path (@see -h)"
        phase_map_img = iio.imread(args.phase_map_path).astype(int)
        phase_map_through, phase_map_edge, neigh_coords_per_px, branches_map, gradient_map = cracks.features.phase_analysis(
            smooth_mask, phase_map_img
        )
        phase_through_ids, phase_through_count = np.unique(phase_map_through, return_counts=True)
        phase_edge_pairs, phase_edge_count = np.unique(
            np.sort(phase_map_edge.reshape(-1, 2), axis=1),
            axis=0,
            return_counts=True
        )

        labels = np.array(PHASES_CONFIG["labels"])

        # Generate summary charts
        through_chart(
            phase_through_ids, phase_through_count, labels,
            args.sample_name, output_file_prefix, output_dir_path
        )
        all_phases_used, edge_counts = edge_stacked_chart(
            phase_edge_pairs, phase_edge_count, labels,
            args.sample_name, output_file_prefix, output_dir_path
        )

        # Generate summary CSV files
        phases_cracks_to_csv(
            labels,
            edge_counts,
            phase_through_ids,
            phase_through_count,
            all_phases_used,
            output_dir_path,
            output_file_prefix
        )

        skeleton_neighbors_to_csv(
            gradient_map,
            neigh_coords_per_px,
            smooth_mask,
            branches_map,
            phase_map_img,
            output_dir_path,
            output_file_prefix
        )

    if "grain_features" in args.features:
        logger.error("Grain features not yet implemented.")
