import os
import logging
from argparse import ArgumentParser
from typing import Tuple
import numpy as np
import pandas as pd
import imageio.v3 as iio
from skimage.morphology import dilation, disk, medial_axis, closing

from utils import configure_logger
import utils.image_logger as image_logger
import utils.output_dir_generator as output_generator
from cracks.derivates.bw_morph import endpoints as _endpoints, branches as _branches


def smooth_mask(crack_mask: np.ndarray, smooth_radius_px: int = 5) -> np.ndarray:
    """
    Smooth a binary crack mask using morphological closing with a disk footprint.

    :param crack_mask: Binary numpy ndarray of the crack mask.
    :param smooth_radius_px: Radius of the structuring element disk in pixels (0 for no smoothing).
    :return: Smoothed binary numpy ndarray.
    """
    if smooth_radius_px == 0:
        return crack_mask
    return closing(crack_mask, footprint=disk(smooth_radius_px))


def skeleton(crack_mask_smooth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 1px wide medial axis skeleton and distance-to-boundary transform.

    :param crack_mask_smooth: Smoothed binary crack mask.
    :return: Tuple of (skeleton_mask, distance_map).
    """
    return medial_axis(crack_mask_smooth, return_distance=True)


def skeleton_endpoints(crack_skeleton: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract endpoints from a 1px wide crack skeleton.

    :param crack_skeleton: Binary 2D numpy array representing the 1px skeleton.
    :return: Tuple of (endpoints_mask, endpoint_coords_array).
    """
    endpoints_mask = _endpoints(crack_skeleton)
    endpoint_coords = np.column_stack(np.where(endpoints_mask.T))
    return endpoints_mask, endpoint_coords


def skeleton_branches(cracks_skeleton: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split skeleton into distinct 1px wide branch paths by subtracting dilated node junctions.

    :param cracks_skeleton: Mask of the crack skeleton (1px wide).
    :return: Tuple of (skeleton_branches_mask, skeleton_nodes_mask, skeleton_nodes_coords).
    """
    # Identify intersection/junction node points where branch paths meet
    skeleton_nodes = _branches(cracks_skeleton)
    skeleton_nodes_coords = np.column_stack(np.where(skeleton_nodes.T))

    # Dilate junction nodes slightly (radius 2 disk) so subtracting removes the whole intersection region
    structuring_element = disk(2)
    dilated_nodes = dilation(skeleton_nodes, structuring_element)

    # Subtract the dilated intersection regions from the skeleton to isolate separate continuous branch segments
    skeleton_branches_mask = np.logical_and(cracks_skeleton, np.logical_not(dilated_nodes))

    return skeleton_branches_mask, skeleton_nodes, skeleton_nodes_coords


if __name__ == '__main__':
    parser = ArgumentParser(description="""
    Process crack mask into derivates: skeleton branches mask, skeleton endpoints mask,
    and skeleton nodes mask. Endpoints and nodes are also saved as coordinates.
    """)

    parser.add_argument(
        "crack_mask_img_path",
        type=str,
        help="Path to the input file. Expected a grayscale image with binary values {0,1} or {0,255}."
    )
    parser.add_argument(
        "-o",
        "--output-dir-path",
        type=str,
        help="Path to output directory. If directory does not exist, it will be created (parent must exist)."
    )
    parser.add_argument(
        "-s",
        "--mask-smoothing",
        type=int,
        help="Mask smoothing disk radius in pixels (0 disables smoothing).",
        default=5
    )
    args = parser.parse_args()

    output_dir_path, output_file_prefix = output_generator.prepare_output_path(args.output_dir_path)
    configure_logger(os.path.join(output_dir_path, f"{output_file_prefix}crack_derivates.log"))

    logging.debug(f"Reading input image {args.crack_mask_img_path}")
    cracks_mask = iio.imread(args.crack_mask_img_path)

    logging.debug(f"Smoothing crack mask with radius {args.mask_smoothing}.")
    cracks_mask_smooth = smooth_mask(cracks_mask, args.mask_smoothing)

    logging.debug("Creating skeletons of cracks.")
    crack_skeleton_mask, skeleton_to_boundary_distance = skeleton(cracks_mask_smooth)
    image_logger.dump_image(
        os.path.join(output_dir_path, f"{output_file_prefix}cracks-skeletons.png"),
        crack_skeleton_mask.astype(np.uint8)
    )
    image_logger.dump_image(
        os.path.join(output_dir_path, f"{output_file_prefix}cracks-skeletons-boundary-distances.tif"),
        skeleton_to_boundary_distance.astype(np.uint16)
    )

    skeletons_overview = np.zeros(cracks_mask.shape + (3,))
    skeletons_overview[cracks_mask.astype(bool), :] = np.array([255, 255, 255])
    skeletons_overview[crack_skeleton_mask.astype(bool), :] = np.array([255, 0, 0])
    image_logger.info(
        skeletons_overview,
        output_dir_path,
        f"{output_file_prefix}[user]cracks-skeletons-overview.png"
    )

    logging.debug("Extracting skeleton endpoints.")
    cracks_endpoints_derivates = skeleton_endpoints(crack_skeleton_mask)
    image_logger.dump_image(
        os.path.join(output_dir_path, f"{output_file_prefix}cracks-skeletons-endpoints-mask.png"),
        cracks_endpoints_derivates[0].astype(np.uint8)
    )
    pd.DataFrame(cracks_endpoints_derivates[1], columns=["x", "y"]).to_csv(
        os.path.join(output_dir_path, f"{output_file_prefix}cracks-skeletons-endpoints-coordinates.csv"),
        index=False
    )

    logging.debug("Cutting skeletons into branches.")
    cracks_branches_derivates = skeleton_branches(crack_skeleton_mask)
    image_logger.dump_image(
        os.path.join(output_dir_path, f"{output_file_prefix}cracks-skeletons-branches-mask.png"),
        cracks_branches_derivates[0].astype(np.uint8)
    )
    image_logger.dump_image(
        os.path.join(output_dir_path, f"{output_file_prefix}cracks-skeletons-nodes-mask.png"),
        cracks_branches_derivates[1].astype(np.uint8)
    )
    pd.DataFrame(cracks_branches_derivates[2], columns=["x", "y"]).to_csv(
        os.path.join(output_dir_path, f"{output_file_prefix}cracks-skeletons-nodes-coordinates.csv"),
        index=False
    )
