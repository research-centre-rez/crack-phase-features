import argparse
import json
import os
from typing import Any, Dict, Optional, Union
import cv2
import imageio.v3 as iio
import numpy as np
from skimage.color import rgb2gray
from tqdm.auto import tqdm

import utils
import utils.image_logger as image_logger
import utils.output_dir_generator as output_generator

CRACKS_CONFIG: Optional[Dict[str, Any]] = None


def _load_input(file_path: str) -> np.ndarray:
    """
    Load an image from file and convert it into a single-channel 8-bit grayscale array.

    :param file_path: Path to the image file.
    :return: 2D uint8 numpy array of grayscale intensities.
    """
    img = iio.imread(file_path).astype(int)
    # Scale down 16-bit images to 8-bit range
    if np.max(img) > 255:
        img = (img // 256).astype(np.uint8)

    # Check dimensionality of the input (grayscale 2D, 1-channel, RGB or RGBA)
    if len(img.shape) == 2:
        return img.astype(np.uint8)

    if img.shape[2] == 1:
        return img[:, :, 0].astype(np.uint8)

    # Drop alpha channel from RGBA images
    if img.shape[2] == 4:
        img = img[:, :, :3]

    # Convert RGB image to grayscale [0..255]
    return (rgb2gray(img.astype(np.uint8), channel_axis=2) * 255).astype(np.uint8)


def threshold_image(gray_img: np.ndarray, config: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Create a false-color RGB image visualizing configured threshold levels.

    :param gray_img: Grayscale 2D uint8 image.
    :param config: Optional dictionary of cracks config (defaults to global CRACKS_CONFIG).
    :return: 3D RGB float/int numpy array of threshold bands.
    """
    if config is None:
        config = CRACKS_CONFIG
    colored = np.zeros(gray_img.shape + (3,))
    for th in config["threshold_levels"]:
        # Assign configured color to pixels falling within the lower and upper threshold bounds
        in_range = np.logical_and(th["range"][0] < gray_img, gray_img <= th["range"][1])
        colored[in_range, :] = np.array(th["color"])
    return colored


def normalize_histogram_left_and_right(img: np.ndarray, new_position_of_histogram_max: Union[int, float]) -> np.ndarray:
    """
    Apply CLAHE and perform piecewise linear normalization around the histogram mode.

    The histogram mode (peak) is shifted to `new_position_of_histogram_max`,
    scaling values to the left ([0..mode]) and right ([mode..255]) proportionally.

    :param img: 2D uint8 input image.
    :param new_position_of_histogram_max: Desired target position for the histogram peak [0..255].
    :return: 2D float ndarray of normalized image intensities.
    """
    # Enhance local contrast using Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(5, 5))
    img_clahe = clahe.apply(img)
    out = np.copy(img_clahe).astype(float)

    # Find the intensity with maximum frequency (histogram mode)
    histogram_counts, _ = np.histogram(img_clahe.reshape(-1), bins=256)
    histogram_max_pos = np.argmax(histogram_counts)

    # Left partition: scale range [0, histogram_max_pos] -> [0, new_position_of_histogram_max]
    left_mask = out <= histogram_max_pos
    if histogram_max_pos > 0:
        out[left_mask] = img_clahe[left_mask] / histogram_max_pos * new_position_of_histogram_max

    # Right partition: scale range (histogram_max_pos, 255] -> (new_position_of_histogram_max, 255]
    right_mask = out > histogram_max_pos
    if histogram_max_pos < 255:
        out[right_mask] = (
            (out[right_mask] - histogram_max_pos) / (255 - histogram_max_pos)
            * (255 - new_position_of_histogram_max) + new_position_of_histogram_max
        )

    return out


def crack_mask_by_thresholds(input_path: str, output_dir_path: str, output_file_prefix: str = "") -> np.ndarray:
    """
    Segment crack mask using dual-threshold hysteresis flood-fill.

    Thresholds are applied as follows:
    - 1st threshold band acts as definitive "seeds" (candidate value > 1).
    - 2nd threshold band acts as "seed potential" (candidate value == 1).
    Pixels in the potential band are included if connected to seeds via forward and backward raster sweeps.

    :param input_path: Path to the input grayscale image.
    :param output_dir_path: Output directory path where debug and result images will be saved.
    :param output_file_prefix: Prefix string for generated files.
    :return: 2D uint8 binary mask of detected cracks (1 = crack, 0 = background).
    """
    # Load input image
    gray_img = _load_input(input_path)
    image_logger.info(gray_img, output_dir_path, f"{output_file_prefix}[user]input-grayscale.png")

    # Normalize histogram
    gray_norm = normalize_histogram_left_and_right(gray_img, CRACKS_CONFIG["median_level"])
    image_logger.info(gray_norm, output_dir_path, f"{output_file_prefix}[user]input-normalized.png")

    # Apply thresholds visualization
    colored = threshold_image(gray_norm)
    image_logger.info(colored, output_dir_path, f"{output_file_prefix}[user]colored.png")

    # Create crack candidate mask
    # th0 defines seed range (value becomes 2), th2 defines potential range (value becomes 1)
    th0 = CRACKS_CONFIG["threshold_levels"][0]["range"]
    th2 = CRACKS_CONFIG["threshold_levels"][1]["range"]
    candidates = (gray_norm < th2[1]).astype(np.uint8) + np.logical_and(th0[0] < gray_norm, gray_norm < th0[1]).astype(np.uint8)

    # Forward flood fill pass (top-left to bottom-right): propagate seeds to adjacent potentials
    for y in tqdm(range(candidates.shape[0]), desc="Flood fill forward"):
        for x in range(candidates.shape[1]):
            if candidates[y, x] == 1 and (
                    (y - 1 >= 0 and candidates[y - 1, x] > 1) or
                    (x - 1 >= 0 and candidates[y, x - 1] > 1)):
                candidates[y, x] += 1

    # Backward flood fill pass (bottom-right to top-left): propagate seeds in reverse direction
    for y in tqdm(np.arange(candidates.shape[0])[::-1], desc="Flood fill backward"):
        for x in np.arange(candidates.shape[1])[::-1]:
            if candidates[y, x] == 1 and (
                    (y + 1 < candidates.shape[0] and candidates[y + 1, x] > 1) or
                    (x + 1 < candidates.shape[1] and candidates[y, x + 1] > 1)):
                candidates[y, x] += 1

    # Final mask contains all confirmed seeds (values > 1)
    mask = (candidates > 1).astype(np.uint8)
    image_logger.info(mask * 255, output_dir_path, f"{output_file_prefix}[user]crack-mask.png")
    image_logger.dump_image(os.path.join(output_dir_path, f"{output_file_prefix}crack-mask.png"), mask)

    return mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    Preprocessing of the crack scans. Target of this routine is to create crack mask.
    """)
    parser.add_argument(
        "input_image_path",
        type=str,
        help="Path to grayscale input image where cracks should be detected.")
    utils.output_dir_generator.add_argparse_argument(parser)
    utils.config_loader.add_argparse_argument_cracks_config_path(parser)
    utils.cli_arguments.add_sample_name(parser)
    args = parser.parse_args()

    output_dir_path, output_file_prefix = output_generator.prepare_output_path(args.output_dir_path, args.sample_name)
    utils.configure_logger(os.path.join(output_dir_path, f"{output_file_prefix}cracks_mask.log"))

    utils.config_loader.Config.load_config(args.cracks_config_path, "cracks")
    CRACKS_CONFIG = utils.config_loader.Config.get_config("cracks")
    with open(os.path.join(output_dir_path, f"{output_file_prefix}cracks_config_used.json"), "wt") as cfg_out:
        json.dump(CRACKS_CONFIG, cfg_out)

    crack_mask_by_thresholds(
        input_path=args.input_image_path,
        output_dir_path=output_dir_path,
        output_file_prefix=output_file_prefix
    )
