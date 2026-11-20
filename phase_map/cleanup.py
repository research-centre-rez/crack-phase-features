import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import skimage.color
from skimage.color import rgb2lab
from skimage.exposure import rescale_intensity
from skimage.filters import gaussian
from tqdm.auto import tqdm

import utils.image_logger as image_logger

logger = logging.getLogger(__name__)


def _phases_selected(key: str, config: Dict[str, Any]) -> List[Any]:
    """
    Filter out configuration list values for phases where 'detect' flag is True.

    :param key: Key name in config dictionary (e.g., 'colors', 'labels').
    :param config: Phases configuration dictionary.
    :return: Filtered list of values corresponding to detected phases.
    """
    return [
        value
        for value, should_detect in zip(config[key], config["detect"])
        if should_detect
    ]


def apply_threshold(phase_map: np.ndarray, config: Dict[str, Any], lightness_clip: Optional[Union[List[int], Tuple[int, ...]]] = None,
                    adjust_intensity: Union[str, Tuple[float, float], List[float]] = "image",
                    gaussian_blur: float = 0, distance_metric: Optional[str] = None, output_dir: Optional[str] = None,
                    output_file_prefix: str = "") -> np.ndarray:
    """
    Clean and segment raw EDS phase map by classifying pixels to reference phase colors.

    Processing steps:
    1. Rescale RGB intensity range according to `adjust_intensity`.
    2. Apply Gaussian blur smoothing if `gaussian_blur` > 0.
    3. Convert to CIE L*a*b* color space.
    4. Compute color distance to each reference phase and assign nearest phase ID (argmin).
    5. Zero out pixels falling outside the `lightness_clip` range.

    :param phase_map: RGB input image (H, W, 3) of the raw phase map.
    :param config: Phase configuration dictionary containing 'colors', 'labels', and 'detect'.
    :param lightness_clip: Tuple or list of [min_L, max_L] lightness thresholds (defaults to [1, 99]).
    :param adjust_intensity: Range specification for skimage.exposure.rescale_intensity.
    :param gaussian_blur: Float sigma value for Gaussian filtering (0 disables blur).
    :param distance_metric: Color distance metric: 'lab', 'ab', or None (CIE DE2000).
    :param output_dir: Directory path for debug image logging.
    :param output_file_prefix: Prefix string for logged image file names.
    :return: 2D integer numpy array (H, W) where each pixel holds its assigned phase index.
    """
    if lightness_clip is None:
        lightness_clip = [1, 99]

    if output_dir is not None:
        cfg_path = os.path.join(output_dir, f"{output_file_prefix}phases_config_used.json")
        with open(cfg_path, "wt") as cfg_file:
            json.dump(config, cfg_file)
        image_logger.info(phase_map, output_dir, f"{output_file_prefix}phase_map_rgb.png")

    # Step 1: Rescale image intensity into specified range
    img = rescale_intensity(phase_map, in_range=adjust_intensity)
    image_logger.info(img, output_dir, f"{output_file_prefix}phase_map_adjusted.png")

    # Step 2: Gaussian blur smoothing
    if gaussian_blur != 0:
        img = gaussian(img, sigma=gaussian_blur, channel_axis=2)
        image_logger.info(img, output_dir, f"{output_file_prefix}phase_map_gaussian_blur.png")

    # Step 3: Convert RGB image and reference colors into CIE L*a*b* space
    img_lab = rgb2lab(img)
    phases_color_rgb = np.array(_phases_selected("colors", config)).astype(np.uint8)
    phases_color_lab = rgb2lab(phases_color_rgb)
    image_logger.info(img_lab, output_dir, f"{output_file_prefix}phase_img_lab.png")

    # Step 4: Classify each pixel to closest reference phase color
    distances = color_distance(img_lab, phases_color_lab, mode=distance_metric)
    phase_map_img = np.argmin(distances, axis=2)

    # Step 5: Zero out pixels with lightness outside valid range (e.g. pores / dark cracks / overexposure)
    phase_map_img[img_lab[:, :, 0] < lightness_clip[0]] = 0
    if len(lightness_clip) > 1:
        phase_map_img[img_lab[:, :, 0] > lightness_clip[1]] = 0

    image_logger.info(phases_color_rgb[phase_map_img], output_dir, f"{output_file_prefix}[user]phase_map.png")
    if output_dir is not None:
        image_logger.dump_image(os.path.join(output_dir, f"{output_file_prefix}phase_map.png"), phase_map_img.astype(np.uint8))

    return phase_map_img


def color_distance(img_lab: np.ndarray, reference_colors_lab: np.ndarray, mode: Optional[str] = None) -> np.ndarray:
    """
    Compute pixel-wise color distance between an L*a*b* image and reference phase colors.

    :param img_lab: 3D numpy array of shape (H, W, 3) in CIE L*a*b* color space.
    :param reference_colors_lab: 2D numpy array of shape (N, 3) containing reference L*a*b* colors.
    :param mode: Distance mode:
        - 'lab' or 'La*b*-Euclidean': Euclidean norm in full 3D L*a*b* space.
        - 'ab' or 'a*b*-Euclidean': Euclidean norm in 2D chromaticity a*b* plane.
        - None: CIEDE2000 perceptual color difference formula.
    :return: 3D numpy array of shape (H, W, N) containing computed color distances.
    """
    h, w = img_lab.shape[:2]
    num_phases = len(reference_colors_lab)

    if mode in ["lab", "La*b*-Euclidean"]:
        # Euclidean distance in 3D (L*, a*, b*) space
        distances = np.array([
            np.linalg.norm(img_lab.reshape(-1, 3) - phase_color_lab, axis=1)
            for phase_color_lab in reference_colors_lab
        ]).T.reshape(h, w, num_phases)
    elif mode in ["ab", "a*b*-Euclidean"]:
        # Euclidean distance ignoring Lightness, using only chromaticity channels (a*, b*)
        distances = np.array([
            np.linalg.norm(img_lab[:, :, 1:].reshape(-1, 2) - phase_color_lab[1:], axis=1)
            for phase_color_lab in reference_colors_lab
        ]).T.reshape(h, w, num_phases)
    else:
        # CIE DE2000 perceptual color distance
        distances = np.stack([
            skimage.color.deltaE_ciede2000(
                reference_color_lab,
                img_lab.reshape(-1, 3).T
            ).reshape(h, w)
            for reference_color_lab in tqdm(
                reference_colors_lab,
                total=num_phases,
                desc="Computing CIE DE2000 per phase"
            )
        ], axis=2)

    return distances

