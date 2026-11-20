import logging
import os
from typing import Optional
import imageio.v3 as iio
import numpy as np

logger = logging.getLogger(__name__)


def dump_image(file_path: Optional[str], img: np.ndarray) -> None:
    """
    Save an image array to the specified file path, normalizing to uint8 if needed.

    :param file_path: Destination file path for saving the image.
    :param img: Numpy array representing the image.
    """
    if file_path is None:
        logger.warning("No path provided. Ignoring image dump.")
        return

    # Normalize image to uint8 if not already in uint8 format
    if img.dtype != np.uint8:
        folder_name = file_path.split(os.path.sep)[-2] if os.path.sep in file_path else ""
        logger.warning(
            f"{folder_name}: Output image does not have requested format (uint8). "
            f"Converting to uint8, range stretched."
        )
        min_val = np.min(img)
        max_val = np.max(img)
        if max_val == min_val:
            img = np.zeros(img.shape, dtype=np.uint8)
        else:
            # Min-max scaling into [0, 255] uint8 range
            img = (((img - min_val) / (max_val - min_val)) * 255).astype(np.uint8)

    iio.imwrite(file_path, img)


def debug(img: np.ndarray, folder_path: Optional[str], filename: str) -> None:
    """
    Dump an image if DEBUG log level is enabled.

    :param img: Numpy array representing the image.
    :param folder_path: Target directory path.
    :param filename: Target file name.
    """
    if folder_path is None:
        logger.warning(f"{filename}: No folder path provided. Ignoring debug.")
        return
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Dumping image to {filename}")
        dump_image(os.path.join(folder_path, filename), img)


def info(img: np.ndarray, folder_path: Optional[str], filename: str) -> None:
    """
    Dump an image if INFO log level is enabled.

    :param img: Numpy array representing the image.
    :param folder_path: Target directory path.
    :param filename: Target file name.
    """
    if folder_path is None:
        logger.warning(f"{filename}: No folder path provided. Ignoring info.")
        return
    if logger.isEnabledFor(logging.INFO):
        logger.info(f"Dumping image to {filename}")
        dump_image(os.path.join(folder_path, filename), img)


def warning(img: np.ndarray, folder_path: Optional[str], filename: str) -> None:
    """
    Dump an image if WARNING log level is enabled.

    :param img: Numpy array representing the image.
    :param folder_path: Target directory path.
    :param filename: Target file name.
    """
    if folder_path is None:
        logger.warning(f"{filename}: No folder path provided. Ignoring warning.")
        return
    if logger.isEnabledFor(logging.WARNING):
        logger.warning(f"Dumping image to {filename}")
        dump_image(os.path.join(folder_path, filename), img)


def error(img: np.ndarray, folder_path: Optional[str], filename: str) -> None:
    """
    Dump an image if ERROR log level is enabled.

    :param img: Numpy array representing the image.
    :param folder_path: Target directory path.
    :param filename: Target file name.
    """
    if folder_path is None:
        logger.warning(f"{filename}: No folder path provided. Ignoring error.")
        return
    if logger.isEnabledFor(logging.ERROR):
        logger.error(f"Dumping image to {filename}")
        dump_image(os.path.join(folder_path, filename), img)