import argparse
import logging
import os
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def add_argparse_argument(parser: argparse.ArgumentParser) -> None:
    """
    Add -o/--output-dir-path argument to the given ArgumentParser.

    :param parser: The argparse.ArgumentParser instance.
    """
    parser.add_argument(
        "-o",
        "--output-dir-path",
        type=str,
        help="Path to output directory. If directory does not exist, it will be created (but parent directory must exist)."
    )


def prepare_output_path(output_dir_path: str, sample_name: Optional[str] = None) -> Tuple[str, str]:
    """
    Create directory for output (if it does not exist) and return target dir and prefix.

    Parent directory must exist. When sample_name is specified, a subdirectory
    with this name is created and sample_name is used as the file prefix.

    :param output_dir_path: The path to the base directory where output files will be stored.
    :param sample_name: Optional name of the sample to use for subfolder and prefix.
    :return: Tuple of (output_dir_path, output_file_prefix).
    :raises FileNotFoundError: If neither output directory nor its parent directory exists.
    """
    if not os.path.exists(output_dir_path):
        parent_dir = os.path.dirname(output_dir_path)
        if not parent_dir or os.path.exists(parent_dir):
            os.makedirs(output_dir_path, exist_ok=True)
        else:
            logger.error("Output directory, nor its parent directory exist.")
            raise FileNotFoundError("Output directory, nor its parent directory exist.")
    else:
        logger.debug("Output directory prepared.")

    if sample_name is not None:
        target_dir_path = os.path.join(output_dir_path, sample_name)
        os.makedirs(target_dir_path, exist_ok=True)
        output_file_prefix = f"{sample_name}-"
    else:
        target_dir_path = output_dir_path
        output_file_prefix = ""

    return target_dir_path, output_file_prefix
