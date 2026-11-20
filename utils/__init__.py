import logging
import utils.output_dir_generator
import utils.image_logger
import utils.cli_arguments
import utils.config_loader


def configure_logger(filename: str, level: int = logging.INFO) -> None:
    """
    Configure the root logger with standard timestamp and format.

    :param filename: Path to the log file.
    :param level: Logging severity level (defaults to logging.INFO).
    """
    logging.basicConfig(
        level=level,
        filename=filename,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )