import argparse


def add_sample_name(parser: argparse.ArgumentParser) -> None:
    """
    Add --sample-name command-line argument to an ArgumentParser.

    :param parser: The argparse.ArgumentParser instance to modify.
    """
    parser.add_argument(
        "--sample-name",
        type=str,
        help="The name of the sample will be used as prefix for all produced outputs."
    )