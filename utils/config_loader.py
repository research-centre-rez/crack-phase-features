import argparse
import json
from typing import Any, Dict, Optional
import yaml


def add_argparse_argument_cracks_config_path(parser: argparse.ArgumentParser) -> None:
    """
    Add --cracks-config-path argument to the given ArgumentParser.

    :param parser: The argparse.ArgumentParser instance.
    """
    parser.add_argument(
        "-c",
        "--cracks-config-path",
        type=str,
        help="Path to the cracks_config.json. In this file are defined crack image processing parameters "
             "(@see config/cracks_config.yaml).",
        default="config/cracks_config.yaml"
    )


def add_argparse_argument_phase_config_path(parser: argparse.ArgumentParser) -> None:
    """
    Add --phase-config-path argument to the given ArgumentParser.

    :param parser: The argparse.ArgumentParser instance.
    """
    parser.add_argument(
        "-c",
        "--phase-config-path",
        type=str,
        help="Path to the phase_config.json. In this file are defined colors of phases used in phase map "
             "(@see config/phases_config.yaml).",
        default="config/phases_config.yaml"
    )


class Config:
    """
    Centralized configuration cache for cracks and phase map configurations.
    """
    cracks_config: Optional[Dict[str, Any]] = None
    phases_config: Optional[Dict[str, Any]] = None

    @staticmethod
    def load_config(config_file_path: str, type: str) -> Dict[str, Any]:
        """
        Load configuration from a JSON or YAML file into memory.

        :param config_file_path: Path to the configuration file (.json or .yaml).
        :param type: Type of configuration ('cracks' or 'phases').
        :return: Loaded configuration dictionary.
        :raises ValueError: If the config type or file format is unsupported.
        """
        with open(config_file_path, 'r') as file:
            if type == 'cracks':
                if config_file_path.endswith('.json'):
                    Config.cracks_config = json.load(file)
                elif config_file_path.endswith('.yaml'):
                    Config.cracks_config = yaml.safe_load(file)
                else:
                    raise ValueError(f"Unknown config file type: {config_file_path}")
                return Config.cracks_config
            elif type == 'phases':
                if config_file_path.endswith('.json'):
                    Config.phases_config = json.load(file)
                elif config_file_path.endswith('.yaml'):
                    Config.phases_config = yaml.safe_load(file)
                else:
                    raise ValueError(f"Unknown config file type: {config_file_path}")
                return Config.phases_config
            else:
                raise ValueError(f"Unknown config type: {type}")

    @staticmethod
    def get_config(type: str) -> Dict[str, Any]:
        """
        Retrieve previously loaded configuration by type.

        :param type: Type of configuration ('cracks' or 'phases').
        :return: Cached configuration dictionary.
        :raises ValueError: If the config type is unknown or not yet loaded.
        """
        if type == 'cracks':
            config = Config.cracks_config
        elif type == 'phases':
            config = Config.phases_config
        else:
            raise ValueError(f"Unknown config type: {type}")

        if config is None:
            raise ValueError("Config not loaded, call `load_config` first.")
        else:
            return config
