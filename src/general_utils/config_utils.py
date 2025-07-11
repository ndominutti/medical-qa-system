import os

import yaml


def get_project_root():
    """Get the project root directory"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(config_relative_path="config.yml") -> dict:
    """Load the configuration file.

    Args:
        config_relative_path (str, optional): config file starting from project's root. Defaults to "config.yml".

    Returns:
        dict: configuration file
    """
    project_root = get_project_root()
    config_path = os.path.join(project_root, config_relative_path)

    with open(config_path, "r") as f:
        return yaml.safe_load(f)
