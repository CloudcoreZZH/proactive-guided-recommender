import os
import yaml
import logging


def load_config(config_name: str) -> dict:
    """Load a YAML config file from the config/ directory."""
    config_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'config')
    config_path = os.path.join(config_dir, config_name)
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_project_root() -> str:
    """Return the absolute path to the project root (pomelo/)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def get_data_path(*parts) -> str:
    """Build a path relative to the project data/ directory."""
    return os.path.join(get_project_root(), 'data', *parts)


def get_results_path(*parts) -> str:
    """Build a path relative to the project results/ directory."""
    return os.path.join(get_project_root(), 'results', *parts)
