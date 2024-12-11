from omegaconf import OmegaConf
import os

def load_config(config_path: str = "config.yaml"):
    """
    Load configuration from a YAML file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    return OmegaConf.load(config_path)