import yaml
from pathlib import Path
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        raise

def validate_transformer_configs(configs: List[Dict[str, Any]]) -> None:
    """
    Validate transformer configurations.
    
    Args:
        configs: List of transformer configurations
        
    Raises:
        ValueError: If any configuration is invalid
    """
    required_fields = {'name', 'enabled'}
    
    for config in configs:
        # Check required fields
        missing_fields = required_fields - set(config.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields in transformer config: {missing_fields}")
        
        # Validate field types
        if not isinstance(config['name'], str):
            raise ValueError(f"Transformer name must be a string, got {type(config['name'])}")
        if not isinstance(config['enabled'], bool):
            raise ValueError(f"Transformer enabled flag must be a boolean, got {type(config['enabled'])}")
        
        # Log transformer status
        status = "enabled" if config['enabled'] else "disabled"
        logger.info(f"Transformer '{config['name']}' is {status}") 