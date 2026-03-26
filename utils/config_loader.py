"""
Simple configuration loader for GEPA system.
Loads YAML configuration files with environment variable substitution.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Simple configuration loader for YAML files."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the config loader.
        
        Args:
            config_path: Path to the configuration file. If None, looks for config.yaml in the project root.
        """
        if config_path is None:
            # Look for config file in the project root
            project_root = Path(__file__).parent.parent
            config_path = project_root / 'config.yaml'
            
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
    
    def _substitute_env_vars(self, data: Any) -> Any:
        """
        Recursively substitute environment variables in configuration values.
        Environment variables should be specified as ${VAR_NAME} or $VAR_NAME.
        """
        if isinstance(data, dict):
            return {key: self._substitute_env_vars(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._substitute_env_vars(item) for item in data]
        elif isinstance(data, str):
            # Simple environment variable substitution
            import re
            def replace_env_var(match):
                var_name = match.group(1)
                return os.getenv(var_name, match.group(0))  # Return original if env var not found
            
            # Handle ${VAR_NAME} and $VAR_NAME patterns
            data = re.sub(r'\$\{([^}]+)\}', replace_env_var, data)
            data = re.sub(r'\$([A-Za-z_][A-Za-z0-9_]*)', replace_env_var, data)
            return data
        else:
            return data
    
    def load(self) -> Dict[str, Any]:
        """
        Load the configuration from YAML file.
        
        Returns:
            Dict[str, Any]: Configuration dictionary.
        
        Raises:
            RuntimeError: If configuration loading fails.
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if config is None:
                raise ValueError("Configuration file is empty or invalid")
            
            # Substitute environment variables
            config = self._substitute_env_vars(config)
            
            logger.info(f"Successfully loaded configuration from {self.config_path}")
            return config
            
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration from {self.config_path}: {e}") from e


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to the configuration file. If None, auto-detects.
    
    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    loader = ConfigLoader(config_path)
    return loader.load()


# Global configuration instance (lazy-loaded)
_global_config: Optional[Dict[str, Any]] = None


def get_config(config_path: Optional[Union[str, Path]] = None, reload: bool = False) -> Dict[str, Any]:
    """
    Get the global configuration instance.
    
    Args:
        config_path: Path to the configuration file. Only used on first load or reload.
        reload: Whether to force reload the configuration.
    
    Returns:
        Dict[str, Any]: Global configuration dictionary.
    """
    global _global_config
    
    if _global_config is None or reload:
        _global_config = load_config(config_path)
    
    return _global_config
