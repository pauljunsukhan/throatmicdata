"""
Configuration management for the throatmicdata package.
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, asdict

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(levelname)s - %(message)s"
    file: str = "data/throatmic.log"

@dataclass
class Config:
    """Global configuration."""
    logging: LoggingConfig = LoggingConfig()
    
    def save(self, path: str = "config.yaml"):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f)
    
    @classmethod
    def load(cls, path: str = "config.yaml") -> 'Config':
        """Load configuration from YAML file."""
        if not os.path.exists(path):
            return cls()
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
            config = cls()
            if data and isinstance(data, dict):
                if 'logging' in data:
                    config.logging = LoggingConfig(**data['logging'])
            return config

# Global configuration instance
config = Config() 