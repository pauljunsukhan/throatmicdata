"""
Configuration management for the throatmicdata package.
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, asdict

@dataclass
class AudioConfig:
    """Audio recording configuration."""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 4096
    format: str = 'wav'
    duration: float = 10.0  # seconds
    clipping_threshold: float = 0.95
    min_level_threshold: float = 0.1

@dataclass
class DatasetConfig:
    """Dataset configuration."""
    repo_id: str = "pauljunsukhan/throatmic_codered"
    metadata_file: str = "data/metadata/metadata.csv"
    recordings_dir: str = "data/recordings"
    dataset_card: str = "DATASET_CARD.md"
    min_words: int = 12
    max_words: int = 25
    min_duration: float = 8.0  # seconds
    max_duration: float = 12.0  # seconds

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(levelname)s - %(message)s"
    file: str = "data/throatmic.log"

@dataclass
class Config:
    """Global configuration."""
    audio: AudioConfig = AudioConfig()
    dataset: DatasetConfig = DatasetConfig()
    logging: LoggingConfig = LoggingConfig()
    
    def save(self, path: str = "config.yaml"):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f)
    
    @classmethod
    def load(cls, path: str = "config.yaml") -> 'Config':
        """Load configuration from YAML file.
        
        If no config file exists, returns default configuration.
        If environment variables are set, they override file settings.
        """
        config = cls()
        
        # Load from file if it exists
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
                if data and isinstance(data, dict):
                    if 'audio' in data:
                        config.audio = AudioConfig(**data['audio'])
                    if 'dataset' in data:
                        config.dataset = DatasetConfig(**data['dataset'])
                    if 'logging' in data:
                        config.logging = LoggingConfig(**data['logging'])
        
        # Environment variables override file settings
        if os.getenv('THROATMIC_REPO_ID'):
            config.dataset.repo_id = os.getenv('THROATMIC_REPO_ID')
        if os.getenv('THROATMIC_LOG_LEVEL'):
            config.logging.level = os.getenv('THROATMIC_LOG_LEVEL')
        if os.getenv('THROATMIC_SAMPLE_RATE'):
            config.audio.sample_rate = int(os.getenv('THROATMIC_SAMPLE_RATE'))
        
        return config
    
    def ensure_directories(self):
        """Create necessary directories."""
        dirs = [
            Path("data"),
            Path(self.dataset.recordings_dir),
            Path(self.dataset.metadata_file).parent,
            Path("data/cache"),
            Path("data/repository")
        ]
        
        for dir in dirs:
            dir.mkdir(parents=True, exist_ok=True)

# Global configuration instance
config = Config()
config.ensure_directories() 