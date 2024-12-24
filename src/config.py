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
class SentenceFilterConfig:
    min_chars: int = 60
    max_chars: int = 200
    min_words: int = 5
    max_words: int = 30
    max_entities: int = 3
    max_persons: int = 2
    max_locations: int = 1
    max_organizations: int = 2
    min_complexity_score: int = 2
    min_total_complexity: int = 3
    pos_ratios: dict = None

    def __post_init__(self):
        if self.pos_ratios is None:
            self.pos_ratios = {
                'VERB': [0.1, 0.35],
                'NOUN': [0.2, 0.45],
                'ADJ': [0.05, 0.25]
            }

@dataclass
class Config:
    """Global configuration."""
    audio: AudioConfig = AudioConfig()
    dataset: DatasetConfig = DatasetConfig()
    logging: LoggingConfig = LoggingConfig()
    sentence_filter: SentenceFilterConfig = SentenceFilterConfig()
    
    def save(self, path: str = "config.yaml"):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f)
    
    @staticmethod
    def load():
        """Load configuration from YAML file."""
        config_path = Path(__file__).parent.parent / "config.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
            
        # Convert dictionaries to dataclass objects
        if 'audio' in config_dict:
            config_dict['audio'] = AudioConfig(**config_dict['audio'])
        if 'dataset' in config_dict:
            config_dict['dataset'] = DatasetConfig(**config_dict['dataset'])
        if 'logging' in config_dict:
            config_dict['logging'] = LoggingConfig(**config_dict['logging'])
        if 'sentence_filter' in config_dict:
            config_dict['sentence_filter'] = SentenceFilterConfig(**config_dict['sentence_filter'])
            
        config = Config()
        for key, value in config_dict.items():
            setattr(config, key, value)
            
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

# Initialize global configuration
config = Config.load()
config.ensure_directories() 