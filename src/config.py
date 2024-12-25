"""
Configuration management for the throatmicdata package.
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Union, Tuple

@dataclass
class AudioConfig:
    """Audio recording configuration."""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 4096
    format: str = 'wav'
    duration: float = 10.0
    clipping_threshold: float = 0.95
    min_level_threshold: float = 0.1
    buffer_ratio: float = 1.2
    min_duration: float = 6.0
    max_duration: float = 30.0

@dataclass
class DatasetConfig:
    """Dataset configuration."""
    repo_id: str = "pauljunsukhan/throatmic_codered"
    metadata_file: str = "data/metadata/metadata.csv"
    recordings_dir: str = "data/recordings"
    dataset_card: str = "DATASET_CARD.md"

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(levelname)s - %(message)s"
    file: str = "data/throatmic.log"

@dataclass
class ComplexityRanges:
    """Complexity range parameters."""
    clause_count: List[int]
    word_length: List[int]
    tree_depth: List[int]

@dataclass
class TimingConfig:
    """Timing parameters for sentence duration estimation."""
    syllable_duration: float
    word_boundary_pause: float
    long_word_penalty: float
    comprehension_buffer: float
    min_duration: float = 6.0      # Audio duration limits
    max_duration: float = 30.0     # Audio duration limits

    def __post_init__(self):
        if self.syllable_duration <= 0:
            raise ValueError("syllable_duration must be positive")
        if self.comprehension_buffer <= 0:
            raise ValueError("comprehension_buffer must be positive")
        if self.min_duration > self.max_duration:
            raise ValueError("min_duration cannot be greater than max_duration")

@dataclass
class SentenceFilterConfig:
    """Sentence filtering configuration."""
    min_chars: int
    max_chars: int
    min_words: int        # Get from config.yaml
    max_words: int        # Get from config.yaml
    max_entities: int
    max_persons: int
    max_locations: int
    max_organizations: int
    min_complexity_score: int
    min_total_complexity: int
    pos_ratios: Dict[str, List[float]]
    complexity_ranges: ComplexityRanges
    timing: TimingConfig

    def __post_init__(self):
        # Validate all numeric ranges
        if self.min_chars > self.max_chars:
            raise ValueError("min_chars cannot be greater than max_chars")
        if self.min_words > self.max_words:
            raise ValueError("min_words cannot be greater than max_words")
        if self.max_entities < 0:
            raise ValueError("max_entities must be non-negative")
        if self.max_persons < 0:
            raise ValueError("max_persons must be non-negative")
        if self.max_locations < 0:
            raise ValueError("max_locations must be non-negative")
        if self.max_organizations < 0:
            raise ValueError("max_organizations must be non-negative")
        if self.min_complexity_score < 0:
            raise ValueError("min_complexity_score must be non-negative")
        if self.min_total_complexity < 0:
            raise ValueError("min_total_complexity must be non-negative")
            
        # Ensure POS ratios are lists
        for pos, value in self.pos_ratios.items():
            if isinstance(value, (int, float)):
                self.pos_ratios[pos] = [0, float(value)]
            elif len(value) != 2:
                raise ValueError(f"POS ratio range for {pos} must be [min, max]")
        
        # Validate complexity ranges
        for field in ['clause_count', 'word_length', 'tree_depth']:
            range_values = getattr(self.complexity_ranges, field)
            if len(range_values) != 2 or range_values[0] > range_values[1]:
                raise ValueError(f"Invalid {field} range: {range_values}")
        
        # Validate timing config if present
        if hasattr(self, 'timing'):
            if self.timing.syllable_duration <= 0:
                raise ValueError("syllable_duration must be positive")
            if self.timing.comprehension_buffer <= 0:
                raise ValueError("comprehension_buffer must be positive")

@dataclass
class Config:
    """Global configuration."""
    sentence_filter: SentenceFilterConfig
    audio: AudioConfig = AudioConfig()
    dataset: DatasetConfig = DatasetConfig()
    logging: LoggingConfig = LoggingConfig()
    
    def __post_init__(self):
        # Convert single values to ranges if needed
        if self.sentence_filter.pos_ratios:
            for pos, value in self.sentence_filter.pos_ratios.items():
                if isinstance(value, (int, float)):
                    # If single value provided, use [0, value] as range
                    self.sentence_filter.pos_ratios[pos] = [0, float(value)]
    
    def save(self, path: str = "config.yaml"):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f)
    
    @classmethod
    def load(cls) -> 'Config':
        """Load configuration from YAML file."""
        config_path = Path(__file__).parent.parent / "config.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
            
        # Convert complexity_ranges dict to ComplexityRanges object
        if 'sentence_filter' in config_dict:
            sf_config = config_dict['sentence_filter']
            if 'complexity_ranges' in sf_config:
                sf_config['complexity_ranges'] = ComplexityRanges(**sf_config['complexity_ranges'])
            # Add timing config conversion
            if 'timing' in sf_config:
                sf_config['timing'] = TimingConfig(**sf_config['timing'])
            config_dict['sentence_filter'] = SentenceFilterConfig(**sf_config)
            
        # Convert dictionaries to dataclass objects
        if 'audio' in config_dict:
            config_dict['audio'] = AudioConfig(**config_dict['audio'])
        if 'dataset' in config_dict:
            config_dict['dataset'] = DatasetConfig(**config_dict['dataset'])
        if 'logging' in config_dict:
            config_dict['logging'] = LoggingConfig(**config_dict['logging'])
            
        return cls(**config_dict)
    
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