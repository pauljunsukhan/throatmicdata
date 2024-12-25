"""
Throat microphone dataset collection and management tools.
"""

from .config import Config, config
from .exceptions import (
    ThroatMicError, RecordingError, DataError, ValidationError,
    DatasetError, DownloadError, UploadError
)
from .nlp_manager import NLPManager
from .sentence_filter import SentenceFilter, ComplexityAnalyzer
from .sentence_repo import SentenceRepository
from .analyzer import AudioQualityControl, DatasetAnalytics
from .recorder import ThroatMicRecorder


__version__ = "0.1.0" 