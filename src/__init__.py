"""
Throat microphone dataset collection and management tools.
"""

from .recorder import ThroatMicRecorder
from .analyzer import AudioQualityControl, DatasetAnalytics
from .exceptions import (
    ThroatMicError, RecordingError, DataError, ValidationError,
    DatasetError, DownloadError, UploadError
)
from . import config

__version__ = "0.1.0" 