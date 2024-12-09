"""
Custom exceptions for the throatmicdata package.
"""

class ThroatMicError(Exception):
    """Base exception for all throatmicdata errors."""
    pass

class RecordingError(ThroatMicError):
    """Raised when there is an error during recording."""
    pass

class DataError(ThroatMicError):
    """Raised when there is an error with data handling."""
    pass

class ValidationError(ThroatMicError):
    """Raised when there is a validation error."""
    pass

class DatasetError(ThroatMicError):
    """Raised when there is an error with dataset operations."""
    pass

class DownloadError(DatasetError):
    """Raised when there is an error downloading the dataset."""
    pass

class UploadError(DatasetError):
    """Raised when there is an error uploading the dataset."""
    pass 