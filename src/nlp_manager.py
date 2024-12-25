import spacy
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class NLPManager:
    _instance = None
    _nlp = None
    
    @classmethod
    def get_instance(cls) -> 'NLPManager':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def get_nlp(self) -> Optional[spacy.language.Language]:
        """Get or load the spaCy model."""
        if self._nlp is None:
            try:
                logger.info("Loading spaCy model...")
                self._nlp = spacy.load("en_core_web_md")
                logger.info("spaCy model loaded successfully")
            except OSError as e:
                logger.error(f"Error loading spaCy model: {e}")
                logger.error("Please run: python -m spacy download en_core_web_md")
                raise
        return self._nlp 