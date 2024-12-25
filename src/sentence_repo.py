"""Sentence repository for managing and storing sentences."""

import json
import logging
import random
import requests
from pathlib import Path
from typing import List, Optional, Dict
from tqdm import tqdm
from .sentence_filter import SentenceFilter, ComplexityAnalyzer
from .nlp_manager import NLPManager
from .config import Config, config
from .exceptions import DataError

logger = logging.getLogger(__name__)

class SentenceRepository:
    """Manages a repository of sentences for recording."""
    
    COMMON_VOICE_URLS = [
        "https://raw.githubusercontent.com/common-voice/common-voice/main/server/data/en/wiki.en.txt",  # Wikipedia sentences
        "https://raw.githubusercontent.com/common-voice/common-voice/main/server/data/en/sentence-collector.txt"  # Community sentences
    ]
    
    def __init__(self, base_dir: Path):
        """Initialize repository with base directory."""
        self.base_dir = Path(base_dir)
        self.repository_dir = self.base_dir / "data" / "repository"
        self.repository_dir.mkdir(parents=True, exist_ok=True)
        self.sentences_file = self.repository_dir / "sentences.json"
        self.used_file = self.repository_dir / "used_sentences.json"
        self.trash_file = self.repository_dir / "trashed_sentences.json"
        
        # Initialize empty lists
        self.sentences: List[str] = []
        self.used_sentences: List[str] = []
        self.trashed_sentences: List[str] = []
        
        # Load existing sentences
        self._load_sentences()
        self._load_used_sentences()
        self._load_trashed_sentences()
        
        # Initialize filters and analyzers
        self.sentence_filter = SentenceFilter()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.nlp_manager = NLPManager.get_instance()
        
        # Get config from sentence filter
        self.config = self.sentence_filter.config
    
    def download_sentences(self, count: int = 100) -> int:
        """Download sentences from Common Voice with optimized two-stage validation."""
        logger.info(f"Downloading {count} Common Voice sentences...")
        candidates = []
        
        try:
            BATCH_SIZE = 1000
            for url in self.COMMON_VOICE_URLS:
                try:
                    logger.info(f"Downloading from {url}")
                    response = requests.get(url)
                    response.raise_for_status()
                    
                    # Stage 1: Quick text-based filtering
                    logger.info("Stage 1: Quick filtering...")
                    sentences = [
                        line.strip() for line in response.text.split('\n')
                        if (line.strip() 
                            and line[0].isupper()
                            and line[-1] in '.!?'
                            and self.config.min_chars <= len(line) <= self.config.max_chars)
                    ]
                    
                    if not sentences:
                        logger.warning(f"No valid sentences found in {url}")
                        continue
                        
                    logger.info(f"Found {len(sentences)} potential sentences after basic filtering")
                    
                    # Take a random sample to process
                    sample_size = min(count * 10, len(sentences))
                    sentences = random.sample(sentences, sample_size)
                    logger.info(f"Processing random sample of {sample_size} sentences")
                    
                    # Stage 2: Full validation
                    logger.info("Stage 2: Full validation...")
                    quick_passed = 0
                    full_passed = 0
                    
                    # Process in batches with minimal pipeline
                    nlp = self.nlp_manager.get_nlp()
                    with tqdm(total=len(sentences), desc="Processing sentences") as pbar:
                        docs = nlp.pipe(
                            sentences,
                            batch_size=BATCH_SIZE,
                            disable=['textcat']  # Keep parser for dependency checks
                        )
                        
                        with tqdm(total=count, desc="Validating sentences") as validation_pbar:
                            for doc in docs:
                                pbar.update(1)  # Update processing progress
                                if len(candidates) >= count * 2:
                                    break
                                    
                                # Quick check first
                                if self._is_good_sentence_quick(doc):
                                    quick_passed += 1
                                    # Full validation only if quick check passes
                                    if self._is_good_sentence(doc):
                                        full_passed += 1
                                        candidates.append(doc.text)
                                        validation_pbar.update(1)
                    
                    logger.info(f"Quick validation passed: {quick_passed}")
                    logger.info(f"Full validation passed: {full_passed}")
                    
                    if len(candidates) >= count * 2:
                        break
                        
                except Exception as e:
                    logger.warning(f"Error downloading from {url}: {e}")
                    continue
            
            # Filter out existing sentences
            new_sentences = [
                s for s in candidates
                if s not in self.used_sentences 
                and s not in self.sentences
                and s not in self.trashed_sentences
            ][:count]
            
            # Add to available sentences
            self.sentences.extend(new_sentences)
            self._save_sentences()
            
            added_count = len(new_sentences)
            logger.info(f"Added {added_count} new sentences")
            return added_count
            
        except Exception as e:
            logger.error(f"Error downloading sentences: {e}")
            raise
    
    def add_sentences(self, new_sentences: List[str]) -> int:
        """Add custom sentences to the repository."""
        # Filter out duplicates and already used sentences
        filtered_sentences = [
            s.strip() for s in new_sentences
            if s.strip() 
            and self.sentence_filter.is_good_sentence(s.strip())
            and s not in self.sentences 
            and s not in self.used_sentences
        ]
        
        self.sentences.extend(filtered_sentences)
        self._save_sentences()
        
        return len(filtered_sentences)
    
    def get_next_sentence(self) -> Optional[str]:
        """Get the next sentence for recording."""
        if not self.sentences:
            return None
        
        sentence = self.sentences.pop(0)
        self.used_sentences.append(sentence)
        
        self._save_sentences()
        self._save_used_sentences()
        
        return sentence
    
    def mark_sentence_used(self, sentence: str) -> None:
        """Mark a sentence as used."""
        if sentence in self.sentences:
            self.sentences.remove(sentence)
            self._save_sentences()
        
        if sentence not in self.used_sentences:
            self.used_sentences.append(sentence)
            self._save_used_sentences()
    
    def trash_sentence(self, sentence: str) -> None:
        """Move a sentence to trash."""
        if sentence in self.sentences:
            self.sentences.remove(sentence)
        if sentence in self.used_sentences:
            self.used_sentences.remove(sentence)
            
        self.trashed_sentences.append(sentence)
        
        self._save_sentences()
        self._save_used_sentences()
        self._save_trashed_sentences()
    
    def clear_sentences(self) -> None:
        """Clear all sentences (both available and used)."""
        self.sentences = []
        self.used_sentences = []
        self._save_sentences()
        self._save_used_sentences()
        logger.info("Cleared all sentences")
    
    def get_stats(self) -> Dict:
        """Get repository statistics."""
        total = len(self.sentences) + len(self.used_sentences)
        recorded = len(self.used_sentences)
        trashed = len(self.trashed_sentences)
        
        return {
            "total_sentences": total,
            "recorded_sentences": recorded,
            "remaining_sentences": len(self.sentences),
            "trashed_sentences": trashed,
            "completion_percentage": (recorded / total * 100) if total > 0 else 0,
            "total_audio_time": recorded * 10 / 60  # Assuming 10 seconds per recording
        }
    
    def _load_sentences(self) -> None:
        """Load sentences from file."""
        if self.sentences_file.exists():
            try:
                with open(self.sentences_file) as f:
                    data = json.load(f)
                    self.sentences = data if isinstance(data, list) else []
            except Exception as e:
                logger.error(f"Error loading sentences: {e}")
                raise DataError(f"Failed to load sentences: {e}")
    
    def _load_used_sentences(self) -> None:
        """Load used sentences from file."""
        if self.used_file.exists():
            try:
                with open(self.used_file) as f:
                    data = json.load(f)
                    self.used_sentences = data if isinstance(data, list) else []
            except Exception as e:
                logger.error(f"Error loading used sentences: {e}")
                raise DataError(f"Failed to load used sentences: {e}")
    
    def _load_trashed_sentences(self) -> None:
        """Load trashed sentences from file."""
        if self.trash_file.exists():
            try:
                with open(self.trash_file) as f:
                    data = json.load(f)
                    self.trashed_sentences = data if isinstance(data, list) else []
            except Exception as e:
                logger.error(f"Error loading trashed sentences: {e}")
                raise DataError(f"Failed to load trashed sentences: {e}")
    
    def _save_sentences(self) -> None:
        """Save sentences to file."""
        with open(self.sentences_file, 'w') as f:
            json.dump(self.sentences, f, indent=2)
    
    def _save_used_sentences(self) -> None:
        """Save used sentences to file."""
        with open(self.used_file, 'w') as f:
            json.dump(self.used_sentences, f, indent=2)
    
    def _save_trashed_sentences(self) -> None:
        """Save trashed sentences to file."""
        with open(self.trash_file, 'w') as f:
            json.dump(self.trashed_sentences, f, indent=2)
    
    def _is_good_sentence_quick(self, doc) -> bool:
        """Quick pre-filter for sentences."""
        try:
            # Word count check
            word_count = len([token for token in doc if not token.is_punct])
            if not (self.config.min_words <= word_count <= self.config.max_words):
                logger.debug(f"Failed word count: {word_count} words")
                return False
            
            # Entity check
            entity_counts = {'PERSON': 0, 'GPE': 0, 'ORG': 0}
            for ent in doc.ents:
                if ent.label_ in entity_counts:
                    entity_counts[ent.label_] += 1
                    if entity_counts[ent.label_] > getattr(self.config, f'max_{ent.label_.lower()}s'):
                        logger.debug(f"Failed entity check: {ent.label_} = {entity_counts[ent.label_]}")
                        return False
            
            # Basic complexity hint
            has_conjunction = any(token.dep_ == 'cc' for token in doc)
            has_subordinate = any(token.dep_ == 'mark' for token in doc)
            
            if not (has_conjunction or has_subordinate):
                logger.debug(f"Failed complexity check for: {doc.text}")
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Error in quick sentence check: {e}")
            return False
    
    def _is_good_sentence(self, doc) -> bool:
        """Check if a sentence is suitable using enhanced complexity analysis."""
        try:
            # Analyze complexity using the already processed doc
            scores = self.complexity_analyzer.analyze_sentence(doc.text)
            if not scores:
                logger.debug(f"Failed complexity analysis for: {doc.text}")
                return False
                
            # Check complexity thresholds
            if scores['total'] < self.config.min_total_complexity:
                logger.debug(f"Failed complexity threshold: {scores['total']} < {self.config.min_total_complexity}")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error in _is_good_sentence: {e}")
            return False