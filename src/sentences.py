"""
Sentence repository for throat mic recordings.
Uses Common Voice sentences for prompts.
"""

import random
from pathlib import Path
import json
import requests
from typing import List, Optional, Dict
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class SentenceRepository:
    """Manages a repository of sentences for recording."""
    
    COMMON_VOICE_URLS = [
        "https://raw.githubusercontent.com/common-voice/common-voice/main/server/data/en/wiki.en.txt",  # Wikipedia sentences
        "https://raw.githubusercontent.com/common-voice/common-voice/main/server/data/en/sentence-collector.txt"  # Community sentences
    ]
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.repository_dir = self.base_dir / "data" / "repository"
        self.repository_dir.mkdir(parents=True, exist_ok=True)
        self.sentences_file = self.repository_dir / "sentences.json"
        self.used_file = self.repository_dir / "used_sentences.json"
        
        # Initialize empty lists
        self.sentences: List[str] = []
        self.used_sentences: List[str] = []
        
        # Load existing sentences
        self._load_sentences()
        self._load_used_sentences()
    
    def _load_sentences(self) -> None:
        """Load sentences from file."""
        if self.sentences_file.exists():
            try:
                with open(self.sentences_file) as f:
                    data = json.load(f)
                    self.sentences = data if isinstance(data, list) else []
            except Exception as e:
                logger.error(f"Error loading sentences: {e}")
    
    def _load_used_sentences(self) -> None:
        """Load used sentences from file."""
        if self.used_file.exists():
            try:
                with open(self.used_file) as f:
                    data = json.load(f)
                    self.used_sentences = data if isinstance(data, list) else []
            except Exception as e:
                logger.error(f"Error loading used sentences: {e}")
    
    def _save_sentences(self) -> None:
        """Save sentences to file."""
        with open(self.sentences_file, 'w') as f:
            json.dump(self.sentences, f, indent=2)
    
    def _save_used_sentences(self) -> None:
        """Save used sentences to file."""
        with open(self.used_file, 'w') as f:
            json.dump(self.used_sentences, f, indent=2)
    
    def _is_good_sentence(self, sentence: str) -> bool:
        """Check if a sentence is suitable for Whisper fine-tuning.
        
        Criteria:
        1. Length: 12-25 words (targeting ~10 seconds of speech)
        2. Complexity: Must have subordinate clauses or complex structure
        3. Natural language: Proper punctuation and structure
        4. No problematic content: No numbers, special chars, etc.
        """
        words = sentence.split()
        
        # Length criteria (12-25 words for ~10 seconds of natural speech)
        if not (12 <= len(words) <= 25):
            return False
            
        # Must start with capital and end with proper punctuation
        if not (sentence[0].isupper() and sentence[-1] in '.!?'):
            return False
            
        # No numbers or special characters
        if any(c.isdigit() or c in '[]{}()@#$%^&*' for c in sentence):
            return False
            
        # No all-caps words except for common acronyms
        allowed_caps = {'US', 'UK', 'EU', 'UN', 'NASA', 'FBI', 'CIA', 'WHO'}
        if any(w.isupper() and w not in allowed_caps for w in words[1:]):
            return False
            
        # Complexity requirements (at least two of these):
        complexity_score = 0
        
        # Has proper comma usage
        if ',' in sentence:
            complexity_score += 1
            
        # Has subordinating conjunctions
        subordinating_conj = {'because', 'although', 'though', 'unless', 'while', 'whereas',
                            'if', 'since', 'before', 'after', 'as', 'when', 'whenever', 'where',
                            'wherever', 'whether', 'which', 'who', 'whoever', 'whom', 'whose'}
        if any(conj in sentence.lower() for conj in subordinating_conj):
            complexity_score += 1
            
        # Has coordinating conjunctions
        coordinating_conj = {'and', 'but', 'or', 'nor', 'for', 'yet', 'so'}
        if any(conj in words for conj in coordinating_conj):
            complexity_score += 1
            
        # Has relative pronouns or complex phrases
        complex_markers = {'that', 'which', 'who', 'whom', 'whose', 'where', 'when'}
        if any(marker in words for marker in complex_markers):
            complexity_score += 1
            
        # Requires at least two complexity markers
        if complexity_score < 2:
            return False
            
        # Check for natural sentence structure
        # Avoid sentences with too many commas or conjunctions
        comma_count = sentence.count(',')
        if comma_count > 4:  # Too many clauses might be unnatural
            return False
            
        return True
    
    def download_sentences(self, count: int = 100) -> int:
        """Download sentences from Common Voice.
        
        Args:
            count: Number of sentences to download
            
        Returns:
            Number of new sentences added
        """
        logger.info(f"Downloading {count} Common Voice sentences...")
        all_sentences = []
        
        try:
            for url in self.COMMON_VOICE_URLS:
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                    
                    # Split into sentences and clean
                    sentences = [
                        line.strip() for line in response.text.split('\n')
                        if line.strip() and self._is_good_sentence(line.strip())
                    ]
                    all_sentences.extend(sentences)
                    
                except Exception as e:
                    logger.warning(f"Error downloading from {url}: {e}")
                    continue
            
            # Filter out any sentences we've already used or have
            new_sentences = [
                s for s in all_sentences 
                if s not in self.used_sentences 
                and s not in self.sentences
            ]
            
            # Shuffle and take requested number
            random.shuffle(new_sentences)
            selected_sentences = new_sentences[:count]
            
            # Add to available sentences
            self.sentences.extend(selected_sentences)
            self._save_sentences()
            
            added_count = len(selected_sentences)
            logger.info(f"Added {added_count} new sentences")
            return added_count
            
        except Exception as e:
            logger.error(f"Error downloading sentences: {e}")
            raise
    
    def add_sentences(self, new_sentences: List[str]) -> int:
        """Add custom sentences to the repository.
        
        Args:
            new_sentences: List of sentences to add
            
        Returns:
            Number of new sentences added
        """
        # Filter out duplicates and already used sentences
        filtered_sentences = [
            s.strip() for s in new_sentences
            if s.strip() 
            and self._is_good_sentence(s.strip())
            and s not in self.sentences 
            and s not in self.used_sentences
        ]
        
        self.sentences.extend(filtered_sentences)
        self._save_sentences()
        
        return len(filtered_sentences)
    
    def clear_sentences(self) -> None:
        """Clear all sentences (both available and used)."""
        self.sentences = []
        self.used_sentences = []
        self._save_sentences()
        self._save_used_sentences()
        logger.info("Cleared all sentences")
    
    def get_next_sentence(self) -> Optional[str]:
        """Get the next sentence for recording."""
        if not self.sentences:
            return None
        
        sentence = self.sentences.pop(0)
        self.used_sentences.append(sentence)
        
        # Save state
        self._save_sentences()
        self._save_used_sentences()
        
        return sentence
    
    def mark_sentence_used(self, sentence: str):
        """Mark a sentence as used."""
        if sentence in self.sentences:
            self.sentences.remove(sentence)
            self._save_sentences()
        
        if sentence not in self.used_sentences:
            self.used_sentences.append(sentence)
            self._save_used_sentences()
    
    def get_remaining_count(self) -> int:
        """Get count of remaining sentences."""
        return len(self.sentences)
    
    def get_used_count(self) -> int:
        """Get count of used sentences."""
        return len(self.used_sentences)
    
    def get_stats(self) -> Dict:
        """Get repository statistics."""
        total = len(self.sentences) + len(self.used_sentences)
        recorded = len(self.used_sentences)
        
        return {
            "total_sentences": total,
            "recorded_sentences": recorded,
            "remaining_sentences": len(self.sentences),
            "completion_percentage": (recorded / total * 100) if total > 0 else 0,
            "total_audio_time": recorded * 10 / 60  # Assuming 10 seconds per recording
        } 