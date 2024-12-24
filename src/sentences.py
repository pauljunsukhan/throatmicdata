"""
Sentence repository for throat mic recordings.
Uses Common Voice sentences for prompts.
"""

import random
from pathlib import Path
import json
import requests
from typing import List, Optional, Dict, Tuple
import logging
from tqdm import tqdm
import spacy
from functools import lru_cache
from config import Config

logger = logging.getLogger(__name__)

class ComplexityAnalyzer:
    def __init__(self):
        """Initialize models and thresholds."""
        try:
            self.nlp = spacy.load("en_core_web_md")
        except OSError as e:
            logger.error(f"Error loading spaCy model: {e}")
            logger.error("Please run: python -m spacy download en_core_web_md")
            raise
            
        # Abstract concept prototypes for similarity checking
        self.abstract_concepts = list(self.nlp.pipe([
            "theory", "concept", "philosophy", "analysis",
            "knowledge", "understanding", "perspective"
        ]))
    
    def analyze_sentence(self, text: str) -> Dict:
        """Complete sentence analysis."""
        try:
            doc = self.nlp(text)
            
            scores = {
                'syntactic': self._analyze_syntactic(doc),
                'vocabulary': self._analyze_vocabulary(doc),
                'logical': self._analyze_logical_structure(doc)
            }
            
            scores['total'] = self.calculate_total_score(scores)
            return scores
        except Exception as e:
            logger.warning(f"Error analyzing sentence: {e}")
            return None
    
    def _analyze_syntactic(self, doc) -> Dict:
        """Analyze syntactic complexity."""
        features = {
            'clause_count': 0,
            'tree_depth': 0,
            'sentence_length': len(doc),
            'syntactic_features': []
        }
        
        # Count clauses and get tree depth
        clause_heads = [token for token in doc if token.dep_ == "ROOT" 
                       or token.dep_ == "advcl" or token.dep_ == "relcl"]
        features['clause_count'] = len(clause_heads)
        
        # Calculate tree depth
        def get_depth(token):
            return 1 + max((get_depth(child) for child in token.children), default=0)
        
        features['tree_depth'] = max(get_depth(token) for token in doc 
                                   if token.dep_ == "ROOT")
        
        # Identify syntactic features
        if any(token.dep_ == "mark" for token in doc):
            features['syntactic_features'].append("subordinate_clause")
        if any(token.dep_ == "relcl" for token in doc):
            features['syntactic_features'].append("relative_clause")
        if any(token.dep_ == "advcl" for token in doc):
            features['syntactic_features'].append("adverbial_clause")
            
        return features
    
    def _analyze_vocabulary(self, doc) -> Dict:
        """Analyze vocabulary sophistication."""
        features = {
            'rare_words': 0,
            'technical_terms': 0,
            'abstract_concepts': 0,
            'avg_word_length': 0
        }
        
        words = [token for token in doc if token.is_alpha and not token.is_stop]
        if not words:
            return features
            
        # Count rare words (using spaCy's frequency data)
        features['rare_words'] = len([w for w in words if w.rank > 10000])
        
        # Identify technical terms (longer words, compound nouns)
        features['technical_terms'] = len([
            token for token in doc.noun_chunks
            if len(token.text) > 8 or (token.root.pos_ == "NOUN" and 
            any(child.pos_ == "NOUN" for child in token.root.children))
        ])
        
        # Check for abstract concepts using word vectors
        for token in words:
            if token.has_vector:
                max_similarity = max(token.similarity(concept) 
                                  for concept in self.abstract_concepts)
                if max_similarity > 0.5:
                    features['abstract_concepts'] += 1
        
        features['avg_word_length'] = sum(len(w.text) for w in words) / len(words)
        
        return features
    
    def _analyze_logical_structure(self, doc) -> Dict:
        """Analyze logical relationships in the sentence."""
        features = {
            'has_causal': False,
            'has_comparison': False,
            'has_conditional': False
        }
        
        # Causal relationships
        causal_markers = ["because", "therefore", "thus", "hence", "so"]
        features['has_causal'] = any(token.text.lower() in causal_markers 
                                   for token in doc)
        
        # Comparisons
        comparison_deps = ["comparison", "prep_than", "prep_as"]
        features['has_comparison'] = any(token.dep_ in comparison_deps 
                                       for token in doc)
        
        # Conditional statements
        conditional_markers = ["if", "unless", "whether"]
        features['has_conditional'] = any(token.text.lower() in conditional_markers 
                                        for token in doc)
        
        return features
    
    def calculate_total_score(self, scores: Dict) -> float:
        """Calculate weighted complexity score."""
        total = 0.0
        
        # Syntactic complexity (0-5)
        total += min(5, scores['syntactic']['clause_count'] * 1.0 +
                       scores['syntactic']['tree_depth'] * 0.5)
        
        # Vocabulary sophistication (0-5)
        vocab = scores['vocabulary']
        total += min(5, vocab['rare_words'] * 0.5 +
                       vocab['technical_terms'] * 1.0 +
                       vocab['abstract_concepts'] * 1.0)
        
        # Logical complexity (0-3)
        logical = scores['logical']
        total += sum(1.0 for feature in logical.values() if feature)
        
        return total

    def get_complexity_description(self, scores: Dict) -> str:
        """Generate human-readable complexity description."""
        total = scores['total']
        
        if total < 3:
            return "Simple but clear"
        elif total < 6:
            return "Moderately complex"
        elif total < 9:
            return "Complex academic/technical"
        else:
            return "Highly complex"

class SentenceFilter:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_md")
        except OSError as e:
            logger.error(f"Error loading spaCy model: {e}")
            logger.error("Please run: python -m spacy download en_core_web_md")
            raise
            
        config = Config.load()
        self.config = config.sentence_filter
        if not self.config:
            logger.error("Missing sentence_filter section in config")
            raise ValueError("Missing sentence_filter configuration")
    
    def analyze_complexity(self, doc) -> Dict:
        """Analyze sentence complexity using various metrics."""
        features = {
            'has_subordinate_clause': False,
            'has_relative_clause': False,
            'has_conjunction': False,
            'has_adverbial_clause': False,
            'clause_count': 0
        }
        
        # Check for subordinate clauses (because, although, if, etc.)
        features['has_subordinate_clause'] = any(
            token.dep_ == 'mark' for token in doc
        )
        
        # Check for relative clauses (who, which, that, etc.)
        features['has_relative_clause'] = any(
            token.dep_ == 'relcl' for token in doc
        )
        
        # Check for coordinating conjunctions (and, but, or)
        features['has_conjunction'] = any(
            token.dep_ == 'cc' for token in doc
        )
        
        # Check for adverbial clauses
        features['has_adverbial_clause'] = any(
            token.dep_ == 'advcl' for token in doc
        )
        
        # Count clauses (rough estimate based on verbs)
        features['clause_count'] = len([
            token for token in doc 
            if token.pos_ == 'VERB' and token.dep_ != 'aux'
        ])
        
        return features

    def get_complexity_score(self, features: Dict) -> int:
        """Calculate overall complexity score."""
        score = 0
        
        if features['has_subordinate_clause']:
            score += 1
        if features['has_relative_clause']:
            score += 1
        if features['has_conjunction']:
            score += 1
        if features['has_adverbial_clause']:
            score += 1
        if features['clause_count'] >= 2:
            score += 1
            
        return score

    def is_good_sentence(self, sentence: str) -> bool:
        """Determine if a sentence is suitable for the dataset."""
        try:
            # Basic text cleanup
            sentence = sentence.strip()
            
            # Basic length checks
            if not (self.config.min_chars <= len(sentence) <= self.config.max_chars):
                return False
                
            # Process with spaCy
            doc = self.nlp(sentence)
            
            # Word count check
            word_count = len([token for token in doc if not token.is_punct])
            if not (self.config.min_words <= word_count <= self.config.max_words):
                return False
            
            # Named entity checks
            entity_counts = {'PERSON': 0, 'GPE': 0, 'ORG': 0}
            for ent in doc.ents:
                if ent.label_ in entity_counts:
                    entity_counts[ent.label_] += 1
                    if entity_counts[ent.label_] > getattr(self.config, f'max_{ent.label_.lower()}s'):
                        return False
            
            if sum(entity_counts.values()) > self.config.max_entities:
                return False
            
            # Complexity analysis
            features = self.analyze_complexity(doc)
            complexity_score = self.get_complexity_score(features)
            
            return complexity_score >= self.config.min_complexity_score
            
        except Exception as e:
            logger.warning(f"Error checking sentence: {e}")
            return False

    def _validate_config(self):
        """Validate configuration values."""
        required_keys = {
            'min_chars', 'max_chars', 'max_entities', 
            'max_persons', 'max_locations', 'max_organizations',
            'min_complexity_score', 'pos_ratios'
        }
        
        missing_keys = required_keys - set(self.config.keys())
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")
        
        # Validate ranges
        if self.config['min_chars'] > self.config['max_chars']:
            raise ValueError("min_chars cannot be greater than max_chars")
        
        if self.config['min_words'] > self.config['max_words']:
            raise ValueError("min_words cannot be greater than max_words")

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
        
        # Get config from sentence filter
        self.config = self.sentence_filter.config
    
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

    def _load_trashed_sentences(self) -> None:
        """Load trashed sentences from file."""
        if self.trash_file.exists():
            try:
                with open(self.trash_file) as f:
                    data = json.load(f)
                    self.trashed_sentences = data if isinstance(data, list) else []
            except Exception as e:
                logger.error(f"Error loading trashed sentences: {e}")
    
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

    def trash_sentence(self, sentence: str) -> None:
        """Move a sentence to the trash list.
        
        Args:
            sentence: The sentence to trash
        """
        # Remove from available sentences if present
        if sentence in self.sentences:
            self.sentences.remove(sentence)
            self._save_sentences()
        
        # Add to trashed sentences if not already there
        if sentence not in self.trashed_sentences:
            self.trashed_sentences.append(sentence)
            self._save_trashed_sentences()
            logger.info(f"Sentence moved to trash: {sentence}")

    def _is_good_sentence(self, sentence: str) -> bool:
        """Check if a sentence is suitable using enhanced complexity analysis."""
        try:
            # Basic checks first (faster)
            if not sentence or not sentence[0].isupper() or sentence[-1] not in '.!?':
                return False
                
            # Analyze complexity
            scores = self.complexity_analyzer.analyze_sentence(sentence)
            if not scores:  # Analysis failed
                return False
                
            # Check complexity thresholds
            if scores['total'] < self.sentence_filter.config.min_total_complexity:
                return False
                
            # Check entity limits with correct attribute names
            entity_limits = {
                'PERSON': self.sentence_filter.config.max_persons,
                'GPE': self.sentence_filter.config.max_locations,
                'ORG': self.sentence_filter.config.max_organizations
            }
            
            doc = self.complexity_analyzer.nlp(sentence)
            entity_counts = {'PERSON': 0, 'GPE': 0, 'ORG': 0}
            for ent in doc.ents:
                if ent.label_ in entity_counts:
                    entity_counts[ent.label_] += 1
                    if entity_counts[ent.label_] > entity_limits[ent.label_]:
                        return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error in _is_good_sentence: {e}")
            return False
    
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
                    docs = self.complexity_analyzer.nlp.pipe(
                        sentences,
                        batch_size=BATCH_SIZE,
                        disable=['textcat']  # Keep parser for dependency checks
                    )
                    
                    with tqdm(total=count, desc="Validating sentences") as pbar:
                        for doc in docs:
                            if len(candidates) >= count * 2:
                                break
                                
                            # Quick check first
                            if self._is_good_sentence_quick(doc):
                                quick_passed += 1
                                # Full validation only if quick check passes
                                if self._is_good_sentence(doc):  # Pass doc instead of sentence
                                    full_passed += 1
                                    candidates.append(doc.text)
                                    pbar.update(1)
                    
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
            scores = self.complexity_analyzer.analyze_sentence(doc)
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
        trashed = len(self.trashed_sentences)
        
        return {
            "total_sentences": total,
            "recorded_sentences": recorded,
            "remaining_sentences": len(self.sentences),
            "trashed_sentences": trashed,
            "completion_percentage": (recorded / total * 100) if total > 0 else 0,
            "total_audio_time": recorded * 10 / 60  # Assuming 10 seconds per recording
        } 