"""Sentence filtering and analysis functionality."""

import logging
import spacy
import json
from pathlib import Path
from typing import Dict, Optional
from .config import Config, config  # Direct import since it's in same directory
from .nlp_manager import NLPManager
import enchant

logger = logging.getLogger(__name__)

class ComplexityAnalyzer:
    def __init__(self):
        """Initialize without loading model."""
        self.nlp_manager = NLPManager.get_instance()
        self.abstract_concepts = None
        self.config = config
    
    def _ensure_abstract_concepts(self):
        """Lazy load abstract concepts."""
        if self.abstract_concepts is None:
            nlp = self.nlp_manager.get_nlp()
            self.abstract_concepts = list(nlp.pipe([
                "theory", "concept", "philosophy", "analysis",
                "knowledge", "understanding", "perspective"
            ]))
    
    def analyze_sentence(self, text: str) -> Dict:
        """Complete sentence analysis."""
        try:
            self._ensure_abstract_concepts()
            nlp = self.nlp_manager.get_nlp()
            doc = nlp(text)
            
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
        """Initialize without loading model."""
        self.nlp_manager = NLPManager.get_instance()
        self.complexity_analyzer = ComplexityAnalyzer()  # Add analyzer
        
        # More realistic timing constants
        self.syllable_duration = 0.3    # 300ms per syllable
        self.word_boundary_pause = 0.1   # 100ms between words
        self.pause_duration = {
            '.': 0.7,    # End of sentence
            ',': 0.3,    # Comma
            ';': 0.4,    # Semicolon
            ':': 0.4,    # Colon
            '(': 0.3,    # Opening parenthesis
            ')': 0.3,    # Closing parenthesis
            '?': 0.7,    # Question mark
            '!': 0.7,    # Exclamation mark
        }
        
        config = Config.load()
        self.config = config.sentence_filter
        if not self.config:
            logger.error("Missing sentence_filter section in config")
            raise ValueError("Missing sentence_filter configuration")
        
        # Validate required config sections
        self._validate_config()

        # Load existing sentences
        self.sentences_file = Path("data/repository/sentences.json")
        self.used_sentences_file = Path("data/repository/used_sentences.json")
        self.trashed_sentences_file = Path("data/repository/trashed_sentences.json")
        self._load_existing_sentences()

        # Initialize British and American English dictionaries
        self.british_dict = enchant.Dict("en_GB")
        self.american_dict = enchant.Dict("en_US")

    def _load_existing_sentences(self):
        """Load existing sentences from JSON files."""
        self.existing_sentences = set()
        for file_path in [self.sentences_file, self.used_sentences_file, self.trashed_sentences_file]:
            if file_path.exists():
                with open(file_path) as f:
                    self.existing_sentences.update(json.load(f))

    def standardize_spelling(self, sentence: str) -> str:
        """Standardize British spellings to American spellings in a sentence."""
        words = sentence.split()
        standardized_words = []
        
        for word in words:
            if self.british_dict.check(word) and not self.american_dict.check(word):
                suggestions = self.american_dict.suggest(word)
                if suggestions:
                    standardized_words.append(suggestions[0])
                else:
                    standardized_words.append(word)
            else:
                standardized_words.append(word)
        
        return ' '.join(standardized_words)

    def estimate_duration(self, text: str) -> float:
        """Estimate the duration of speaking this sentence without using NLP."""
        total_duration = 0.0
        
        # Split into words and punctuation
        import re
        tokens = re.findall(r"[\w']+|[.,!?;:]", text)
        
        for i, token in enumerate(tokens):
            # Check if token is punctuation
            if token in self.pause_duration:
                total_duration += self.pause_duration[token]
                continue
            
            # Add syllable duration for words
            syllables = self._count_syllables(token)
            total_duration += syllables * self.config.timing.syllable_duration
            
            # Add word boundary pauses (except for last word)
            if i < len(tokens) - 1 and token not in self.pause_duration:
                total_duration += self.config.timing.word_boundary_pause
            
            # Add extra time for longer words (cognitive processing)
            if len(token) > 6:
                total_duration += self.config.timing.long_word_penalty
        
        # Add general comprehension buffer
        total_duration *= self.config.timing.comprehension_buffer
        
        return total_duration

    def _count_syllables(self, word: str) -> int:
        """
        Estimate syllable count for a word.
        This is a simple implementation - could be improved with a dictionary.
        """
        word = word.lower()
        count = 0
        vowels = 'aeiouy'
        prev_char_is_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_char_is_vowel:
                count += 1
            prev_char_is_vowel = is_vowel
            
        # Handle silent e
        if word.endswith('e'):
            count -= 1
        # Ensure at least one syllable
        return max(1, count)

    def is_good_sentence(self, sentence: str) -> bool:
        """Two-stage sentence validation."""
        try:
            # Standardize spelling before any checks
            sentence = self.standardize_spelling(sentence)

            # Quick filter: Check if sentence is already used or trashed
            if sentence in self.existing_sentences:
                logger.debug("Sentence already exists in one of the JSON files")
                return False

            # Stage 1: Quick checks first (no NLP)
            logger.debug("Stage 1: Quick filtering")
            if not self._check_basic_length(sentence):
                return False

            # Get NLP doc once for all subsequent checks
            doc = self.nlp_manager.get_nlp()(sentence)

            # Stage 2: Detailed NLP analysis
            logger.debug("Stage 2: Detailed NLP analysis")
            if not self._is_good_sentence(doc):
                return False

            # Final stage: Duration check
            if hasattr(self.config, 'timing'):
                estimated_duration = self.estimate_duration(sentence)
                if not (self.config.timing.min_duration <= estimated_duration <= self.config.timing.max_duration):
                    logger.debug(f"Failed duration check: {estimated_duration:.1f}s")
                    return False

            return True

        except Exception as e:
            logger.warning(f"Error in sentence validation: {e}")
            return False

    def analyze_complexity(self, doc) -> Dict:
        """Analyze sentence complexity using various metrics."""
        features = {
            'has_subordinate_clause': False,
            'has_relative_clause': False,
            'has_conjunction': False,
            'has_adverbial_clause': False,
            'clause_count': 0
        }
        
        # Check for subordinate clauses
        features['has_subordinate_clause'] = any(
            token.dep_ == 'mark' for token in doc
        )
        
        # Check for relative clauses
        features['has_relative_clause'] = any(
            token.dep_ == 'relcl' for token in doc
        )
        
        # Check for coordinating conjunctions
        features['has_conjunction'] = any(
            token.dep_ == 'cc' for token in doc
        )
        
        # Check for adverbial clauses
        features['has_adverbial_clause'] = any(
            token.dep_ == 'advcl' for token in doc
        )
        
        # Count clauses
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

    def _check_complexity(self, doc) -> bool:
        """Check sentence complexity using ranges."""
        try:
            # Safety check for empty doc
            if len(doc) == 0:
                logger.debug("Empty document")
                return False

            # Get clause count (with error handling)
            try:
                clauses = len([token for token in doc if token.dep_ == "ROOT"])
                if not (self.config.complexity_ranges.clause_count[0] <= 
                       clauses <= 
                       self.config.complexity_ranges.clause_count[1]):
                    logger.debug(f"Failed clause count: {clauses}")
                    return False
            except Exception as e:
                logger.warning(f"Error counting clauses: {e}")
                return False

            # Tree depth check (with error handling)
            try:
                def get_tree_depth(token):
                    return 1 + max((get_tree_depth(child) for child in token.children), default=0)
                
                roots = [token for token in doc if token.head == token]
                if not roots:
                    logger.debug("No root found in parse tree")
                    return False
                    
                tree_depth = get_tree_depth(roots[0])
                if not (self.config.complexity_ranges.tree_depth[0] <= 
                       tree_depth <= 
                       self.config.complexity_ranges.tree_depth[1]):
                    logger.debug(f"Failed tree depth: {tree_depth}")
                    return False
            except Exception as e:
                logger.warning(f"Error calculating tree depth: {e}")
                return False

            # Word length check (with error handling)
            try:
                content_tokens = [t for t in doc if not t.is_punct]
                if not content_tokens:
                    logger.debug("No content tokens found")
                    return False
                    
                avg_word_length = sum(len(t.text) for t in content_tokens) / len(content_tokens)
                if not (self.config.complexity_ranges.word_length[0] <= 
                       avg_word_length <= 
                       self.config.complexity_ranges.word_length[1]):
                    logger.debug(f"Failed word length: {avg_word_length:.1f}")
                    return False
            except Exception as e:
                logger.warning(f"Error calculating word length: {e}")
                return False

            return True
            
        except Exception as e:
            logger.warning(f"Error in complexity check: {e}")
            return False

    def _check_pos_ratios(self, doc) -> bool:
        """Check if POS ratios are within ranges."""
        try:
            content_words = len([token for token in doc if not token.is_stop and not token.is_punct])
            if content_words == 0:
                return False

            pos_counts = {}
            for token in doc:
                if not token.is_stop and not token.is_punct:
                    pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1

            for pos, range_values in self.config.pos_ratios.items():
                ratio = pos_counts.get(pos, 0) / content_words
                min_ratio, max_ratio = range_values
                if not (min_ratio <= ratio <= max_ratio):
                    logger.debug(f"Failed POS ratio for {pos}: {ratio:.2f} not in range [{min_ratio:.2f}, {max_ratio:.2f}]")
                    return False

            return True
        except Exception as e:
            logger.warning(f"Error checking POS ratios: {e}")
            return False


    def _validate_config(self):
        """Validate required configuration sections."""
        required_keys = {
            'min_chars', 'max_chars', 'min_words', 'max_words',
            'max_entities', 'max_persons', 'max_locations', 'max_organizations',
            'min_complexity_score', 'min_total_complexity'
        }
        
        # Convert dataclass to dict for validation
        config_dict = vars(self.config)
        missing_keys = required_keys - set(config_dict.keys())
        
        if missing_keys:
            logger.error(f"Missing required config keys: {missing_keys}")
            raise ValueError(f"Missing required configuration keys: {missing_keys}")

    def _check_basic_length(self, text: str) -> bool:
        """Check basic length constraints from config."""
        try:
            # Character length check
            char_count = len(text)
            if not (self.config.min_chars <= char_count <= self.config.max_chars):
                logger.debug(f"Failed char count: {char_count} not in [{self.config.min_chars}, {self.config.max_chars}]")
                return False
            
            # Word count check
            word_count = len(text.split())
            if not (self.config.min_words <= word_count <= self.config.max_words):
                logger.debug(f"Failed word count: {word_count} not in [{self.config.min_words}, {self.config.max_words}]")
                return False
            
            return True
        except Exception as e:
            logger.warning(f"Error in basic length check: {e}")
            return False

    def _check_entities(self, doc) -> bool:
        """Check entity limits from config."""
        try:
            entity_counts = {
                'PERSON': 0,
                'GPE': 0,  # Locations
                'ORG': 0,
                'total': 0
            }
            
            for ent in doc.ents:
                if ent.label_ in entity_counts:
                    entity_counts[ent.label_] += 1
                entity_counts['total'] += 1
                
            if entity_counts['total'] > self.config.max_entities:
                logger.debug(f"Too many entities: {entity_counts['total']}")
                return False
                
            if entity_counts['PERSON'] > self.config.max_persons:
                logger.debug(f"Too many persons: {entity_counts['PERSON']}")
                return False
                
            if entity_counts['GPE'] > self.config.max_locations:
                logger.debug(f"Too many locations: {entity_counts['GPE']}")
                return False
                
            if entity_counts['ORG'] > self.config.max_organizations:
                logger.debug(f"Too many organizations: {entity_counts['ORG']}")
                return False
                
            return True
        except Exception as e:
            logger.warning(f"Error checking entities: {e}")
            return False

    def _is_good_sentence(self, doc) -> bool:
        """Enhanced NLP validation using config settings and spaCy features."""
        try:
            # Entity checks first (using config limits)
            if not self._check_entities(doc):
                return False

            # 1. Structural Complexity (from config)
            features = self.analyze_complexity(doc)
            
            # Check clause count range
            if not (self.config.complexity_ranges.clause_count[0] <= 
                    features['clause_count'] <= 
                    self.config.complexity_ranges.clause_count[1]):
                logger.debug(f"Failed clause count: {features['clause_count']}")
                return False
            
            # Check tree depth range
            tree_depth = max(len(list(token.ancestors)) for token in doc)
            if not (self.config.complexity_ranges.tree_depth[0] <= 
                    tree_depth <= 
                    self.config.complexity_ranges.tree_depth[1]):
                logger.debug(f"Failed tree depth: {tree_depth}")
                return False

            # 2. POS Distribution (from config)
            token_count = len([t for t in doc if not t.is_punct])
            for pos, ratio_range in self.config.pos_ratios.items():
                pos_count = len([t for t in doc if t.pos_ == pos])
                ratio = pos_count / token_count
                if not (ratio_range[0] <= ratio <= ratio_range[1]):
                    logger.debug(f"Failed {pos} ratio: {ratio:.2f}")
                    return False

            # 3. Overall Complexity Score (from config)
            complexity_score = (
                features['clause_count'] +
                (tree_depth / 2) +
                (1 if features['has_subordinate_clause'] else 0) +
                (1 if features['has_relative_clause'] else 0) +
                (1 if features['has_conjunction'] else 0) +
                (1 if features['has_adverbial_clause'] else 0)
            )
            
            if complexity_score < self.config.min_complexity_score:
                logger.debug(f"Failed complexity score: {complexity_score}")
                return False
            
            if sum(features.values()) < self.config.min_total_complexity:
                logger.debug(f"Failed total complexity: {sum(features.values())}")
                return False

            # 4. Basic Semantic Structure
            main_clauses = [t for t in doc if t.dep_ == "ROOT"]
            if len(main_clauses) != 1:
                logger.debug("Failed main clause check")
                return False
            
            subjects = [t for t in doc if "subj" in t.dep_]
            if not subjects:
                logger.debug("Failed subject check")
                return False

            return True

        except Exception as e:
            logger.warning(f"Error in enhanced NLP validation: {e}")
            return False
