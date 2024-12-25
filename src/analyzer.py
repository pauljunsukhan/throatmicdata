"""
Audio quality and dataset analysis module.
"""

import os
from typing import Dict, List, Optional, Tuple
import json
import csv
from datetime import datetime
import numpy as np
import librosa
import soundfile as sf
import statistics
from . import SentenceFilter, ComplexityAnalyzer, Config
import logging
from .nlp_manager import NLPManager
from tqdm import tqdm
from pathlib import Path
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

class AudioQualityControl:
    """Audio quality control class for analyzing individual recordings."""
    
    def __init__(self, config):
        self.sample_rate = config.audio.sample_rate
        self.min_duration = config.audio.min_duration
        self.max_duration = config.audio.max_duration
        self.min_amplitude = config.audio.min_level_threshold
        self.max_amplitude = config.audio.clipping_threshold
        self.silence_threshold = 0.3
        self.min_snr = 10.0
        
    def analyze_recording(self, audio_file: str) -> Dict:
        """Analyze a single audio recording for quality metrics."""
        try:
            y, sr = librosa.load(audio_file, sr=self.sample_rate)
            duration = librosa.get_duration(y=y, sr=sr)
            
            amplitude_db = librosa.amplitude_to_db(np.abs(y), ref=np.max)
            mean_amplitude = np.mean(amplitude_db)
            max_amplitude = np.max(amplitude_db)
            
            silence_mask = amplitude_db < self.silence_threshold
            silence_ratio = np.sum(silence_mask) / len(y)
            
            signal = y[~silence_mask] if np.any(~silence_mask) else y
            noise = y[silence_mask] if np.any(silence_mask) else np.zeros_like(y)
            signal_power = np.mean(signal**2)
            noise_power = np.mean(noise**2) if np.any(noise) else 1e-10
            snr = 10 * np.log10(signal_power / noise_power)
            
            logger.debug(f"Analyzing {os.path.basename(audio_file)}:")
            logger.debug(f"  Duration: {duration:.1f}s")
            logger.debug(f"  Mean amplitude: {mean_amplitude:.1f}dB")
            logger.debug(f"  Max amplitude: {max_amplitude:.1f}dB")
            logger.debug(f"  Silence ratio: {silence_ratio*100:.1f}%")
            logger.debug(f"  SNR: {snr:.1f}dB")
            
            return {
                'duration': duration,
                'mean_amplitude_db': float(mean_amplitude),
                'max_amplitude_db': float(max_amplitude),
                'silence_ratio': float(silence_ratio),
                'snr_db': float(snr),
                'quality_issues': self._check_quality_issues(
                    duration, mean_amplitude, max_amplitude, silence_ratio, snr
                )
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {audio_file}: {str(e)}")
            return {
                'error': str(e),
                'quality_issues': ['Failed to analyze recording']
            }
    
    def _check_quality_issues(
        self, duration: float, mean_amplitude: float, 
        max_amplitude: float, silence_ratio: float, snr: float
    ) -> List[str]:
        """Check for quality issues in the recording."""
        issues = []
        
        if duration < self.min_duration:
            issues.append(f'Recording too short ({duration:.1f}s < {self.min_duration}s)')
        elif duration > self.max_duration:
            issues.append(f'Recording too long ({duration:.1f}s > {self.max_duration}s)')
            
        if mean_amplitude < self.min_amplitude:
            issues.append(f'Recording too quiet (mean {mean_amplitude:.1f}dB < {self.min_amplitude}dB)')
        elif max_amplitude > self.max_amplitude:
            issues.append(f'Recording too loud (max {max_amplitude:.1f}dB > {self.max_amplitude}dB)')
            
        if silence_ratio > self.silence_threshold:
            issues.append(f'Too much silence ({silence_ratio*100:.1f}% > {self.silence_threshold*100}%)')
            
        if snr < self.min_snr:
            issues.append(f'Poor signal-to-noise ratio ({snr:.1f}dB < {self.min_snr}dB)')
            
        return issues
    
class DatasetAnalytics:
    """Analytics class for the entire dataset."""
    
    def __init__(self, config: Config):
        self.config = config
        self.quality_control = AudioQualityControl(self.config)
        self.sentence_filter = SentenceFilter()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.base_dir = Path(self.config.dataset.recordings_dir)
        self._rows = None
        
    def analyze_dataset(self, metadata_file: str) -> Dict:
        """Analyze all recordings in the dataset."""
        try:
            return self._analyze_dataset_internal(metadata_file)
        except Exception as e:
            logger.error(f"Error analyzing dataset: {str(e)}")
            return {'error': str(e)}

    def _analyze_dataset_internal(self, metadata_file: str) -> Dict:
        """Analyze all recordings in the dataset."""
        try:
            with open(metadata_file, 'r') as f:
                reader = csv.DictReader(f)
                self._rows = list(reader)
                
            total_recordings = len(self._rows)
            if total_recordings == 0:
                return {'error': 'No recordings found in metadata'}
                
            print("Analyzing audio files...")
            quality_issues = []
            total_duration = 0
            recordings_with_issues = 0
            durations = []
            valid_recordings = 0
            
            for row in tqdm(self._rows, desc="Analyzing", total=total_recordings):
                audio_path = None
                try:
                    # Use recordings_dir instead of base_dir
                    audio_path = self.recordings_dir / Path(row['audio_filepath']).name
                    
                    with sf.SoundFile(str(audio_path)) as f:
                        duration = len(f) / f.samplerate
                        total_duration += duration
                        durations.append(duration)
                        valid_recordings += 1
                    
                    quality_result = self.quality_control.analyze_recording(str(audio_path))
                    
                    if quality_result.get('quality_issues'):
                        quality_issues.append({
                            'file': str(audio_path),
                            'issues': quality_result['quality_issues']
                        })
                        recordings_with_issues += 1
                    
                except Exception as e:
                    error_file = str(audio_path) if audio_path else f"filepath: {row.get('audio_filepath', 'unknown')}"
                    logger.error(f"Error analyzing {error_file}: {str(e)}")
                    quality_issues.append({
                        'file': error_file,
                        'error': str(e)
                    })
                    recordings_with_issues += 1

            # Handle the case where no valid recordings were processed
            if not durations:
                return {
                    'error': 'No valid recordings could be processed',
                    'total_recordings': total_recordings,
                    'recordings_with_issues': recordings_with_issues,
                    'quality_issues': quality_issues
                }

            duration_analysis = self._calculate_duration_stats(durations)
            
            print("\nAnalyzing text complexity...")
            texts = [row['text'] for row in self._rows]
            complexity_scores = []
            syntactic_features = {'tree_depth': [], 'subordinate': 0, 'relative': 0, 'adverbial': 0}
            vocabulary_features = {'rare_words': [], 'technical_terms': [], 'abstract_concepts': [], 'word_length': []}
            logical_features = {'causal': 0, 'comparison': 0, 'conditional': 0}
            
            for text in tqdm(texts, desc="Processing", total=len(texts)):
                scores = self.complexity_analyzer.analyze_sentence(text)
                if scores:
                    complexity_scores.append(scores['total'])
                    
                    syn = scores['syntactic']
                    syntactic_features['tree_depth'].append(syn['tree_depth'])
                    if 'subordinate_clause' in syn['syntactic_features']:
                        syntactic_features['subordinate'] += 1
                    if 'relative_clause' in syn['syntactic_features']:
                        syntactic_features['relative'] += 1
                    if 'adverbial_clause' in syn['syntactic_features']:
                        syntactic_features['adverbial'] += 1
                    
                    vocab = scores['vocabulary']
                    vocabulary_features['rare_words'].append(vocab['rare_words'])
                    vocabulary_features['technical_terms'].append(vocab['technical_terms'])
                    vocabulary_features['abstract_concepts'].append(vocab['abstract_concepts'])
                    vocabulary_features['word_length'].append(vocab['avg_word_length'])
                    
                    logical = scores['logical']
                    if logical['has_causal']: logical_features['causal'] += 1
                    if logical['has_comparison']: logical_features['comparison'] += 1
                    if logical['has_conditional']: logical_features['conditional'] += 1
            
            avg_complexity = statistics.mean(complexity_scores) if complexity_scores else 0
            
            results = {
                'total_recordings': total_recordings,
                'valid_recordings': valid_recordings,
                'total_duration_minutes': total_duration / 60,
                'recordings_with_issues': recordings_with_issues,
                'quality_issues': quality_issues,
                'duration_analysis': duration_analysis,
                'sentence_complexity': {
                    'avg_complexity': avg_complexity,
                    'complexity_distribution': self._get_complexity_distribution(complexity_scores),
                    'syntactic_features': {
                        'avg_tree_depth': statistics.mean(syntactic_features['tree_depth']) if syntactic_features['tree_depth'] else 0,
                        'pct_subordinate': (syntactic_features['subordinate'] / total_recordings) * 100,
                        'pct_relative': (syntactic_features['relative'] / total_recordings) * 100,
                        'pct_adverbial': (syntactic_features['adverbial'] / total_recordings) * 100
                    },
                    'vocabulary_features': {
                        'avg_rare_words': statistics.mean(vocabulary_features['rare_words']) if vocabulary_features['rare_words'] else 0,
                        'avg_technical_terms': statistics.mean(vocabulary_features['technical_terms']) if vocabulary_features['technical_terms'] else 0,
                        'avg_abstract_concepts': statistics.mean(vocabulary_features['abstract_concepts']) if vocabulary_features['abstract_concepts'] else 0,
                        'avg_word_length': statistics.mean(vocabulary_features['word_length']) if vocabulary_features['word_length'] else 0
                    },
                    'logical_features': {
                        'pct_causal': (logical_features['causal'] / total_recordings) * 100,
                        'pct_comparison': (logical_features['comparison'] / total_recordings) * 100,
                        'pct_conditional': (logical_features['conditional'] / total_recordings) * 100
                    }
                },
                'coverage_analysis': {
                    'vocabulary_coverage': self._analyze_vocabulary_coverage(texts),
                    'structural_coverage': self._analyze_structural_coverage(texts)
                }
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing dataset: {str(e)}")
            return {'error': str(e)}
        
    def _get_complexity_distribution(self, scores: List[float]) -> Dict[str, int]:
        """Calculate complexity distribution."""
        distribution = {
            'simple': 0,
            'moderate': 0,
            'complex': 0,
            'very_complex': 0
        }
        
        for score in scores:
            if score < 3:
                distribution['simple'] += 1
            elif score < 6:
                distribution['moderate'] += 1
            elif score < 9:
                distribution['complex'] += 1
            else:
                distribution['very_complex'] += 1
                
        return distribution

    def _analyze_vocabulary_coverage(self, texts: List[str]) -> Dict:
        """Analyze vocabulary coverage."""
        nlp = self.complexity_analyzer.nlp_manager.get_nlp()
        words = []
        for text in texts:
            doc = nlp(text)
            words.extend([token.text.lower() for token in doc if token.is_alpha])
        
        word_freq = Counter(words)
        return {
            'total_words': len(words),
            'unique_words': len(word_freq),
            'vocabulary_density': len(word_freq) / len(words) if words else 0,
            'top_words': word_freq.most_common(20)
        }

    def _analyze_structural_coverage(self, texts: List[str]) -> Dict:
        """Analyze structural coverage."""
        nlp = self.complexity_analyzer.nlp_manager.get_nlp()
        patterns = set()
        lengths = []
        pos_dist = defaultdict(int)
        
        for text in texts:
            doc = nlp(text)
            pattern = ' '.join(token.pos_ for token in doc)
            patterns.add(pattern)
            lengths.append(len([t for t in doc if not t.is_punct]))
            for token in doc:
                if not token.is_punct:
                    pos_dist[token.pos_] += 1
        
        return {
            'unique_patterns': len(patterns),
            'sentence_length_stats': {
                'mean': statistics.mean(lengths) if lengths else 0,
                'median': statistics.median(lengths) if lengths else 0,
                'min': min(lengths) if lengths else 0,
                'max': max(lengths) if lengths else 0
            },
            'pos_distribution': dict(pos_dist)
        }

    def _calculate_duration_stats(self, durations: List[float]) -> Dict:
        """Calculate duration statistics."""
        mean = statistics.mean(durations)
        median = statistics.median(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        std_dev = statistics.stdev(durations) if len(durations) > 1 else 0
        
        distribution = {
            '<6s': 0,
            '6-8s': 0,
            '8-10s': 0,
            '10-12s': 0,
            '12-15s': 0,
            '>15s': 0
        }
        
        for duration in durations:
            if duration < 6:
                distribution['<6s'] += 1
            elif duration < 8:
                distribution['6-8s'] += 1
            elif duration < 10:
                distribution['8-10s'] += 1
            elif duration < 12:
                distribution['10-12s'] += 1
            elif duration < 15:
                distribution['12-15s'] += 1
            else:
                distribution['>15s'] += 1
        
        return {
            'mean': mean,
            'median': median,
            'min': min_duration,
            'max': max_duration,
            'std': std_dev,
            'distribution': distribution
        }

class DatasetAnalyzer(DatasetAnalytics):
    """Analyzer class that handles formatting and display of analytics results."""
    
    def __init__(self, config):
        self.config = config
        self.base_dir = Path(self.config.dataset.recordings_dir).parent
        self.recordings_dir = Path(self.config.dataset.recordings_dir)
        self.quality_control = AudioQualityControl(config)
        self.complexity_analyzer = ComplexityAnalyzer()
        self.last_analysis = None

    def analyze_dataset(self, metadata_file: str) -> Dict:
        """Analyze dataset with custom display formatting."""
        results = super()._analyze_dataset_internal(metadata_file)
        
        if results and 'error' not in results:
            print(self._format_analysis_display(results))
            self.last_analysis = results  # Cache the results
        
        return results

    def generate_report(self, metadata_file: str, output_file: Optional[str] = None) -> str:
        """Generate report using cached results if available."""
        if self.last_analysis:
            return self._generate_report_from_results(self.last_analysis, output_file)
        else:
            # Fall back to full analysis if no cached results
            return super().generate_report(metadata_file, output_file)

    def _format_analysis_display(self, analysis: Dict) -> str:
        """Format analysis results for display."""
        if 'error' in analysis:
            return f"Error in analysis: {analysis['error']}"

        pos_dist = analysis['coverage_analysis']['structural_coverage']['pos_distribution']
        total_pos = sum(pos_dist.values())

        display = f"""
Quality Analysis:
Total Recordings: {analysis['total_recordings']}
Total Duration: {analysis['total_duration_minutes']:.1f} minutes
Recordings with Issues: {analysis['recordings_with_issues']} ({analysis['recordings_with_issues']/analysis['total_recordings']*100:.1f}%)

Duration Statistics:
  Mean: {analysis['duration_analysis']['mean']:.1f}s
  Median: {analysis['duration_analysis']['median']:.1f}s
  Range: {analysis['duration_analysis']['min']:.1f}s - {analysis['duration_analysis']['max']:.1f}s
  Std Dev: {analysis['duration_analysis']['std']:.1f}s

Duration Distribution:
  <6s:    {analysis['duration_analysis']['distribution']['<6s']} recordings ({analysis['duration_analysis']['distribution']['<6s']/analysis['total_recordings']*100:.1f}%)
  6-8s:   {analysis['duration_analysis']['distribution']['6-8s']} recordings ({analysis['duration_analysis']['distribution']['6-8s']/analysis['total_recordings']*100:.1f}%)
  8-10s:  {analysis['duration_analysis']['distribution']['8-10s']} recordings ({analysis['duration_analysis']['distribution']['8-10s']/analysis['total_recordings']*100:.1f}%)
  10-12s: {analysis['duration_analysis']['distribution']['10-12s']} recordings ({analysis['duration_analysis']['distribution']['10-12s']/analysis['total_recordings']*100:.1f}%)
  12-15s: {analysis['duration_analysis']['distribution']['12-15s']} recordings ({analysis['duration_analysis']['distribution']['12-15s']/analysis['total_recordings']*100:.1f}%)
  >15s:   {analysis['duration_analysis']['distribution']['>15s']} recordings ({analysis['duration_analysis']['distribution']['>15s']/analysis['total_recordings']*100:.1f}%)

Coverage Analysis:
Vocabulary:
  Total Words: {analysis['coverage_analysis']['vocabulary_coverage']['total_words']}
  Unique Words: {analysis['coverage_analysis']['vocabulary_coverage']['unique_words']}
  Vocabulary Density: {analysis['coverage_analysis']['vocabulary_coverage']['vocabulary_density']:.2f}

Sentence Structure:
  Average Length: {analysis['coverage_analysis']['structural_coverage']['sentence_length_stats']['mean']:.1f} words
  Length Range: {analysis['coverage_analysis']['structural_coverage']['sentence_length_stats']['min']} - {analysis['coverage_analysis']['structural_coverage']['sentence_length_stats']['max']} words
  Unique Patterns: {analysis['coverage_analysis']['structural_coverage']['unique_patterns']}

Part of Speech Distribution:"""

        for pos, count in sorted(pos_dist.items()):
            percentage = (count / total_pos) * 100 if total_pos > 0 else 0
            display += f"\n  {pos}: {percentage:.1f}%"

        display += f"""

Sentence Complexity:
Average Complexity Score: {analysis['sentence_complexity']['avg_complexity']:.1f}

Complexity Distribution:
  Simple: {analysis['sentence_complexity']['complexity_distribution']['simple']} sentences
  Moderate: {analysis['sentence_complexity']['complexity_distribution']['moderate']} sentences
  Complex: {analysis['sentence_complexity']['complexity_distribution']['complex']} sentences
  Very Complex: {analysis['sentence_complexity']['complexity_distribution']['very_complex']} sentences

Syntactic Features:
  Average Tree Depth: {analysis['sentence_complexity']['syntactic_features']['avg_tree_depth']:.1f}
  Subordinate Clauses: {analysis['sentence_complexity']['syntactic_features']['pct_subordinate']:.1f}%
  Relative Clauses: {analysis['sentence_complexity']['syntactic_features']['pct_relative']:.1f}%
  Adverbial Clauses: {analysis['sentence_complexity']['syntactic_features']['pct_adverbial']:.1f}%

Vocabulary Features:
  Average Rare Words: {analysis['sentence_complexity']['vocabulary_features']['avg_rare_words']:.1f}
  Average Technical Terms: {analysis['sentence_complexity']['vocabulary_features']['avg_technical_terms']:.1f}
  Average Abstract Concepts: {analysis['sentence_complexity']['vocabulary_features']['avg_abstract_concepts']:.1f}
  Average Word Length: {analysis['sentence_complexity']['vocabulary_features']['avg_word_length']:.1f}

Logical Features:
  Causal Relations: {analysis['sentence_complexity']['logical_features']['pct_causal']:.1f}%
  Comparisons: {analysis['sentence_complexity']['logical_features']['pct_comparison']:.1f}%
  Conditionals: {analysis['sentence_complexity']['logical_features']['pct_conditional']:.1f}%
"""
        return display

    def _generate_report_from_results(self, analysis: Dict, output_file: Optional[str] = None) -> str:
        """Generate report from analysis results."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        pos_dist = analysis['coverage_analysis']['structural_coverage']['pos_distribution']
        total_pos = sum(pos_dist.values())
        
        report = f"""Dataset Analysis Report
Generated: {timestamp}

Quality Analysis:
Total Recordings: {analysis['total_recordings']}
Total Duration: {analysis['total_duration_minutes']:.1f} minutes
Recordings with Issues: {analysis['recordings_with_issues']} ({analysis['recordings_with_issues']/analysis['total_recordings']*100:.1f}%)

Duration Statistics:
  Mean: {analysis['duration_analysis']['mean']:.1f}s
  Median: {analysis['duration_analysis']['median']:.1f}s
  Range: {analysis['duration_analysis']['min']:.1f}s - {analysis['duration_analysis']['max']:.1f}s
  Std Dev: {analysis['duration_analysis']['std']:.1f}s

Duration Distribution:
  <6s:    {analysis['duration_analysis']['distribution']['<6s']} recordings ({analysis['duration_analysis']['distribution']['<6s']/analysis['total_recordings']*100:.1f}%)
  6-8s:   {analysis['duration_analysis']['distribution']['6-8s']} recordings ({analysis['duration_analysis']['distribution']['6-8s']/analysis['total_recordings']*100:.1f}%)
  8-10s:  {analysis['duration_analysis']['distribution']['8-10s']} recordings ({analysis['duration_analysis']['distribution']['8-10s']/analysis['total_recordings']*100:.1f}%)
  10-12s: {analysis['duration_analysis']['distribution']['10-12s']} recordings ({analysis['duration_analysis']['distribution']['10-12s']/analysis['total_recordings']*100:.1f}%)
  12-15s: {analysis['duration_analysis']['distribution']['12-15s']} recordings ({analysis['duration_analysis']['distribution']['12-15s']/analysis['total_recordings']*100:.1f}%)
  >15s:   {analysis['duration_analysis']['distribution']['>15s']} recordings ({analysis['duration_analysis']['distribution']['>15s']/analysis['total_recordings']*100:.1f}%)

Coverage Analysis:
Vocabulary:
  Total Words: {analysis['coverage_analysis']['vocabulary_coverage']['total_words']}
  Unique Words: {analysis['coverage_analysis']['vocabulary_coverage']['unique_words']}
  Vocabulary Density: {analysis['coverage_analysis']['vocabulary_coverage']['vocabulary_density']:.2f}

Top 20 Most Frequent Words:"""
        
        for word, freq in analysis['coverage_analysis']['vocabulary_coverage']['top_words']:
            report += f"\n  {word}: {freq}"
        
        report += f"""

Sentence Structure:
  Average Length: {analysis['coverage_analysis']['structural_coverage']['sentence_length_stats']['mean']:.1f} words
  Length Range: {analysis['coverage_analysis']['structural_coverage']['sentence_length_stats']['min']} - {analysis['coverage_analysis']['structural_coverage']['sentence_length_stats']['max']} words
  Unique Patterns: {analysis['coverage_analysis']['structural_coverage']['unique_patterns']}

Part of Speech Distribution:"""

        for pos, count in sorted(pos_dist.items()):
            percentage = (count / total_pos) * 100 if total_pos > 0 else 0
            report += f"\n  {pos}: {count} ({percentage:.1f}%)"

        report += f"""

Sentence Complexity:
Average Complexity Score: {analysis['sentence_complexity']['avg_complexity']:.1f}

Complexity Distribution:
  Simple: {analysis['sentence_complexity']['complexity_distribution']['simple']} sentences
  Moderate: {analysis['sentence_complexity']['complexity_distribution']['moderate']} sentences
  Complex: {analysis['sentence_complexity']['complexity_distribution']['complex']} sentences
  Very Complex: {analysis['sentence_complexity']['complexity_distribution']['very_complex']} sentences

Syntactic Features:
  Average Tree Depth: {analysis['sentence_complexity']['syntactic_features']['avg_tree_depth']:.1f}
  Subordinate Clauses: {analysis['sentence_complexity']['syntactic_features']['pct_subordinate']:.1f}%
  Relative Clauses: {analysis['sentence_complexity']['syntactic_features']['pct_relative']:.1f}%
  Adverbial Clauses: {analysis['sentence_complexity']['syntactic_features']['pct_adverbial']:.1f}%

Vocabulary Features:
  Average Rare Words: {analysis['sentence_complexity']['vocabulary_features']['avg_rare_words']:.1f}
  Average Technical Terms: {analysis['sentence_complexity']['vocabulary_features']['avg_technical_terms']:.1f}
  Average Abstract Concepts: {analysis['sentence_complexity']['vocabulary_features']['avg_abstract_concepts']:.1f}
  Average Word Length: {analysis['sentence_complexity']['vocabulary_features']['avg_word_length']:.1f}

Logical Features:
  Causal Relations: {analysis['sentence_complexity']['logical_features']['pct_causal']:.1f}%
  Comparisons: {analysis['sentence_complexity']['logical_features']['pct_comparison']:.1f}%
  Conditionals: {analysis['sentence_complexity']['logical_features']['pct_conditional']:.1f}%

Quality Issues:"""
        
        if analysis['quality_issues']:
            for issue in analysis['quality_issues']:
                report += f"\n{os.path.basename(issue['file'])}:"
                if 'issues' in issue:
                    for problem in issue['issues']:
                        report += f"\n  - {problem}"
                else:
                    report += f"\n  - {issue['error']}"
        else:
            report += "\nNo quality issues found."
            
        if output_file:
            try:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w') as f:
                    f.write(report)
                logger.info(f"Report saved to {output_file}")
            except Exception as e:
                logger.error(f"Error saving report to file: {e}")
                
        return report