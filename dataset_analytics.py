import librosa
import numpy as np
from pathlib import Path
import json
import csv
from collections import Counter
import nltk
from typing import Dict, List, Optional
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('cmudict')

class AudioQualityControl:
    def __init__(self):
        self.AMPLITUDE_THRESHOLD = 0.1
        self.CLIPPING_THRESHOLD = 0.95
        self.MIN_DURATION = 8.0  # seconds
        
    def analyze_recording(self, audio_path: str) -> Dict[str, any]:
        """Analyze a single recording for quality issues"""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Calculate metrics
            peak_amplitude = np.max(np.abs(y))
            rms = np.sqrt(np.mean(y**2))
            has_clipping = np.any(np.abs(y) > self.CLIPPING_THRESHOLD)
            
            # Check for silence or near-silence
            silent_threshold = 0.01
            silent_regions = librosa.effects.split(y, top_db=20)
            total_silence = duration - sum((end - start) / sr for start, end in silent_regions)
            
            issues = []
            if peak_amplitude < self.AMPLITUDE_THRESHOLD:
                issues.append("Low volume")
            if has_clipping:
                issues.append("Audio clipping detected")
            if duration < self.MIN_DURATION:
                issues.append("Recording too short")
            if total_silence > duration * 0.3:  # More than 30% silence
                issues.append("Excessive silence")
            
            return {
                'path': audio_path,
                'duration': duration,
                'peak_amplitude': float(peak_amplitude),
                'rms_level': float(rms),
                'has_clipping': bool(has_clipping),
                'silence_percentage': float(total_silence / duration * 100),
                'issues': issues,
                'quality_score': self._calculate_quality_score(peak_amplitude, has_clipping, 
                                                             duration, total_silence/duration)
            }
        except Exception as e:
            return {'path': audio_path, 'error': str(e), 'issues': ['Failed to analyze']}
    
    def _calculate_quality_score(self, peak_amplitude: float, has_clipping: bool, 
                               duration: float, silence_ratio: float) -> float:
        """Calculate a quality score from 0 to 100"""
        score = 100.0
        
        # Penalize for low volume
        if peak_amplitude < self.AMPLITUDE_THRESHOLD:
            score -= 30 * (1 - peak_amplitude/self.AMPLITUDE_THRESHOLD)
        
        # Penalize for clipping
        if has_clipping:
            score -= 30
        
        # Penalize for duration issues
        if duration < self.MIN_DURATION:
            score -= 20 * (1 - duration/self.MIN_DURATION)
        
        # Penalize for excessive silence
        if silence_ratio > 0.3:
            score -= 20 * (silence_ratio - 0.3)
        
        return max(0, min(100, score))

class DatasetAnalytics:
    def __init__(self):
        self.phoneme_dict = nltk.corpus.cmudict.dict()
        
    def analyze_text_coverage(self, sentences: List[str]) -> Dict[str, any]:
        """Analyze text coverage including vocabulary and phonetics"""
        # Tokenize all sentences
        words = []
        for sentence in sentences:
            words.extend(nltk.word_tokenize(sentence.lower()))
        
        # Get vocabulary statistics
        vocab = Counter(words)
        
        # Get phonetic coverage
        phonemes = []
        for word in set(words):
            if word in self.phoneme_dict:
                phonemes.extend(self.phoneme_dict[word][0])
        phoneme_counts = Counter(phonemes)
        
        return {
            'vocabulary_size': len(vocab),
            'total_words': len(words),
            'unique_phonemes': len(phoneme_counts),
            'most_common_words': vocab.most_common(10),
            'most_common_phonemes': phoneme_counts.most_common(10),
            'average_sentence_length': len(words) / len(sentences),
            'word_frequency': dict(vocab),
            'phoneme_frequency': dict(phoneme_counts)
        }
    
    def analyze_dataset_balance(self, metadata_file: str) -> Dict[str, any]:
        """Analyze dataset balance and coverage"""
        with open(metadata_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            texts = [row['text'] for row in reader]
        
        # Get text coverage
        text_coverage = self.analyze_text_coverage(texts)
        
        # Add dataset-specific metrics
        text_coverage.update({
            'total_recordings': len(texts),
            'total_unique_sentences': len(set(texts)),
            'duplicate_sentences': len(texts) - len(set(texts))
        })
        
        return text_coverage
    
    def generate_report(self, metadata_file: str, output_file: Optional[str] = None) -> str:
        """Generate a human-readable report of dataset analytics"""
        analytics = self.analyze_dataset_balance(metadata_file)
        
        report = [
            "Dataset Analytics Report",
            "=====================",
            f"\nDataset Size:",
            f"  Total Recordings: {analytics['total_recordings']}",
            f"  Unique Sentences: {analytics['total_unique_sentences']}",
            f"  Vocabulary Size: {analytics['vocabulary_size']}",
            f"\nAverage Metrics:",
            f"  Words per Sentence: {analytics['average_sentence_length']:.1f}",
            f"  Unique Phonemes: {analytics['unique_phonemes']}",
            "\nMost Common Words:",
        ]
        
        for word, count in analytics['most_common_words']:
            report.append(f"  {word}: {count}")
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
        
        return report_text

def analyze_recording_quality(audio_dir: str, output_file: Optional[str] = None) -> Dict[str, any]:
    """Analyze quality of all recordings in a directory"""
    qc = AudioQualityControl()
    results = []
    
    for audio_file in Path(audio_dir).glob("*.wav"):
        result = qc.analyze_recording(str(audio_file))
        results.append(result)
    
    # Calculate summary statistics
    quality_scores = [r['quality_score'] for r in results if 'quality_score' in r]
    summary = {
        'total_recordings': len(results),
        'average_quality_score': np.mean(quality_scores) if quality_scores else 0,
        'recordings_with_issues': sum(1 for r in results if r.get('issues')),
        'common_issues': Counter([issue for r in results 
                                for issue in r.get('issues', [])]).most_common(),
        'recordings': results
    }
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    return summary 