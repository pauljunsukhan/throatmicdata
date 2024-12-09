"""
Audio quality analysis module.
"""

import os
from typing import Dict, List, Optional, Tuple
import json
import csv
from datetime import datetime
import numpy as np
import librosa
import soundfile as sf

class AudioQualityControl:
    """Audio quality control class for analyzing individual recordings."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.min_duration = 8.0  # minimum duration in seconds
        self.max_duration = 12.0  # maximum duration in seconds
        self.min_amplitude = -50  # minimum amplitude in dB
        self.max_amplitude = 0    # maximum amplitude in dB
        self.silence_threshold = -40  # silence threshold in dB
        
    def analyze_recording(self, audio_file: str) -> Dict:
        """Analyze a single audio recording for quality metrics."""
        try:
            # Load audio file
            y, sr = librosa.load(audio_file, sr=self.sample_rate)
            
            # Calculate duration
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Calculate amplitude statistics
            amplitude_db = librosa.amplitude_to_db(np.abs(y), ref=np.max)
            mean_amplitude = np.mean(amplitude_db)
            max_amplitude = np.max(amplitude_db)
            
            # Calculate silence ratio
            silence_mask = amplitude_db < self.silence_threshold
            silence_ratio = np.sum(silence_mask) / len(y)
            
            # Calculate SNR
            signal = y[~silence_mask] if np.any(~silence_mask) else y
            noise = y[silence_mask] if np.any(silence_mask) else np.zeros_like(y)
            signal_power = np.mean(signal**2)
            noise_power = np.mean(noise**2) if np.any(noise) else 1e-10
            snr = 10 * np.log10(signal_power / noise_power)
            
            print(f"\nAnalyzing {os.path.basename(audio_file)}:")
            print(f"  Duration: {duration:.1f}s")
            print(f"  Mean amplitude: {mean_amplitude:.1f}dB")
            print(f"  Max amplitude: {max_amplitude:.1f}dB")
            print(f"  Silence ratio: {silence_ratio*100:.1f}%")
            print(f"  SNR: {snr:.1f}dB")
            
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
            
        if silence_ratio > 0.3:
            issues.append(f'Too much silence ({silence_ratio*100:.1f}% > 30%)')
            
        if snr < 10:
            issues.append(f'Poor signal-to-noise ratio ({snr:.1f}dB < 10dB)')
            
        return issues

class DatasetAnalytics:
    """Analytics class for the entire dataset."""
    
    def __init__(self):
        self.quality_control = AudioQualityControl()
        
    def analyze_dataset(self, metadata_file: str) -> Dict:
        """Analyze all recordings in the dataset."""
        try:
            with open(metadata_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
            total_recordings = len(rows)
            total_duration = 0
            quality_issues = []
            recordings_with_issues = 0
            
            # Sentence complexity metrics
            sentence_stats = {
                'total_words': [],
                'has_comma': 0,
                'has_subordinating_conj': 0,
                'has_coordinating_conj': 0,
                'has_relative_pronouns': 0
            }
            
            # Define markers for analysis
            subordinating_conj = {'because', 'although', 'though', 'unless', 'while', 'whereas',
                               'if', 'since', 'before', 'after', 'as', 'when', 'whenever', 'where',
                               'wherever', 'whether', 'which', 'who', 'whoever', 'whom', 'whose'}
            coordinating_conj = {'and', 'but', 'or', 'nor', 'for', 'yet', 'so'}
            relative_pronouns = {'that', 'which', 'who', 'whom', 'whose', 'where', 'when'}
            
            print(f"\nAnalyzing {total_recordings} recordings...")
            for i, row in enumerate(rows, 1):
                print(f"\rProgress: {i}/{total_recordings} ({i/total_recordings*100:.1f}%)", end='', flush=True)
                
                # Audio quality analysis
                audio_file = row['audio_filepath']
                if os.path.exists(audio_file):
                    analysis = self.quality_control.analyze_recording(audio_file)
                    if 'error' in analysis:
                        quality_issues.append({
                            'file': audio_file,
                            'issues': analysis['quality_issues']
                        })
                        recordings_with_issues += 1
                    else:
                        total_duration += analysis['duration']
                        if analysis['quality_issues']:
                            quality_issues.append({
                                'file': audio_file,
                                'issues': analysis['quality_issues']
                            })
                            recordings_with_issues += 1
                
                # Sentence complexity analysis
                text = row['text']
                words = text.split()
                sentence_stats['total_words'].append(len(words))
                
                if ',' in text:
                    sentence_stats['has_comma'] += 1
                    
                if any(conj in text.lower() for conj in subordinating_conj):
                    sentence_stats['has_subordinating_conj'] += 1
                    
                if any(conj in text.lower() for conj in coordinating_conj):
                    sentence_stats['has_coordinating_conj'] += 1
                    
                if any(pron in text.lower() for pron in relative_pronouns):
                    sentence_stats['has_relative_pronouns'] += 1
            
            print("\nAnalysis complete!")
            
            # Calculate sentence complexity metrics
            avg_words = np.mean(sentence_stats['total_words'])
            min_words = np.min(sentence_stats['total_words'])
            max_words = np.max(sentence_stats['total_words'])
            
            # Calculate percentages
            pct_with_comma = sentence_stats['has_comma'] / total_recordings * 100
            pct_with_sub_conj = sentence_stats['has_subordinating_conj'] / total_recordings * 100
            pct_with_coord_conj = sentence_stats['has_coordinating_conj'] / total_recordings * 100
            pct_with_rel_pron = sentence_stats['has_relative_pronouns'] / total_recordings * 100
            
            return {
                'total_recordings': total_recordings,
                'total_duration_minutes': total_duration / 60,
                'recordings_with_issues': recordings_with_issues,
                'quality_issues': quality_issues,
                'sentence_stats': {
                    'avg_words': float(avg_words),
                    'min_words': int(min_words),
                    'max_words': int(max_words),
                    'pct_with_comma': float(pct_with_comma),
                    'pct_with_subordinating_conj': float(pct_with_sub_conj),
                    'pct_with_coordinating_conj': float(pct_with_coord_conj),
                    'pct_with_relative_pronouns': float(pct_with_rel_pron)
                }
            }
            
        except Exception as e:
            return {
                'error': str(e)
            }
    
    def generate_report(self, metadata_file: str, output_file: Optional[str] = None) -> str:
        """Generate a detailed report of the dataset analysis."""
        analysis = self.analyze_dataset(metadata_file)
        
        if 'error' in analysis:
            report = f"Error analyzing dataset: {analysis['error']}"
        else:
            report = f"""Dataset Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Summary:
- Total Recordings: {analysis['total_recordings']}
- Total Duration: {analysis['total_duration_minutes']:.1f} minutes
- Recordings with Issues: {analysis['recordings_with_issues']} ({analysis['recordings_with_issues']/analysis['total_recordings']*100:.1f}%)

Quality Issues:
"""
            for issue in analysis['quality_issues']:
                report += f"\n{os.path.basename(issue['file'])}:"
                for problem in issue['issues']:
                    report += f"\n  - {problem}"
                    
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
                
        return report